""" 
A state belief, which is the result of filtering 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from typing import List

import torch
from torch import distributions


class ExtendedIndependent(distributions.Independent):
    def __init__(
        self, base_distribution, reinterpreted_batch_ndims, validate_args=None
    ):
        self._validate_args = validate_args
        super(ExtendedIndependent, self).__init__(
            base_distribution, reinterpreted_batch_ndims, validate_args
        )

    @property
    def shape(self):
        return self.mean.shape

    def __getitem__(self, item):
        return ExtendedIndependent(
            self.base_dist[item], self.reinterpreted_batch_ndims, self._validate_args
        )

    def __setitem__(self, item, value):
        self.base_dist[item] = value.base_dist

    def clone(self):
        return ExtendedIndependent(
            self.base_dist.clone(), self.reinterpreted_batch_ndims, self._validate_args
        )


class ExtendedNormal(distributions.Normal):
    def __init__(self, loc, scale, validate_args=None):
        self._validate_args = validate_args
        super(ExtendedNormal, self).__init__(loc, scale, validate_args)

    def __getitem__(self, item):
        return ExtendedNormal(self.loc[item], self.scale[item], self._validate_args)

    def __setitem__(self, item, value):
        self.loc[item] = value.loc
        self.scale[item] = value.scale

    def clone(self):
        return ExtendedNormal(self.loc.clone(), self.scale.clone(), self._validate_args)


class StateBelief:
    def __init__(self, tensor_dict, state_type):
        self.any_tensor = None
        for tensor in tensor_dict.values():
            if tensor is not None:
                self.any_tensor = tensor
                break
        assert self.any_tensor is not None
        if isinstance(state_type, str):
            assert state_type in {"predicted", "corrected"}
            state_type = [
                state_type,
            ] * self.any_tensor.shape[0]
        else:
            assert type(state_type) == list
            n_valid = sum(t != "invalid" for t in state_type)
            for t in tensor_dict.values():
                if t is not None:
                    assert t.shape[0] == n_valid
        self.tensor_dict = tensor_dict

        assert all(t in {"predicted", "corrected", "invalid"} for t in state_type)
        self.state_type = (
            state_type  # list of length B, can be "predicted, corrected, invalid"
        )

    @property
    def batch_dim(self):
        return len(self.state_type)

    @property
    def valid_batch_dim(self):
        return self.any_tensor.shape[0]

    @property
    def valid_list(self) -> List[bool]:
        return [t != "invalid" for t in self.state_type]

    @property
    def valid_mask(self) -> torch.Tensor:
        return torch.Tensor(self.valid_list).to(self.any_tensor.device).bool()

    @property
    def valid_state_types(self):
        return self.state_type[self.valid_mask]

    def idx_to_valid_idx(self, idx):
        assert 0 <= idx < len(self.state_type)
        assert self.state_type[idx] != "invalid"
        valid_idx = torch.cumsum(self.valid_mask, dim=0)[idx].item() - 1
        return valid_idx

    def expand_invalid(self, *, valid_mask: torch.Tensor):
        # expand the belief to match the size of 'valid_mask', with valid states given by '1's in the mask
        if not sum(self.valid_list) == sum(valid_mask).item():
            raise ValueError("Cannot change number of *valid* states")
        if len(valid_mask) < len(self.state_type):
            raise ValueError("Can only expand")
        elif len(valid_mask) == len(self.state_type) and not torch.equal(
            valid_mask, self.valid_mask
        ):
            raise ValueError("Changing state type not allowed")

        valid_ctr = 0
        new_state_type = []
        for is_valid in valid_mask:
            if is_valid:
                new_state_type.append(self.state_type[valid_ctr])
                valid_ctr += 1
            else:
                new_state_type.append("invalid")

        self.state_type = new_state_type

    def __getattr__(self, item):
        return self.tensor_dict[item]

    def __getitem__(self, item):
        # We assume that item is a boolean mask, for __all__ states
        # Throw an error if an invalid state is selected
        # Return a new state belief without any invalid states
        if isinstance(item, int):
            mask = torch.zeros(self.batch_dim).bool()
            mask[item] = True
            item = mask

        assert isinstance(item, torch.Tensor)
        assert item.dim() == 1
        assert item.dtype == torch.bool
        assert len(item) == len(self.state_type)

        selected_invalid = (~self.valid_mask) & item
        if torch.any(selected_invalid):
            raise ValueError("Invalid states cannot be selected")

        # Mask flagging, for all valid items, the items which are selected
        selected_valid = item[self.valid_mask]
        new_tensor_dict = {
            k: (v[selected_valid] if v is not None else None)
            for k, v in self.tensor_dict.items()
        }

        # Mask flagging the items which are selected among all items (including invalid)
        selected_all = item & self.valid_mask
        type_masked = [t for m, t in zip(selected_all, self.state_type) if m]

        return self.__class__.from_tensor_dict(new_tensor_dict, type_masked)

    def clone_overwrite(self, src_belief, to_overwrite_mask):
        """
        Overwrite states in this belief by states in `src_belief`, with mask given by `to_overwrite_mask`.
        Only states which are valid in this belief can be overwritten.
        It is asserted that all valid states in src_belief end up in this state.
        """
        assert len(to_overwrite_mask) == self.batch_dim
        assert len(to_overwrite_mask) == src_belief.batch_dim
        assert sum(to_overwrite_mask).item() == src_belief.valid_batch_dim

        valid_here = self.valid_mask
        if torch.any(to_overwrite_mask & (~valid_here)):
            raise ValueError(
                "Only states which are valid in this belief can be overwritten"
            )

        valids_to_overwrite = to_overwrite_mask[self.valid_mask]

        new_tensor_dict = {}
        for name, tensor in self.tensor_dict.items():
            if tensor is None or src_belief.tensor_dict[name] is None:
                clone = None
            else:
                clone = tensor.clone()
                clone[valids_to_overwrite] = src_belief.tensor_dict[name]
            new_tensor_dict[name] = clone

        new_state_type = []
        for idx in range(len(to_overwrite_mask)):
            if to_overwrite_mask[idx]:
                new_state_type.append(src_belief.state_type[idx])
            else:
                new_state_type.append(self.state_type[idx])

        return self.__class__.from_tensor_dict(new_tensor_dict, new_state_type)

    @classmethod
    def concatenate(cls, state_batch_list):
        from collections import defaultdict

        self = cls.__new__(cls)
        tensor_dict_list = defaultdict(list)
        state_type_list = []
        for state_batch in state_batch_list:
            for key, val in state_batch.tensor_dict.items():
                tensor_dict_list[key].append(val)
            state_type_list.extend(state_batch.state_type)

        tensor_dict = {}
        for key, value in tensor_dict_list.items():
            if hasattr(value[0].__class__, "concat"):
                tensor_dict[key] = value[0].__class__.concat(value)
            else:
                tensor_dict[key] = torch.concat(value, dim=0)

        self.tensor_dict = tensor_dict
        self.state_type = state_type_list

        return self
