""" 
A general filter cell 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import json

import torch
from torch import nn

from ball_model.data import IDEAL_TABLE, MEAS_DT, RADIUS_OF_BALL


class FilterCell(nn.Module):
    def __init__(self):
        super(FilterCell, self).__init__()
        # dummy variable for inferring the device and dtype
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def dtype(self):
        return self.dummy.dtype

    @property
    def device(self):
        return self.dummy.device

    def predict(self, state, num_steps, with_uncertainty=True):
        if isinstance(num_steps, int):
            num_steps = torch.ones(state.batch_dim).long() * num_steps

        assert num_steps.dim() == 1
        assert num_steps.shape[0] == state.batch_dim
        assert torch.all(num_steps > 0)
        max_steps = torch.max(num_steps)
        if all([s == num_steps[0] for s in num_steps]):
            for step in range(1, max_steps + 1):
                state = self._predict_singlestep(
                    state, with_uncertainty=with_uncertainty
                )
        else:
            for step in range(1, max_steps + 1):
                mask = num_steps >= step
                state_pred = self._predict_singlestep(
                    state[mask], with_uncertainty=with_uncertainty
                )
                state_pred.expand_invalid(valid_mask=mask)
                state = state.clone_overwrite(state_pred, mask)
        return state


def load_filter_cell(filter_cell_type, filter_cell_kwargs, device):
    from ball_model.filter.ekf.ekf_cell import EkfFilterCell

    filter_cell_kwargs = dict(**filter_cell_kwargs)

    if filter_cell_type == "ekf":
        if "state_dependent_Q" in filter_cell_kwargs:
            del filter_cell_kwargs["state_dependent_Q"]
        if "state_dependent_R" in filter_cell_kwargs:
            del filter_cell_kwargs["state_dependent_R"]
        if "kd_km_offset" not in filter_cell_kwargs:
            filter_cell_kwargs["kd_km_offset"] = 0.05
        if filter_cell_kwargs["impact_model_classname"] == "NNImpactModel":
            filter_cell_kwargs["impact_model_classname"] = "MatrixImpactModel"
            filter_cell_kwargs["impact_model_kwargs"] = {}

        filter_cell = EkfFilterCell(
            dt=MEAS_DT,
            table_z=IDEAL_TABLE[2],
            gravity_z=9.802,
            radius_of_ball=RADIUS_OF_BALL,
            **filter_cell_kwargs,
        ).to(device)
    else:
        raise ValueError

    return filter_cell


def load_filter_cell_from_run(run_directory, checkpoint_step, device):
    with open(run_directory.joinpath("config.json"), "r") as handle:
        config = json.load(handle)

    filter_cell_type = config["filter_cell_type"]
    filter_cell_kwargs = config["filter_cell_kwargs"]

    filter_cell = load_filter_cell(filter_cell_type, filter_cell_kwargs, device)

    if checkpoint_step is not None:
        ckpt_dir = run_directory.joinpath("checkpoints")
        checkpoint_data = torch.load(
            ckpt_dir.joinpath(f"step_{checkpoint_step}.pkl"), map_location=device
        )

        filter_cell.load_state_dict(checkpoint_data["filter_cell"])

    return filter_cell
