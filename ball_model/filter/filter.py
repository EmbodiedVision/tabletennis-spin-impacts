""" 
Sequence filter, leveraging filter cells 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""

import torch

from ball_model.data import Trajectory


class SequenceFilter:
    def __init__(self, filter_cell):
        self.filter_cell = filter_cell

    def filter_sequence_batch(
        self,
        trajectory_or_batch,
        use_side_info_mask,
        filter_length,
        compute_loss,
        beginning_of_trajectory,
    ):
        # measurements: T x B x m
        # frame_indices: T x B, (-1) for non-available data
        # filter_length: number of observations to include for filtering (predict rest)

        if isinstance(trajectory_or_batch, Trajectory):
            trajectory_batch = trajectory_or_batch.as_batch()
        else:
            trajectory_batch = trajectory_or_batch

        device = self.filter_cell.device
        dtype = self.filter_cell.dtype

        measurements = torch.Tensor(trajectory_batch.positions).to(
            device=device, dtype=dtype
        )
        frame_indices = torch.Tensor(trajectory_batch.frame_indices).to(
            device=device, dtype=torch.long
        )
        first_up_frames = torch.Tensor(trajectory_batch.first_up_frames).to(
            device=device, dtype=torch.long
        )
        side_info = torch.Tensor(trajectory_batch.side_info).to(
            device=device, dtype=dtype
        )

        if filter_length is None:
            filter_length = len(measurements)
        else:
            assert 2 <= filter_length <= len(measurements)

        initial_prior, initial_posterior = self.filter_cell.initialize_filter(
            measurements,
            frame_indices,
            first_up_frames,
            side_info,
            use_side_info_mask,
            beginning_of_trajectory,
        )
        assert len(initial_prior) == len(initial_posterior)

        # start prediction from last posterior
        state = initial_posterior[-1]

        bs = measurements.shape[1]
        device = measurements.device
        predicted_states = list(initial_prior)
        corrected_states = list(initial_posterior)

        # compute loss for initial states (required in VRNN)
        metrics = {}
        if compute_loss:
            for step in range(len(initial_prior)):
                state_pred = predicted_states[step]
                state_corr = corrected_states[step]
                step_metrics = self.filter_cell.loss(
                    state_pred, state_corr, measurements[step, :, :]
                )
                for metric_name, value in step_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = torch.zeros(bs, device=device)
                    metrics[metric_name] += value

        local_masks = []

        traj_len = torch.zeros(bs, device=device)

        for step in range(len(initial_prior), len(measurements)):
            global_mask_prev = frame_indices[step - 1, :] >= 0
            global_mask = frame_indices[step, :] >= 0

            if not torch.any(global_mask):
                break

            pred_steps = frame_indices[step, :] - frame_indices[step - 1, :]
            local_mask = global_mask[global_mask_prev]
            local_masks.append(local_mask)

            state_pred = self.filter_cell.predict(
                state[local_mask], pred_steps[global_mask]
            )
            predicted_states.append(state_pred)

            if step < filter_length:
                state_corr = self.filter_cell.correct(
                    state_pred, measurements[step, global_mask, :]
                )
                state = state_corr
            else:
                state_corr = None
                state = state_pred

            corrected_states.append(state_corr)

            if compute_loss:
                step_metrics = self.filter_cell.loss(
                    state_pred, state_corr, measurements[step, global_mask, :]
                )
                for metric_name, value in step_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = torch.zeros(bs, device=device)
                    metrics[metric_name][global_mask] += value

            traj_len[global_mask] += 1

        for k, v in metrics.items():
            metrics[k] = v / traj_len

        return predicted_states, corrected_states, traj_len, metrics

    def filter_sequence_batch_varying_length(
        self,
        trajectory_or_batch,
        use_side_info_mask,
        filter_length_list,
        beginning_of_trajectory,
    ):
        # measurements: T x B x m
        # frame_indices: T x B, nan for non-available data
        # filter_length: number of observations to include for filtering (predict rest)

        if isinstance(trajectory_or_batch, Trajectory):
            trajectory_batch = trajectory_or_batch.as_batch()
        else:
            trajectory_batch = trajectory_or_batch

        device = self.filter_cell.device
        dtype = self.filter_cell.dtype

        measurements = torch.Tensor(trajectory_batch.positions).to(
            device=device, dtype=dtype
        )
        frame_indices = torch.Tensor(trajectory_batch.frame_indices).to(
            device=device, dtype=torch.long
        )
        first_up_frames = torch.Tensor(trajectory_batch.first_up_frames).to(
            device=device, dtype=torch.long
        )
        side_info = torch.Tensor(trajectory_batch.side_info).to(
            device=device, dtype=dtype
        )

        assert len(filter_length_list) == trajectory_batch.batchsize
        assert all(2 <= l <= len(measurements) for l in filter_length_list)

        initial_prior, initial_posterior = self.filter_cell.initialize_filter(
            measurements,
            frame_indices,
            first_up_frames,
            side_info,
            use_side_info_mask,
            beginning_of_trajectory,
        )
        # initial_prior/posterior: T_0: BxS, T_1: BxS, ...
        assert len(initial_prior) == len(initial_posterior)

        # start prediction from last posterior
        state = initial_posterior[-1]

        bs = measurements.shape[1]
        device = measurements.device
        predicted_states = list(initial_prior)
        corrected_states = list(initial_posterior)
        all_states = list(initial_posterior)

        traj_len = torch.zeros(bs, device=device)

        for step in range(len(initial_prior), len(measurements)):
            global_mask = frame_indices[step, :] >= 0

            if not torch.any(global_mask):
                break

            pred_steps = frame_indices[step, :] - frame_indices[step - 1, :]

            state_pred = self.filter_cell.predict(
                state[global_mask], pred_steps[global_mask]
            )
            state_pred.expand_invalid(valid_mask=global_mask)
            predicted_states.append(state_pred)

            in_filter_phase = (
                torch.Tensor([step < l for l in filter_length_list]).bool().to(device)
            )
            states_to_correct = global_mask & in_filter_phase
            state_corr = self.filter_cell.correct(
                state_pred[states_to_correct], measurements[step, states_to_correct, :]
            )
            state_corr.expand_invalid(valid_mask=states_to_correct)
            corrected_states.append(state_corr)

            state = state_pred.clone_overwrite(state_corr, states_to_correct)
            all_states.append(state)

            traj_len[global_mask] += 1

        return predicted_states, corrected_states, all_states, traj_len
