""" 
Run filter on a single trajectory 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import numpy as np
import torch

from ball_model import EXPERIMENT_DIR
from ball_model.data import MEAS_DT, get_data
from ball_model.filter.filter import SequenceFilter
from ball_model.filter.filter_cell import load_filter_cell_from_run




def main():
    run_name = "model_ekf_g_lp7aug_sialways_s1"
    checkpoint_step = 10_000
    traj_fullname = "lp7%7010"
    filter_length = 20
    use_side_info = True

    run_dir = EXPERIMENT_DIR.joinpath(run_name)
    checkpoint_dir = run_dir.joinpath("checkpoints")
    if checkpoint_step == "last":
        checkpoint_step = max(
            int(filename.stem.split("_")[1]) for filename in checkpoint_dir.iterdir()
        )
        print(f"Last checkpoint is at {checkpoint_step}")

    filter_cell = load_filter_cell_from_run(run_dir, checkpoint_step, device="cpu")
    sequence_filter = SequenceFilter(filter_cell)

    trajectory = get_data(traj_fullname)
    timestamps = trajectory.frame_indices * MEAS_DT
    positions = trajectory.positions
    use_side_info_mask = torch.Tensor((int(use_side_info) * np.ones(1))).bool()

    (
        predicted_states,
        corrected_states,
        traj_len,
        metrics,
    ) = sequence_filter.filter_sequence_batch(
        trajectory,
        use_side_info_mask,
        filter_length=filter_length,
        compute_loss=False,
        beginning_of_trajectory=True,
    )

    filtered_timestamps = timestamps[:filter_length]
    predicted_timestamps = timestamps[filter_length:]
    filtered_mean_obs = np.stack(
        [s.meas_dist.mean[0] for s in corrected_states[:filter_length]]
    )
    predicted_mean_obs = np.stack(
        [s.meas_dist.mean[0] for s in predicted_states[filter_length:]]
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(np.array(filtered_timestamps), filtered_mean_obs[:, 2])
    ax[0].plot(np.array(predicted_timestamps), predicted_mean_obs[:, 2])
    ax[0].plot(timestamps, positions[:, 2], alpha=0.5)
    ax[1].plot(
        filtered_mean_obs[:, 0],
        filtered_mean_obs[:, 1],
    )
    ax[1].plot(
        predicted_mean_obs[:, 0],
        predicted_mean_obs[:, 1],
    )
    ax[1].plot(positions[:, 0], positions[:, 1], alpha=0.5)
    ax[1].set_xlim([1, -0.7])
    plt.show()


if __name__ == "__main__":
    with torch.no_grad():
        main()
