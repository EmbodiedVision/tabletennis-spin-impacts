""" 
Generate a rotation-augmented dataset 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""

import math
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R

from ball_model import DATA_DIR
from ball_model.data.hdf_utils import load_dataset, write_dataset


def rotate_single(positions, n_aug_per_traj, seed):
    rotated_positions_list = []
    angle_list = []

    min_z = np.argmin(positions[:, 2])
    anchor_point = positions[min_z, :]
    rel_positions = positions - anchor_point

    rng = np.random.RandomState(seed)
    for aug_idx in range(n_aug_per_traj):
        if aug_idx == 0:
            angle = 0
        else:
            angle = (2 * rng.rand() - 1) * math.pi
        rot_mat = R.from_rotvec([0, 0, angle]).as_matrix()
        rotated_rel_positions = (rot_mat @ rel_positions.T).T
        rotated_positions = rotated_rel_positions + anchor_point
        rotated_positions_list.append(rotated_positions)
        angle_list.append(angle)

    return rotated_positions_list, angle_list


def augment_trajectory_data(trajectories, n_aug_per_traj):
    # Augment train split only
    augmented_trajectories = []
    for trajectory in trajectories:
        trajectory = deepcopy(trajectory)
        trajectory.data_group = trajectory.data_group + "aug"

        if trajectory.split == "train":
            # augment train set
            rotated_positions, angles = rotate_single(
                trajectory.positions, n_aug_per_traj, seed=int(trajectory.index)
            )
            ctr = 0
            for positions, angle in zip(rotated_positions, angles):
                aug_traj = deepcopy(trajectory)
                aug_traj.positions = positions
                assert len(aug_traj.side_info) == 7
                aug_traj.side_info[0] = np.sin(angle)
                aug_traj.side_info[1] = np.cos(angle)
                aug_traj.index = trajectory.index + "." + str(ctr)
                ctr += 1
                augmented_trajectories.append(aug_traj)
        else:
            augmented_trajectories.append(trajectory)

    return augmented_trajectories


if __name__ == "__main__":
    for dataset in ["lp7"]:
        trajectories = load_dataset(
            DATA_DIR.joinpath(f"{dataset}.hdf5")
        )
        augmented_trajectories = augment_trajectory_data(
            trajectories, n_aug_per_traj=20
        )
        write_dataset(
            augmented_trajectories,
            DATA_DIR.joinpath(f"{dataset}aug.hdf5"),
        )
