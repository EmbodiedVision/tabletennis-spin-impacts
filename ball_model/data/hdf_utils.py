""" 
Utils for reading and writing HDF trajectory storage 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""

import h5py
import numpy as np


class HdfTrajectory:
    def __init__(
        self,
        data_group,
        index,
        timestamps,
        frame_indices,
        positions,
        first_up_idx,
        side_info,
        split,
    ):
        self.data_group = data_group
        self.index = index
        self.timestamps = timestamps
        self.frame_indices = frame_indices
        self.positions = positions
        self.first_up_idx = first_up_idx
        self.side_info = side_info
        self.split = split


def write_dataset(trajectories, file_path):
    print(f"Writing dataset {file_path}")
    splits = [t.split for t in trajectories]
    split_names, split_counts = np.unique(splits, return_counts=True)
    print(split_names, split_counts, split_counts / np.sum(split_counts))

    with h5py.File(file_path, "w") as hdf_file:
        group = hdf_file.create_group("processed")
        for trajectory in trajectories:
            traj = group.create_group(trajectory.data_group + "%" + trajectory.index)
            traj["timestamps"] = trajectory.timestamps
            traj["frame_indices"] = trajectory.frame_indices
            traj["positions"] = trajectory.positions
            traj["side_info"] = trajectory.side_info
            traj.attrs["first_up_idx"] = trajectory.first_up_idx
            traj.attrs["split"] = trajectory.split


def load_dataset(file_path):
    trajectories = []

    with h5py.File(file_path, "r") as hdf_file:
        group = hdf_file["processed"]

        for traj_fullname in list(group.keys()):
            trajectory = HdfTrajectory(
                data_group=traj_fullname.split("%")[0],
                index=traj_fullname.split("%")[1],
                timestamps=np.array(group[traj_fullname]["timestamps"]),
                frame_indices=np.array(group[traj_fullname]["frame_indices"]),
                positions=np.array(group[traj_fullname]["positions"]),
                first_up_idx=group[traj_fullname].attrs["first_up_idx"],
                side_info=np.array(group[traj_fullname]["side_info"]),
                split=group[traj_fullname].attrs["split"],
            )

            trajectories.append(trajectory)

    return trajectories
