""" 
Utils for browsing and loading trajectories 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import h5py
import numpy as np
import torch

from ball_model import DATA_DIR

# x, y, z coordinate of table center
IDEAL_TABLE = np.array(
    [0.1325, 1.7425, -0.461]
)
RADIUS_OF_BALL = 0.02

# measurement frequency is 180Hz
MEAS_DT = 1 / 180

HDF_CACHE = None
HDF_CACHE_DATAGROUP = None


class Trajectory:
    """
    A single trajectory
    """

    def __init__(
        self,
        frame_indices,
        positions,
        side_info,
        timestamps=None,
        first_up_frame=-1,
    ):
        assert frame_indices.ndim == 1
        assert positions.ndim == 2
        assert frame_indices.shape[0] == positions.shape[0]
        assert positions.shape[1] == 3
        if timestamps is not None:
            assert timestamps.ndim == 1
            assert timestamps.shape[0] == positions.shape[0]

        self.frame_indices = frame_indices
        self.positions = positions
        self.side_info = side_info
        self.timestamps = timestamps
        # Frame number (as in frame_indices) of first measurement after impact
        self.first_up_frame = first_up_frame

    @property
    def first_up_idx(self):
        # Index of first measurement after impact
        if self.first_up_frame == -1:
            return -1
        else:
            return np.nonzero(self.frame_indices == self.first_up_frame)[0]

    def as_batch(self):
        first_up_frames = np.array([self.first_up_frame])
        if isinstance(self.frame_indices, np.ndarray):
            len_batch = np.ones(1).astype(int) * len(self.timestamps)
        else:
            len_batch = torch.ones(1).long() * len(self.timestamps)
            first_up_frames = torch.LongTensor(first_up_frames)
        return TrajectoryBatch(
            self.frame_indices[:, None],
            self.positions[:, None, :],
            self.side_info[None, :],
            first_up_frames,
            len_batch,
        )

    def __len__(self):
        return len(self.frame_indices)


class TrajectoryBatch:
    """
    A batch of rajectories
    """

    def __init__(self, frame_indices, positions, side_info, first_up_frames, length):
        self.frame_indices = frame_indices  # T x B
        self.positions = positions  # T x B x 3
        self.side_info = side_info  # B x m
        self.first_up_frames = first_up_frames  # B
        self.length = length  # B

    def __repr__(self):
        return f"t: {self.frame_indices.shape}, p: {self.positions.shape}, s: {self.side_info.shape}, l: {self.length.shape}"

    @property
    def batchsize(self):
        return self.frame_indices.shape[1]

    @staticmethod
    def from_list(trajectory_list, equal_length=False):
        if isinstance(trajectory_list[0].positions, torch.Tensor):
            device = trajectory_list[0].positions.device
            dtype = trajectory_list[0].positions.dtype
        else:
            device = "cpu"
            dtype = torch.float64

        len_list = [len(t) for t in trajectory_list]
        len_tensor = torch.Tensor(len_list).long().to(device)

        first_up_frame_list = [t.first_up_frame for t in trajectory_list]
        first_up_frame_tensor = torch.Tensor(first_up_frame_list).long().to(device)

        if equal_length:
            assert all(l == len_list[0] for l in len_list[1:])
            frame_idx_tensor = torch.stack(
                [t.frame_indices for t in trajectory_list], dim=1
            )
            pos_tensor = torch.stack([t.positions for t in trajectory_list], dim=1)
            side_info_tensor = torch.stack(
                [t.side_info for t in trajectory_list], dim=0
            )
            return TrajectoryBatch(
                frame_idx_tensor,
                pos_tensor,
                side_info_tensor,
                first_up_frame_tensor,
                len_tensor,
            )
        else:
            max_len = max(len_list)
            frame_idx_tensor = (-1) * torch.ones(max_len, len(trajectory_list)).to(
                device, dtype=torch.long
            )
            pos_tensor = np.nan * torch.ones(max_len, len(trajectory_list), 3).to(
                device, dtype=dtype
            )
            assert trajectory_list[0].side_info.ndim == 1
            side_info_dim = trajectory_list[0].side_info.shape[0]
            side_info_tensor = np.nan * torch.ones(
                len(trajectory_list), side_info_dim
            ).to(device, dtype=dtype)
            for idx in range(len(trajectory_list)):
                frame_idx_tensor[: len_list[idx], idx] = torch.Tensor(
                    trajectory_list[idx].frame_indices
                ).to(device, dtype=torch.long)
                pos_tensor[: len_list[idx], idx, :] = torch.Tensor(
                    trajectory_list[idx].positions
                ).to(device, dtype=dtype)
                side_info_tensor[idx] = torch.Tensor(trajectory_list[idx].side_info).to(
                    device, dtype=dtype
                )
            return TrajectoryBatch(
                frame_idx_tensor,
                pos_tensor,
                side_info_tensor,
                first_up_frame_tensor,
                len_tensor,
            )

    def to(self, device=None, dtype=None):
        return TrajectoryBatch(
            self.frame_indices.to(device=device, dtype=dtype),
            self.positions.to(device=device, dtype=dtype),
            self.side_info.to(device=device, dtype=dtype),
            self.first_up_frames.to(device=device),
            self.length.to(device=device),
        )

    def __getitem__(self, item):
        # item: time_slice, batch_slice
        assert isinstance(item, tuple)
        assert len(item) == 2
        if isinstance(item[1], int):
            return Trajectory(
                self.frame_indices[item[0], item[1]],
                self.positions[item[0], item[1], :],
                self.side_info[item[1], :],
                first_up_frame=self.first_up_frames[item[1]],
            )
        else:
            return TrajectoryBatch(
                self.frame_indices[item[0], item[1]],
                self.positions[item[0], item[1], :],
                self.side_info[item[1], :],
                self.first_up_frames[item[1]],
                self.length[item[1]],
            )


def _get_hdf_file(data_group):
    global HDF_CACHE
    global HDF_CACHE_DATAGROUP

    if HDF_CACHE_DATAGROUP != data_group:
        if HDF_CACHE:
            HDF_CACHE.close()
        file_path = DATA_DIR.joinpath(data_group + ".hdf5")
        HDF_CACHE = h5py.File(file_path, "r")
        HDF_CACHE_DATAGROUP = data_group

    return HDF_CACHE


def get_data(traj_fullname):
    """
    Get a `Trajectory` object by the trajectories name, e.g. lp7%7000
    """
    data_group = traj_fullname.split("%")[0]
    hdf_file = _get_hdf_file(data_group)
    group = hdf_file["processed"]

    timestamps = np.array(group[traj_fullname]["timestamps"])
    frame_indices = np.array(group[traj_fullname]["frame_indices"])
    positions = np.array(group[traj_fullname]["positions"])
    first_up_idx = group[traj_fullname].attrs["first_up_idx"]
    side_info = np.array(group[traj_fullname]["side_info"])

    traj_idx = traj_fullname.split("%")[1].split(".")[0]
    assert len(traj_idx) == 4

    # For the 'unseen' trajectories (lp9X),
    # frame orientation is not available.
    if traj_idx.startswith("9"):
        assert len(side_info) == 7
        side_info[0] = np.nan
        side_info[1] = np.nan

    if first_up_idx != -1:
        first_up_frame = frame_indices[first_up_idx]
    else:
        first_up_frame = -1

    return Trajectory(
        frame_indices,
        positions,
        side_info,
        timestamps=timestamps,
        first_up_frame=first_up_frame,
    )


def iter_traj(data_group, split_name="all"):
    """
    Get all available trajectories for a particular 'data group' (launcher orientation),
    filtered by its split (train/val/test).
    """
    hdf_file = _get_hdf_file(data_group)
    group = hdf_file["processed"]

    found = False
    for traj_fullname in list(group.keys()):
        traj_split_name = group[traj_fullname].attrs["split"]
        traj_data_group = traj_fullname.split("%")[0]
        assert traj_data_group == data_group, f"{traj_data_group} != {data_group}"
        if split_name == "all":
            found = True
            yield traj_fullname, traj_split_name
        elif traj_split_name == split_name:
            found = True
            yield traj_fullname

    if not found:
        raise RuntimeError(f"No trajectory was found for {data_group}!")
