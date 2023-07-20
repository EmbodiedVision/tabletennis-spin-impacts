""" 
Train the model 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import os
from time import time

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ball_model import EXPERIMENT_DIR
from ball_model.data import TrajectoryBatch, get_data, iter_traj
from ball_model.filter.filter import SequenceFilter
from ball_model.filter.filter_cell import load_filter_cell
from ball_model.evaluation import evaluate


experiment = Experiment("train_model")
run_basedir = EXPERIMENT_DIR
os.makedirs(run_basedir, exist_ok=True)
experiment.observers.append(FileStorageObserver(run_basedir))


@experiment.config
def config():
    filter_cell_type = "ekf"
    n_epochs = 10_000
    train_data_groups = ["lp7aug"]
    eval_data_groups = ["lp7aug"]
    device = "cpu"
    batchsize = 64
    chunk_size = 50
    evaluate_every = 500
    checkpoint_every = 100
    multistep_loss = False
    adam_lr = 5e-3
    use_side_info = "always"


@experiment.named_config
def ekf_default_args():
    filter_cell_kwargs = {
        "jacobian_type": "manual",
        "impact_model_classname": "MatrixImpactModel",
        "impact_model_kwargs": {},
        "kd_km_offset": 0.05
    }


def load_trajectories(train_data_groups, chunk_size, device):
    # Load all trajectories, store as [t, x, y, z]
    # Pad up to longest trajectory
    # warn("Training on *all* trajectories!")
    trajectories = []
    n_dropped = 0
    for data_group in train_data_groups:
        for traj_fullname in tqdm(
            list(iter_traj(data_group=data_group, split_name="train"))
        ):
            trajectory = get_data(traj_fullname)
            if len(trajectory.timestamps) >= chunk_size:
                trajectories.append(trajectory)
            else:
                n_dropped += 1

    print(
        f"Dropped {n_dropped} trajs shorter than {chunk_size}, keeping {len(trajectories)}"
    )
    trajectory_batch = TrajectoryBatch.from_list(trajectories).to(device)
    return trajectory_batch


def sample_trajectory_batch(full_trajectory_batch, batchsize, chunk_size):
    n_traj = full_trajectory_batch.batchsize
    if chunk_size is None:
        # no chunking
        if batchsize == "all":
            batch = full_trajectory_batch
        else:
            all_indices = np.arange(0, n_traj)
            np.random.shuffle(all_indices)
            batch_indices = all_indices[:batchsize]
            batch = full_trajectory_batch[:, batch_indices]
    else:
        # chunking
        if batchsize == "all":
            traj_idx_list = range(n_traj)
        else:
            traj_idx_list = np.random.randint(0, n_traj, batchsize)
        start_idx_list = [
            np.random.randint(
                0,
                int(full_trajectory_batch.length[traj_idx].cpu().item())
                - chunk_size
                + 1,
            )
            for traj_idx in traj_idx_list
        ]
        batch = TrajectoryBatch.from_list(
            [
                full_trajectory_batch[start_idx : start_idx + chunk_size, int(traj_idx)]
                for start_idx, traj_idx in zip(start_idx_list, traj_idx_list)
            ],
            equal_length=True,
        )
    return batch


@experiment.main
def run_experiment(
    n_epochs,
    train_data_groups,
    eval_data_groups,
    device,
    batchsize,
    chunk_size,
    filter_cell_type,
    filter_cell_kwargs,
    evaluate_every,
    checkpoint_every,
    use_side_info,
    multistep_loss,
    adam_lr,
    _run,
):
    run_directory = run_basedir.joinpath(_run._id)
    writer = SummaryWriter(run_directory)

    print("Loading trajectories")
    start = time()

    trajectory_data = load_trajectories(train_data_groups, chunk_size, device)
    end = time()
    print(f"Loaded {trajectory_data.batchsize} trajectories in {end-start} sec")

    filter_cell = load_filter_cell(filter_cell_type, filter_cell_kwargs, device)

    trajectory_data = trajectory_data.to(dtype=filter_cell.dtype)

    print(filter_cell)

    sequence_filter = SequenceFilter(filter_cell)

    parameters = filter_cell.parameters()

    print(filter_cell.state_dict().keys())

    optim = torch.optim.Adam(parameters, lr=adam_lr)

    for epoch in range(n_epochs):

        trajectory_batch = sample_trajectory_batch(
            trajectory_data, batchsize, chunk_size
        )

        if multistep_loss:
            filter_length = trajectory_batch.positions.shape[0] // 2
        else:
            filter_length = trajectory_batch.positions.shape[0]

        if use_side_info == "always":
            use_side_info_mask = torch.ones(batchsize).bool().to(device)
        elif use_side_info == "never":
            use_side_info_mask = torch.zeros(batchsize).bool().to(device)
        elif use_side_info == "random":
            use_side_info_mask = (torch.rand(batchsize) > 0.5).to(device)
        else:
            raise ValueError

        (
            states_pred,
            states_corr,
            traj_len,
            metrics,
        ) = sequence_filter.filter_sequence_batch(
            trajectory_batch,
            use_side_info_mask,
            filter_length,
            compute_loss=True,
            beginning_of_trajectory=False,
        )

        loss = metrics["loss"].mean()

        print(epoch, loss)

        optim.zero_grad()
        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore", category=UserWarning)
        loss.backward()
        grad_clip_norm = 1000
        nn.utils.clip_grad_norm_(filter_cell.parameters(), grad_clip_norm, norm_type=2)
        optim.step()

        if (epoch + 1) % checkpoint_every == 0 and (_run._id is not None):
            checkpoint_data = {
                "epoch": epoch + 1,
                "filter_cell": filter_cell.state_dict(),
            }
            ckpt_dir = run_directory.joinpath("checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_file = ckpt_dir.joinpath(f"step_{epoch+1}.pkl")
            torch.save(checkpoint_data, ckpt_file)

        if (epoch + 1) % evaluate_every == 0:
            eval_split_whitelist = ["val"]
            evaluate(
                run_directory,
                _run._id,
                epoch + 1,
                filter_cell,
                eval_data_groups,
                eval_split_whitelist,
                writer,
            )

        if writer:
            for metric_name, value in metrics.items():
                writer.add_scalar(
                    f"train/{metric_name}", value.mean().item(), epoch + 1
                )


if __name__ == "__main__":
    experiment.run_commandline()
