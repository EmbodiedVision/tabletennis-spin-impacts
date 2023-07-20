""" 
Utils for evaluation 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ball_model.data import MEAS_DT, TrajectoryBatch, get_data, iter_traj
from ball_model.filter.filter import SequenceFilter


def filter_predict_batch(
    sequence_filter,
    trajectory_batch,
    use_side_info_mask,
    filter_length_list,
    beginning_of_trajectory,
):
    assert trajectory_batch.batchsize == len(filter_length_list)

    _, _, all_states, _ = sequence_filter.filter_sequence_batch_varying_length(
        trajectory_batch,
        use_side_info_mask,
        filter_length_list=filter_length_list,
        beginning_of_trajectory=beginning_of_trajectory,
    )

    traj_stats = []
    for traj_idx in tqdm(list(range(trajectory_batch.batchsize))):
        l = trajectory_batch.length[traj_idx].item()
        moments = {
            "mean": [],
            "cov": [],
        }
        for state in all_states[:l]:
            if state:
                valid_idx = state.idx_to_valid_idx(traj_idx)
                mean = state.meas_dist.mean[valid_idx].cpu().numpy()
                if hasattr(state.meas_dist, "covariance_matrix"):
                    cov = state.meas_dist.covariance_matrix[valid_idx].cpu().numpy()
                else:
                    cov = (
                        torch.diag_embed(state.meas_dist.variance[valid_idx])
                        .cpu()
                        .numpy()
                    )
            else:
                mean = np.nan * np.ones(3)
                cov = np.nan * np.ones(3, 3)

            moments["mean"].append(mean)
            moments["cov"].append(cov)

        state_mean = np.stack(moments["mean"])
        state_cov = np.stack(moments["cov"])

        filter_type = [
            {"corrected": "f", "predicted": "p"}[state.state_type[traj_idx]]
            if state
            else "f"
            for state in all_states[:l]
        ]

        positions = trajectory_batch.positions[:l, traj_idx, :].cpu().numpy()
        maxabserr = np.max(np.abs(state_mean - positions), axis=1)
        euclerr = np.sqrt(np.sum((state_mean - positions) ** 2, axis=-1))

        traj_stats.append(
            {
                "state_mean": state_mean,
                "state_cov": state_cov,
                "filter_type": filter_type,
                "maxabserr": maxabserr,
                "euclerr": euclerr,
            }
        )
    return traj_stats


def compute_evaluations(
    run_dir,
    run_name,
    checkpoint_step,
    data_group,
    split_whitelist,
    min_pred_length_sec,
    use_side_info,
    sequence_filter,
    save_to_file,
):
    # min_prediction_length_sec: filter such that we predict at least t seconds
    # generate batch of trajectories and corresponding filter lengths
    traj_meta_list = []
    trajectory_list = []
    filter_len_list = []
    for traj_fullname, traj_split in tqdm(list(iter_traj(data_group))):
        if traj_split not in split_whitelist:
            continue
        trajectory = get_data(traj_fullname)
        timestamps = trajectory.frame_indices * MEAS_DT
        secs_until_end = timestamps[-1] - timestamps
        mask = secs_until_end > min_pred_length_sec
        if sum(mask) <= 1:
            filter_length = 10
        else:
            filter_length = max(np.max(mask.nonzero()) + 1, 10)

        traj_meta_list.append(
            (traj_fullname, traj_split, len(timestamps), filter_length)
        )
        trajectory_list.append(trajectory)
        filter_len_list.append(filter_length)

    trajectory_batch = TrajectoryBatch.from_list(trajectory_list)

    assert type(use_side_info) == bool
    use_side_info_mask = (
        (int(use_side_info) * torch.ones(len(trajectory_list)))
        .bool()
        .to(sequence_filter.filter_cell.device)
    )

    filter_results = filter_predict_batch(
        sequence_filter,
        trajectory_batch,
        use_side_info_mask,
        filter_len_list,
        beginning_of_trajectory=True,
    )

    records = []
    for traj_metadata, record in zip(traj_meta_list, filter_results):
        traj_fullname, traj_split, traj_length, filter_length = traj_metadata
        record = dict(
            **record,
            **{
                "run_name": run_name,
                "checkpoint_step": checkpoint_step,
                "traj_fullname": traj_fullname,
                "traj_split": traj_split,
                "traj_length": traj_length,
                "filter_length": filter_length,
                "min_pred_length_sec": min_pred_length_sec,
                "use_side_info": use_side_info,
            },
        )
        record["maxabserr_last5"] = np.max(record["maxabserr"][-5:])
        record["maxeuclerr_last5"] = np.max(record["euclerr"][-5:])
        records.append(record)

    df = pd.DataFrame(records)
    if save_to_file:
        eval_dir = run_dir.joinpath("filter_evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        df.to_pickle(
            eval_dir.joinpath(
                f"filter_evaluation_step{checkpoint_step}_{data_group}_{min_pred_length_sec}_si{use_side_info}.pkl"
            )
        )
    return df


def evaluate(
    run_dir,
    run_name,
    checkpoint_step,
    filter_cell,
    eval_data_groups,
    split_whitelist,
    writer,
    min_pred_length_sec_list=None,
):
    if min_pred_length_sec_list is None:
        min_pred_length_sec_list = [0.1, 0.5, 1.0]

    for min_pred_length_sec in min_pred_length_sec_list:
        for data_group in eval_data_groups:
            for use_side_info in [True, False]:
                with torch.no_grad():
                    df = compute_evaluations(
                        run_dir=run_dir,
                        run_name=run_name,
                        checkpoint_step=checkpoint_step,
                        data_group=data_group,
                        split_whitelist=split_whitelist,
                        min_pred_length_sec=min_pred_length_sec,
                        use_side_info=use_side_info,
                        sequence_filter=SequenceFilter(filter_cell),
                        save_to_file=True,
                    )
                for split in split_whitelist:
                    df_filt = df if split == "all" else df[df["traj_split"] == split]
                    max_abs_err = df_filt["maxabserr_last5"].mean()
                    if writer:
                        writer.add_scalar(
                            f"val/maxabserr_last5_{data_group}_si{use_side_info}_{split}_{min_pred_length_sec}",
                            max_abs_err.item(),
                            checkpoint_step,
                        )
