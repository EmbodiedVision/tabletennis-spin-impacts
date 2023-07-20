""" 
Generate training jobs

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from itertools import product

cmd_list = []

TRAIN_DATA_GROUPS = {
    "lp7": "['lp7']",
    "lp7aug": "['lp7aug']",
}
USE_SIDE_INFO = ["always", "never",]
SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


for (
    (group_name, group_string),
    use_side_info,
    seed,
) in product(
    TRAIN_DATA_GROUPS.items(),
    USE_SIDE_INFO,
    SEED,
):
    run_id = (
        f"model_ekf_g_{group_name}_si{use_side_info}_s{seed}"
    )
    cmd_list.append(
        f"python -m ball_model.training.train_model_ekf "
        f"--id={run_id} "
        f"with ekf_default_args n_epochs=10000 evaluate_every=500 "
        f"use_side_info={use_side_info} "
        f'"train_data_groups={group_string}" '
        f'"eval_data_groups={group_string}" '
        f"seed={seed} \n"
    )


with open("train_jobs_ekf.txt", "w") as handle:
    handle.writelines(cmd_list)
