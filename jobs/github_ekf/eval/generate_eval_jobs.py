""" 
Generate evaluation jobs

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from ball_model import EXPERIMENT_DIR


def main():
    run_base_dir = EXPERIMENT_DIR
    run_ids = []
    for subdir in run_base_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("_"):
            continue
        run_ids.append(subdir.name)

    run_ids = sorted(run_ids)

    eval_cmds = []
    for run_id in run_ids:
        for eval_data_group in ["lp7", "lp9"]:
            eval_cmds.append(
                f"python -m ball_model.evaluation.post_train_eval "
                f"--run_name {run_id} "
                f"--checkpoint_step 10000 "
                f"--eval_data_groups {eval_data_group}\n"
            )

    with open("eval_jobs_ekf.txt", "w") as handle:
        handle.writelines(eval_cmds)


if __name__ == "__main__":
    main()
