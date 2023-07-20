""" 
Script to evaluate a model 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import argparse

from ball_model import EXPERIMENT_DIR
from ball_model.filter.filter_cell import load_filter_cell_from_run
from ball_model.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--checkpoint_step", type=int, required=True)
    parser.add_argument("--eval_data_groups", nargs="+", required=True)
    args = parser.parse_args()

    run_name = args.run_name
    run_dir = EXPERIMENT_DIR.joinpath(run_name)
    checkpoint_step = args.checkpoint_step
    split_whitelist = ["test", "val"]
    device = "cpu"
    writer = None

    filter_cell = load_filter_cell_from_run(run_dir, checkpoint_step, device)

    evaluate(
        run_dir,
        run_name,
        checkpoint_step=checkpoint_step,
        filter_cell=filter_cell,
        eval_data_groups=args.eval_data_groups,
        split_whitelist=split_whitelist,
        writer=writer,
        min_pred_length_sec_list=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )


if __name__ == "__main__":
    main()
