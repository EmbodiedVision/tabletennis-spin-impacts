""" 
Utils for detecting table impacts 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""

import numpy as np


def detect_impacts(positions):
    vz = (
        positions[1:, 2] - positions[:-1, 2]
    )
    down_and_two_before = np.concatenate(
        (np.array([False, False]), (vz[2:] < 0) & (vz[1:-1] < 0) & (vz[:-2] < 0))
    )
    up_and_two_after = np.concatenate(
        ((vz[:-2] > 0) & (vz[1:-1] > 0) & (vz[2:] > 0), np.array([False, False]))
    )
    last_down_before_up = np.concatenate(
        (
            down_and_two_before[:-1] & up_and_two_after[1:],
            np.array([False]),
            np.array([False]),
        )
    )
    last_down_before_up_idx = np.nonzero(last_down_before_up)[0]
    first_up_idx = last_down_before_up_idx + 1
    return first_up_idx


def detect_impact(positions):
    impacts = detect_impacts(positions)
    assert len(impacts) == 1
    return impacts[0]
