"""
Mappings from launcher parameters s_(phi,theta) to launch angles
The regression coefficients are computed in the notebook `ball_model/data/notebooks/visualization.ipynb`

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def launcher_phi_to_angle(phi):
    slope_phi = -56.08558569861544
    intercept_phi = -60.094624032259446
    rot_z_deg_fn = lambda x: x * slope_phi + intercept_phi
    rot_z_deg = rot_z_deg_fn(phi)
    return rot_z_deg


def launcher_theta_to_angle(theta):
    theta_switch = 0.8740863280312021
    slope_theta_1 = -32.82521212632685
    intercept_theta_1 = 81.98843931100635
    intercept_theta_2 = 53.29637017666002
    rot_x_deg_fn = lambda x: (
        (x < theta_switch) * (slope_theta_1 * x + intercept_theta_1)
        + ((x >= theta_switch) * intercept_theta_2)
    )
    rot_x_deg = rot_x_deg_fn(theta)
    return rot_x_deg


def build_launcher_rot_mat(side_info_phi_theta, ignore_phi=False):
    """ Compute rotation matrix for launcher head orientation """
    assert side_info_phi_theta.shape[-1] == 2
    if isinstance(side_info_phi_theta, torch.Tensor):
        side_info_phi_theta = side_info_phi_theta.cpu().numpy()
    if ignore_phi:
        # shoot in -y direction
        rot_z_deg = -90 * np.ones(*side_info_phi_theta.shape[:-1])
    else:
        phi = side_info_phi_theta[..., 0]
        rot_z_deg = launcher_phi_to_angle(phi)

    theta = side_info_phi_theta[..., 1]
    rot_x_deg = launcher_theta_to_angle(theta)
    rot = Rotation.from_euler(
        "ZY", np.stack([rot_z_deg, -(90 - rot_x_deg)], axis=-1), degrees=True
    )
    rot_mat = rot.as_matrix()
    return rot_mat


def build_aug_rot_mat(side_info_aug_rot):
    """ Compute rotation matrix for launcher frame orientation """
    assert side_info_aug_rot.shape[-1] == 2
    if isinstance(side_info_aug_rot, torch.Tensor):
        side_info_aug_rot = side_info_aug_rot.cpu().numpy()
    r_sin = side_info_aug_rot[..., 0]
    r_cos = side_info_aug_rot[..., 1]
    z = np.zeros_like(r_sin)
    i = np.ones_like(r_sin)
    rot_mat = np.stack(
        (
            np.stack((r_cos, -r_sin, z), axis=-1),
            np.stack((r_sin, r_cos, z), axis=-1),
            np.stack((z, z, i), axis=-1),
        ),
        axis=-2,
    )
    return rot_mat
