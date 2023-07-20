""" 
Check manually computed Jacobians with autodiff 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import numpy as np
import torch

from ball_model.differentiable_simulator.simulator import (
    MatrixImpactModel,
    Simulator,
    State,
)


def run_gradcheck():
    torch.manual_seed(43)

    impact_model_auto = MatrixImpactModel(
        jacobian_type="auto",
    )
    impact_model_manual = MatrixImpactModel(
        jacobian_type="manual",
    )
    impact_model_manual.load_state_dict(impact_model_auto.state_dict())

    dt = 0.05
    radius_of_ball = 0.02
    gravity_z = 9.802
    table_z = -0.5 - radius_of_ball
    simulator_auto = Simulator(
        dt,
        impact_model_auto,
        "auto",
        table_z,
        gravity_z,
        radius_of_ball,
        kd_km_offset=0.05,
    )
    simulator_manual = Simulator(
        dt,
        impact_model_manual,
        "manual",
        table_z,
        gravity_z,
        radius_of_ball,
        kd_km_offset=0.05,
    )

    n_samples = 1000
    x0 = torch.zeros(n_samples, 3)
    x0[:, 2] = -0.4
    v0 = 5 * (2 * torch.rand(n_samples, 3) - 1)
    w0 = 5 * (2 * torch.rand(n_samples, 3) - 1)
    Ad = 0.2 * torch.ones(n_samples, 1)
    Am = 0.2 * torch.ones(n_samples, 1)

    z0 = State.from_components(x0, v0, w0, Ad, Am)

    z1_manual, J_manual = simulator_manual.predict(z0, with_jacobian=True)

    z1_auto, J_auto = simulator_auto.predict(z0, with_jacobian=True)

    print(f"Prediction err: {torch.max(torch.abs(z1_manual.t-z1_auto.t))}")
    print(f"Jacobian err: {torch.max(torch.abs(J_auto-J_manual))}")

    max_err_idx = torch.argmax(torch.abs(J_auto - J_manual))
    nd_idx = np.unravel_index(max_err_idx, J_auto.shape)
    print(f"Index of max deviation: {nd_idx}")
    print(f"Auto val: {J_auto[nd_idx]}, Manual val: {J_manual[nd_idx]}")


if __name__ == "__main__":
    run_gradcheck()
