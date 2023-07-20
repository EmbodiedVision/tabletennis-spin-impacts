""" 
Extended Kalman Filter cell 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import math
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn import functional as F

from ball_model.data.rotations import build_aug_rot_mat, build_launcher_rot_mat
from ball_model.differentiable_simulator.simulator import (  # noqa # pylint: disable=unused-import
    MatrixImpactModel,
    ODETransitionModel,
    Simulator,
    State,
)
from ball_model.filter.filter_cell import FilterCell
from ball_model.filter.state_belief import StateBelief


# softplus inverse
def sp_inv(val):
    return math.log(math.expm1(val))


class EKFStateBelief(StateBelief):
    def __init__(self, mean, covariance_matrix, Q, R, side_info, state_type):
        assert tuple(covariance_matrix.shape[-2:]) == (mean.size, mean.size)
        tensor_dict = {
            "mean": mean,
            "covariance_matrix": covariance_matrix,
            "Q": Q,
            "R": R,
            "side_info": side_info,
        }
        super(EKFStateBelief, self).__init__(tensor_dict, state_type=state_type)

    @staticmethod
    def from_tensor_dict(tensor_dict, state_type):
        return EKFStateBelief(**tensor_dict, state_type=state_type)

    @property
    def impact_here(self):
        return self.mean.impact_here

    @property
    def impact_before(self):
        return self.mean.impact_before

    @property
    def meas_dist(self):
        # Return measurement distribution
        mean = self.mean.X
        slice = self.mean.SLICE["X"]
        cov_matrix = self.covariance_matrix[:, slice, slice] + self.R
        dist = MultivariateNormal(mean, cov_matrix, validate_args=False)
        return dist


class EkfFilterCell(FilterCell):
    def __init__(
        self,
        dt,
        impact_model_classname,
        impact_model_kwargs,
        jacobian_type,
        table_z,
        gravity_z,
        radius_of_ball,
        kd_km_offset,
    ):
        super(EkfFilterCell, self).__init__()
        self.dt = dt

        self.initial_W_bi = nn.Parameter(
            torch.zeros(1, 3), requires_grad=False
        )  # before impact
        self.initial_W_ai = nn.Parameter(
            torch.zeros(1, 3), requires_grad=False
        )  # after impact
        self.initial_Ad = nn.Parameter(math.sqrt(0.1) * torch.ones(1, 1))
        self.initial_Am = nn.Parameter(math.sqrt(0.1) * torch.ones(1, 1))

        self.initial_pos_var_logit = nn.Parameter(sp_inv(1e-4) * torch.ones(1, 3))
        self.initial_vel_var_logit = nn.Parameter(sp_inv(1e-2) * torch.ones(1, 3))
        self.initial_W_bi_var_logit = nn.Parameter(sp_inv(1) * torch.ones(1, 3))
        self.initial_W_ai_var_logit = nn.Parameter(sp_inv(1) * torch.ones(1, 3))
        self.initial_Ad_var_logit = nn.Parameter(sp_inv(1e-2) * torch.ones(1, 1))
        self.initial_Am_var_logit = nn.Parameter(sp_inv(1e-2) * torch.ones(1, 1))

        self.trans_pos_var_logit = nn.Parameter(sp_inv(1e-4) * torch.ones(1, 3))
        self.trans_vel_var_logit = nn.Parameter(sp_inv(1e-2) * torch.ones(1, 3))
        self.trans_W_var_logit = nn.Parameter(sp_inv(1e-3) * torch.ones(1, 3))
        self.trans_Ad_var_logit = nn.Parameter(sp_inv(1e-2) * torch.ones(1, 1))
        self.trans_Am_var_logit = nn.Parameter(sp_inv(1e-2) * torch.ones(1, 1))

        self.obs_pos_var_logit = nn.Parameter(sp_inv(1e-3) * torch.ones(1, 3))

        self.impact_model = globals()[impact_model_classname](
            **impact_model_kwargs, jacobian_type=jacobian_type
        )
        self.simulator = Simulator(
            dt,
            self.impact_model,
            jacobian_type,
            table_z,
            gravity_z,
            radius_of_ball,
            kd_km_offset,
        )

        # input motor settings,  output spin + logits in canonical orientation
        self.initial_W_net = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

        self.Q_net = nn.Sequential(
            nn.Linear(11, 256),
            nn.ReLU(),
            nn.Linear(256, 11),
        )

        self.R_net = nn.Sequential(
            nn.Linear(11, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        self.to(dtype=torch.float64)

    def load_state_dict(self, state_dict, strict=True):
        # Remove deprecated weights from impact model
        filtered_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("impact_model.nn"):
                filtered_dict[k] = v
        super().load_state_dict(filtered_dict, strict=strict)

    @property
    def Kd_Km(self):
        s = namedtuple("s", "Ad Am")
        return self.simulator.transition_model.compute_kd_km(
            s(self.initial_Ad, self.initial_Am)
        )

    @property
    def StateBelief(self):
        return EKFStateBelief

    def get_Q(self, batch_dim):
        all_logits = torch.cat(
            (
                [
                    self.trans_pos_var_logit,
                    self.trans_vel_var_logit,
                    self.trans_W_var_logit,
                    self.trans_Ad_var_logit,
                    self.trans_Am_var_logit,
                ]
            ),
            dim=-1,
        )
        all_logits = all_logits.expand(batch_dim, all_logits.shape[-1])
        var = F.softplus(all_logits) + 1e-6
        Q = torch.diag_embed(var)
        return Q

    def get_R(self, batch_dim):
        all_logits = self.obs_pos_var_logit
        all_logits = all_logits.expand(batch_dim, all_logits.shape[-1])
        var = F.softplus(all_logits) + 1e-6
        R = torch.diag_embed(var)
        return R

    def initialize_filter(
        self,
        measurements,
        frame_indices,
        first_up_frames,
        side_info,
        use_side_info_mask,
        beginning_of_trajectory,
    ):
        # we explicitly supply the filter with the first measurements,
        # and a finite difference of the first two measurements divided by the first dt
        assert measurements.dim() == 3  # T x B x m
        assert frame_indices.dim() == 2  # T x B
        assert side_info.dim() == 2  # B x k
        assert measurements.shape[0] > 1
        assert measurements.shape[:-1] == frame_indices.shape

        pos_0 = measurements[0, :, :]
        pos_1 = measurements[1, :, :]

        initial_vel = (measurements[1, :, :] - measurements[0, :, :]) / (
            (frame_indices[1, :] - frame_indices[0, :]) * self.dt
        )[..., None]
        n_traj = measurements.shape[1]

        # rot_z_ccw_sin,rot_z_ccw_cos,phi,theta,top_left_motor,top_right_motor,bottom_motor
        assert side_info.shape[-1] == 7

        initial_W = self.initial_W_bi.expand(n_traj, 3).clone()
        initial_W_var = (
            torch.diag_embed(F.softplus(self.initial_W_bi_var_logit) + 1e-6)
            .expand(n_traj, 3, 3)
            .clone()
        )

        # for those trajectories where use_side_info_mask is set, overwrite initial_W, initial_W_var
        if torch.any(use_side_info_mask):
            side_info_masked = side_info[use_side_info_mask]

            # If we are at the beginning of a trajectory (during evaluation), we estimate
            # phi from the first two measurements. If phi_est is None, we use
            # the value from side_info
            if beginning_of_trajectory:
                dx = (measurements[1, :, 0] - measurements[0, :, 0])[use_side_info_mask]
                dy = (measurements[1, :, 1] - measurements[0, :, 1])[use_side_info_mask]
                # how much do we have to rotate "x"
                angle_x = torch.atan2(dy, dx)
                # how much do we have to rotate "-y" around z
                rot_z_est = angle_x - (270 / 180) * math.pi
                rot_sin = torch.sin(rot_z_est)
                rot_cos = torch.cos(rot_z_est)
                rot_mat_phi_theta = build_launcher_rot_mat(
                    side_info_masked[:, 2:4], ignore_phi=True
                )
            else:
                rot_sin = side_info_masked[:, 0]
                rot_cos = side_info_masked[:, 1]
                rot_mat_phi_theta = build_launcher_rot_mat(side_info_masked[:, 2:4])

            rot_mat_aug = build_aug_rot_mat(torch.stack((rot_sin, rot_cos), dim=-1))
            rot_mat_aug = torch.tensor(rot_mat_aug).to(self.device)
            rot_mat_phi_theta = torch.tensor(rot_mat_phi_theta).to(self.device)

            rot_mat = rot_mat_aug @ rot_mat_phi_theta
            rot_mat_T = rot_mat.transpose(-1, -2)

            initial_W_ai = self.initial_W_ai.expand(n_traj, 3).clone()
            initial_W_ai_var = (
                torch.diag_embed(F.softplus(self.initial_W_ai_var_logit) + 1e-6)
                .expand(n_traj, 3, 3)
                .clone()
            )

            initial_W_ai = (rot_mat @ initial_W_ai[use_side_info_mask][..., None])[
                ..., 0
            ]
            initial_W_ai_var = (
                rot_mat @ initial_W_ai_var[use_side_info_mask] @ rot_mat_T
            )
            init_tensor = self.initial_W_net(side_info_masked[:, 4:7])
            initial_W_bi = init_tensor[:, :3]
            initial_W_bi_var_logit = init_tensor[:, 3:6]
            initial_W_bi_var = torch.diag_embed(
                F.softplus(initial_W_bi_var_logit) + 1e-6
            )
            initial_W_bi = (rot_mat @ initial_W_bi[..., None])[..., 0]
            initial_W_bi_var = rot_mat @ initial_W_bi_var @ rot_mat_T

            is_after_impact = (
                (first_up_frames > -1) & (frame_indices[1, :] >= first_up_frames)
            )[use_side_info_mask]

            initial_W[use_side_info_mask] = torch.where(
                is_after_impact[:, None], initial_W_ai, initial_W_bi
            )
            initial_W_var[use_side_info_mask] = torch.where(
                is_after_impact[:, None, None], initial_W_ai_var, initial_W_bi_var
            )

        initial_mean = torch.cat(
            (
                pos_0,
                initial_vel,
                initial_W,
                self.initial_Ad.expand(n_traj, 1),
                self.initial_Am.expand(n_traj, 1),
            ),
            dim=-1,
        )

        initial_pos_var = torch.diag_embed(
            F.softplus(self.initial_pos_var_logit) + 1e-6
        ).expand(n_traj, 3, 3)

        initial_vel_var = torch.diag_embed(
            F.softplus(self.initial_vel_var_logit) + 1e-6
        ).expand(n_traj, 3, 3)

        initial_Ad_var = (
            F.softplus(self.initial_Ad_var_logit)[..., None].expand(n_traj, 1, 1) + 1e-6
        )
        initial_Am_var = (
            F.softplus(self.initial_Am_var_logit)[..., None].expand(n_traj, 1, 1) + 1e-6
        )

        z3 = torch.zeros(n_traj, 3, 3)
        z31 = torch.zeros(n_traj, 3, 1)
        z13 = torch.zeros(n_traj, 1, 3)
        z11 = torch.zeros(n_traj, 1, 1)

        initial_cov = torch.cat(
            [
                torch.cat((initial_pos_var, z3, z3, z31, z31), dim=-1),
                torch.cat((z3, initial_vel_var, z3, z31, z31), dim=-1),
                torch.cat((z3, z3, initial_W_var, z31, z31), dim=-1),
                torch.cat((z13, z13, z13, initial_Ad_var, z11), dim=-1),
                torch.cat((z13, z13, z13, z11, initial_Am_var), dim=-1),
            ],
            dim=-2,
        )

        initial_mean_0 = initial_mean.clone()
        initial_mean_1 = initial_mean.clone()
        initial_mean_1[..., :3] = pos_1

        Q = self.get_Q(n_traj)
        R = self.get_R(n_traj)
        initial_prior = (None, None)
        initial_posterior = []
        for m in [initial_mean_0, initial_mean_1]:
            mean_obj = State(m, impact_here=None, impact_before=None)
            belief_obj = EKFStateBelief(
                mean_obj,
                initial_cov,
                Q,
                R,
                side_info,
                state_type="corrected",
            )
            initial_posterior.append(belief_obj)
        initial_posterior = tuple(initial_posterior)

        return initial_prior, initial_posterior

    def _predict_singlestep(self, state, with_uncertainty=True):
        Q = self.get_Q(state.batch_dim)
        if with_uncertainty:
            pred_mean, A = self.simulator.predict(state.mean, with_jacobian=True)
            AT = torch.transpose(A, -1, -2)
            P_p = Q + A @ state.covariance_matrix @ AT
        else:
            pred_mean = self.simulator.predict(state.mean, with_jacobian=False)
            P_p = torch.zeros_like(state.covariance_matrix)
        R = self.get_R(state.batch_dim)
        state_belief = EKFStateBelief(
            pred_mean, P_p, Q, R, state.side_info, state_type="predicted"
        )
        return state_belief

    def correct(self, predicted_state, measurement):
        assert all(t == "predicted" for t in predicted_state.state_type)
        R = self.get_R(predicted_state.batch_dim)
        # we only observe the position ('X')
        h_diag = torch.zeros_like(predicted_state.mean.t)
        h_diag[..., predicted_state.mean.SLICE["X"]] = 1
        H = torch.diag_embed(h_diag)
        H = H[..., predicted_state.mean.SLICE["X"], :]
        HT = H.transpose(-1, -2)
        P_p = predicted_state.covariance_matrix
        K = P_p @ HT @ torch.linalg.inv(H @ P_p @ HT + R)
        x_corr = (
            predicted_state.mean.t
            + (K @ (measurement - predicted_state.mean.X)[:, :, None])[..., 0]
        )
        I = torch.eye(predicted_state.mean.size, device=K.device)
        I = I.expand(predicted_state.mean.batch_dim, *I.shape)
        P_m = (I - (K @ H)) @ P_p
        state_corr = State(
            x_corr,
            impact_here=predicted_state.impact_here,
            impact_before=predicted_state.impact_before,
        )
        Q = predicted_state.Q
        state_gaussian = EKFStateBelief(
            state_corr, P_m, Q, R, predicted_state.side_info, state_type="corrected"
        )
        return state_gaussian

    def smooth(
        self,
        next_smoothed: StateBelief,
        next_pred: StateBelief,
        state_corr: StateBelief,
    ):
        AT = next_pred.A.transpose(-1, -2)
        P_p = next_pred.covariance_matrix
        J = state_corr.covariance_matrix @ AT @ torch.linalg.inv(P_p)
        JT = J.transpose(-1, -2)
        P_m = state_corr.covariance_matrix
        P_s = P_m + J @ (next_smoothed.covariance_matrix - P_p) @ JT
        x_s = (
            state_corr.mean.t
            + (J @ (next_smoothed.mean.t - next_pred.mean.t)[:, :, None])[..., 0]
        )
        Q = None
        R = None
        state_mean = State(
            x_s,
            impact_here=state_corr.impact_here,
            impact_before=state_corr.impact_before,
        )
        return EKFStateBelief(
            state_mean,
            P_s,
            Q,
            R,
            next_pred.side_info,
            state_type="smoothed",
        )

    def loss(self, predicted_state, filtered_state, measurement):
        # predicted_state may be None during the warmup phase
        if predicted_state is not None:
            obs_logll = predicted_state.meas_dist.log_prob(measurement)
            return {"loss": -1.0 * obs_logll, "obs_logll": obs_logll}
        else:
            return {}
