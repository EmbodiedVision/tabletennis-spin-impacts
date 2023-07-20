""" 
Differentiable simulator of the ball flight with spin and impact 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import numpy as np
import torch
from functorch import jacfwd, vmap
from torch import nn

FIELD_ORDER = ["X", "V", "W", "Ad", "Am"]
FIELD_SIZE = [3, 3, 3, 1, 1]
OFFSETS = {
    "x": 0,
    "y": 1,
    "z": 2,
}
CUM_SUM = np.cumsum(
    np.array(
        [
            0,
        ]
        + FIELD_SIZE
    )
)
FIRST_INDEX = {k: CUM_SUM[idx] for idx, k in enumerate(FIELD_ORDER)}
SLICE = {
    k: slice(FIRST_INDEX[k], FIRST_INDEX[k] + FIELD_SIZE[idx])
    for idx, k in enumerate(FIELD_ORDER)
}


class State:
    def __init__(self, t, impact_here=None, impact_before=None):
        assert t.shape[-1] == CUM_SUM[-1]
        self.size = CUM_SUM[-1]
        self.SLICE = SLICE
        self.t = t
        if impact_here is None:
            impact_here = torch.zeros(t.shape[:-1]).bool().to(t.device)
        if impact_before is None:
            impact_before = torch.zeros(t.shape[:-1]).bool().to(t.device)
        self.impact_here = impact_here
        self.impact_before = impact_before

    def clone(self):
        return State(
            self.t.clone(),
            self.impact_here.clone() if self.impact_here is not None else None,
            self.impact_before.clone() if self.impact_before is not None else None,
        )

    @property
    def batch_dim(self):
        return self.t.shape[0]

    @property
    def shape(self):
        return self.t.shape

    @property
    def device(self):
        return self.t.device

    @property
    def dtype(self):
        return self.t.dtype

    @classmethod
    def from_components(cls, X, V, W, Ad, Am):
        for idx, arr in enumerate([X, V, W, Ad, Am]):
            assert arr.shape[-1] == FIELD_SIZE[idx]
        return cls(torch.cat((X, V, W, Ad, Am), dim=-1))

    @classmethod
    def stack_from_list(cls, state_list):
        stacked_components = {}
        for field in ["X", "V", "W", "Ad", "Am"]:
            stacked = torch.stack([getattr(state, field) for state in state_list])
            stacked_components[field] = stacked
        return cls.from_components(**stacked_components)

    @classmethod
    def concat(cls, state_list):
        t_list, impact_here_list, impact_before_list = [], [], []
        for state in state_list:
            t_list.append(state.t)
            impact_here_list.append(state.impact_here)
            impact_before_list.append(state.impact_before)
        return State(
            torch.cat(t_list),
            torch.cat(impact_here_list),
            torch.cat(impact_before_list),
        )

    def __getattr__(self, item):
        """ Return e.g. Wx, at FIRST_INDEX['W']+OFFSETS['x'] """
        if item in FIELD_ORDER:
            # to access X, V, W, Ad, Am
            return self.t[..., SLICE[item]]
        elif item[:-1] in ["X", "V", "W"] and item[-1] in OFFSETS.keys():
            # to access e.g. Xx, Xy, Xz, ...
            index = FIRST_INDEX[item[:-1]] + OFFSETS[item[-1]]
            t = self.t[..., index]
            return t
        else:
            return AttributeError

    def __getitem__(self, item):
        return State(self.t[item], self.impact_here[item], self.impact_before[item])

    def __setitem__(self, key, value):
        self.X[key] = value.X
        self.V[key] = value.V
        self.W[key] = value.W
        self.Ad[key] = value.Ad
        self.Am[key] = value.Am
        if value.impact_here is not None:
            self.impact_here[key] = value.impact_here
        if value.impact_before is not None:
            self.impact_before[key] = value.impact_before


def create_rot_mat(angle):
    z = torch.zeros_like(angle)
    o = torch.ones_like(angle)
    rot_mat = torch.stack(
        (
            torch.cos(angle),
            -torch.sin(angle),
            z,
            torch.sin(angle),
            torch.cos(angle),
            z,
            z,
            z,
            o,
        ),
        dim=-1,
    )
    rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 3, 3)
    return rot_mat


class MatrixImpactModel(nn.Module):
    def __init__(
        self,
        jacobian_type,
    ):
        super(MatrixImpactModel, self).__init__()
        self.jacobian_type = jacobian_type
        base_matrix = np.eye(6)
        base_matrix[2, 2] = -1
        self.base_matrix = nn.Parameter(torch.Tensor(base_matrix))

    def _impact_tensor(self, VW_cat):
        if VW_cat.dim() == 1:
            # probably vmap call
            squeeze = True
            VW_cat = VW_cat.unsqueeze(0)
        elif VW_cat.dim() == 2:
            squeeze = False
        else:
            raise ValueError
        batch_dim = VW_cat.shape[0]
        V = VW_cat[..., :3]
        W = VW_cat[..., 3:]
        VW = torch.cat((V, W), dim=-1)
        matrix = self.base_matrix.expand(batch_dim, *self.base_matrix.shape)
        VW_next = (matrix @ VW[..., None])[..., 0]
        V_next = VW_next[..., :3]
        W_next = VW_next[..., 3:]
        VW_next = torch.cat((V_next, W_next), dim=-1)
        if squeeze:
            VW_next = VW_next.squeeze(0)
        return VW_next

    def impact(self, x, with_jacobian):
        if self.jacobian_type == "auto":
            return self._impact_auto_jacobian(x, with_jacobian)
        elif self.jacobian_type == "manual":
            return self._impact_manual_jacobian(x, with_jacobian)
        else:
            raise ValueError

    def _impact_auto_jacobian(self, x, with_jacobian):
        VW = torch.cat((x.V, x.W), dim=-1)
        VW_next = self._impact_tensor(VW)
        V_next = VW_next[..., :3]
        W_next = VW_next[..., 3:]
        x_next = x.from_components(x.X, V_next, W_next, x.Ad, x.Am)
        if with_jacobian:
            jac_fcn = jacfwd(self._impact_tensor)
            J_VW = vmap(jac_fcn)(VW)
            J = (
                torch.eye(11, dtype=x.dtype)
                .to(x.device)[None, :, :]
                .expand(x.batch_dim, 11, 11)
                .clone()
            )
            J[:, 3:9, 3:9] = J_VW
            return x_next, J
        else:
            return x_next

    # Impact with manual jacobians
    def _impact_manual_jacobian(self, x, with_jacobian):
        V_rot = x.V
        W_rot = x.W
        VW_rot = torch.cat((V_rot, W_rot), dim=-1)
        matrix = self.base_matrix.expand(x.batch_dim, *self.base_matrix.shape)
        VW_rot_next = (matrix @ VW_rot[..., None])[..., 0]
        V_rot_next = VW_rot_next[..., :3]
        W_rot_next = VW_rot_next[..., 3:]
        V_next = V_rot_next
        W_next = W_rot_next
        x_next = x.from_components(x.X, V_next, W_next, x.Ad, x.Am)
        if with_jacobian:
            J_vw = matrix
            J = (
                torch.eye(11, dtype=x.dtype)
                .to(x.device)[None, :, :]
                .expand(x.batch_dim, 11, 11)
                .clone()
            )
            J[:, 3:9, 3:9] = J_vw

            return x_next, J
        else:
            return x_next


def cross_reimpl(a, b):
    # torch.cross not supported by vmap
    assert a.shape[-1] == 3
    assert b.shape[-1] == 3
    assert a.shape == b.shape
    return torch.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        dim=-1,
    )


def expand_batch(base_tensor, batch_dim):
    return base_tensor.expand(*batch_dim, *base_tensor.shape)


def jac_norm_vec(x):
    # Compute derivative of ||x|| * x wrt x
    assert x.dim() == 2
    assert x.shape[-1] == 3
    bs = x.shape[0]
    x_norm = torch.norm(x, p=2, dim=-1)[:, None, None]
    derivative = (1 / x_norm) * x[:, None, :] * x[:, :, None]
    I = torch.eye(3).to(x.device)
    derivative += I.expand(bs, 3, 3) * x_norm
    return derivative


def stack_blocks(nested_list):
    if isinstance(nested_list[0], torch.Tensor):
        assert nested_list[0].dim() == 1
        return torch.stack(nested_list, dim=1)
    else:
        return torch.stack([stack_blocks(l) for l in nested_list], dim=1)


def jac_cross_a(a, b):
    """
    Compute derivative of (a x b) wrt a
              0     b3   -b2    a1
    a x b =  -b3    0     b1  . a2
              b2   -b1    0     a3
    """
    assert a.shape == b.shape
    assert a.dim() == 2

    b1 = b[:, 0]
    b2 = b[:, 1]
    b3 = b[:, 2]
    z = torch.zeros_like(b1)
    block_mat = stack_blocks(
        [
            [z, b3, -b2],
            [-b3, z, b1],
            [b2, -b1, z],
        ]
    )
    return block_mat


def jac_cross_b(a, b):
    """
    Compute derivative of (a x b) wrt b
              0    -a3    a2    b1
    a x b =   a3    0    -a1  . b2
             -a2    a1    0     b3
    """
    assert a.shape == b.shape
    assert a.dim() == 2

    a1 = a[:, 0]
    a2 = a[:, 1]
    a3 = a[:, 2]
    z = torch.zeros_like(a1)

    block_mat = stack_blocks(
        [
            [z, -a3, a2],
            [a3, z, -a1],
            [-a2, a1, z],
        ]
    )
    return block_mat


class ODETransitionModel(nn.Module):
    def __init__(self, gravity_z, kd_km_offset):
        super(ODETransitionModel, self).__init__()
        self.gravity_z = gravity_z
        self.gravity = nn.Parameter(torch.Tensor(np.array([0, 0, -self.gravity_z])))
        self.kd_km_offset = kd_km_offset

    def compute_kd_km(self, s):
        Ad = s.Ad
        Am = s.Am

        Kd = Ad ** 2 + self.kd_km_offset
        Kd_jac_Ad = (2 * Ad)[..., None, :]

        Km = Am ** 2 + self.kd_km_offset
        Km_jac_Am = (2 * Am)[..., None, :]

        return Kd, Km, Kd_jac_Ad, Km_jac_Am

    def compute_accel(self, s, with_jacobian, within_vmap):
        """
        Compute acceleration v_dot and derivatives v_dot wrt z (X, V, W, Ad, Am)
        """
        V = s.V
        W = s.W

        Kd, Km, Kd_jac_Ad, Km_jac_Am = self.compute_kd_km(s)

        V_norm = torch.linalg.vector_norm(V, ord=2, dim=-1, keepdim=True)
        # use simple reimplementation of 'cross' as torch.cross is not yet covered by vmap
        if within_vmap:
            W_V_cross = cross_reimpl(W, V)
        else:
            # torch.cross is way faster
            W_V_cross = torch.cross(W, V, dim=-1)

        # W_V_cross = torch.cross(s.W, s.V, dim=-1)
        V_dot = -Kd * V_norm * V + Km * W_V_cross + self.gravity

        if with_jacobian:
            dVdotdP = torch.zeros(*s.t.shape[:-1], 3, 3, dtype=s.dtype).to(s.device)
            dVdotdV = -Kd[:, :, None] * jac_norm_vec(V) + Km[:, :, None] * jac_cross_b(
                W, V
            )
            dVdotdW = Km[:, :, None] * jac_cross_a(W, V)
            dVdotdAd = -(V_norm * V)[..., None] @ Kd_jac_Ad
            dVdotdAm = W_V_cross[..., None] @ Km_jac_Am
            dVdotdZ = torch.cat((dVdotdP, dVdotdV, dVdotdW, dVdotdAd, dVdotdAm), dim=-1)
        else:
            dVdotdZ = None

        return V_dot, dVdotdZ

    def forward(self, s, dt, with_jacobian, within_vmap=False):
        t = s.t
        X = s.X
        V = s.V
        W = s.W
        Ad = s.Ad
        Am = s.Am

        assert V.shape[-1] == 3
        assert W.shape[-1] == 3
        assert t.dim() == 2  # batch x pos
        batchsize = t.shape[0]
        assert dt.shape == (batchsize,)

        Kd, Km, _, _ = self.compute_kd_km(s)

        Vdot, dVdotdZ = self.compute_accel(s, with_jacobian, within_vmap)

        dtV = dt[:, None]

        X_next = X + dtV * V
        V_norm = torch.linalg.vector_norm(V, ord=2, dim=-1, keepdim=True)
        # use simple reimplementation of 'cross' as torch.cross is not yet covered by vmap
        if within_vmap:
            W_V_cross = cross_reimpl(W, V)
        else:
            # torch.cross is way faster
            W_V_cross = torch.cross(W, V, dim=-1)
        # W_V_cross = torch.cross(s.W, s.V, dim=-1)
        V_next = V + dtV * Vdot
        W_next = W
        Ad_next = Ad
        Am_next = Am
        pred = s.from_components(X_next, V_next, W_next, Ad_next, Am_next)

        if with_jacobian:
            I = torch.eye(3).expand(batchsize, 3, 3).to(t.device, dtype=s.t.dtype)
            Id = torch.ones(1).expand(batchsize, 1, 1).to(t.device, dtype=s.t.dtype)
            Im = torch.ones(1).expand(batchsize, 1, 1).to(t.device, dtype=s.t.dtype)
            J_z = torch.zeros((batchsize, s.size, s.size)).to(t.device, dtype=s.t.dtype)
            dtM = dt[:, None, None]

            def set_block(block_name, block_val):
                slice_row = s.SLICE[block_name.split("_")[0][1:]]
                slice_col = s.SLICE[block_name.split("_")[1][1:]]
                J_z[:, slice_row, slice_col] = block_val

            set_block("dX_dX", I)
            set_block("dW_dW", I)
            set_block("dAd_dAd", Id)
            set_block("dAm_dAm", Im)
            set_block("dX_dV", I * dtM)
            # torch.diag_embed(v) @ M == v[:, :, None] * M
            # M @ torch.diag_embed(v) == v[:, None, :] * M
            set_block(
                "dV_dV",
                (I + dtM * dVdotdZ[..., :, s.SLICE["V"]]),
            )
            set_block("dV_dW", dtM * dVdotdZ[..., :, s.SLICE["W"]])
            set_block("dV_dAd", dtM * dVdotdZ[..., :, s.SLICE["Ad"]])
            set_block("dV_dAm", dtM * dVdotdZ[..., :, s.SLICE["Am"]])

            # dzdt
            J_t = torch.zeros((batchsize, s.size, 1)).to(t.device, dtype=s.t.dtype)
            J_t[:, :3, 0] = V
            J_t[:, 3:6, 0] = -Kd * V_norm * V + Km * W_V_cross + self.gravity

            return pred, J_z, J_t
        else:
            return pred


class Simulator:
    def __init__(
        self,
        dt,
        impact_model,
        jacobian_type,
        table_z,
        gravity_z,
        radius_of_ball,
        kd_km_offset,
    ):
        assert jacobian_type in ["auto", "manual"]
        self.dt = dt
        self.impact_model = impact_model
        self.jacobian_type = jacobian_type
        self.table_z = table_z
        self.gravity_z = gravity_z
        self.radius_of_ball = radius_of_ball
        self.transition_model = ODETransitionModel(self.gravity_z, kd_km_offset)

    @property
    def State(self):
        return State

    def _predict_freefall(self, x, dt, with_jacobian, within_vmap=False):
        return self.transition_model(
            x, dt, with_jacobian=with_jacobian, within_vmap=within_vmap
        )

    def _handle_impact(self, x, with_jacobian=False):
        return self.impact_model.impact(x, with_jacobian)

    def _compute_impact_time(self, x, with_jacobian, within_vmap):
        """
        Return impact time and dt/dz
        """
        xz_before = x.Xz
        vz_before = x.Vz
        height_difference = -((xz_before - self.radius_of_ball) - self.table_z)
        height_difference = torch.clip(height_difference, max=-1e-4)

        # gravity_z is *positive* here
        accel_z = -self.gravity_z * torch.ones(x.batch_dim, dtype=x.dtype).to(x.device)
        # We do not leverage compute_accel to compute and integrate
        # an acceleration to determine the impact time.
        # As the penetration condition comes from Euler integration,
        # integrating accelerations at this point may lead to no
        # impacts although the penetration criterion holds true.
        # Incorporating gravity does not hurt, as this can
        # only reduce the time until impact.
        dt = torch.ones(x.batch_dim, dtype=x.dtype).to(x.device) * self.dt
        sqrt = torch.sqrt(vz_before ** 2 + 2 * accel_z * height_difference)
        t_to_impact = (vz_before + sqrt) / (-accel_z)

        if with_jacobian:
            dt0dz = torch.zeros(sqrt.shape[0], 1, 11, dtype=x.dtype).to(x.device)
            dt0dz0xz = 1 / sqrt
            dt0dz0vz = (1 / self.gravity_z) * (1 + vz_before * 1 / sqrt)
            dt0dz[:, 0, 2] = dt0dz0xz
            dt0dz[:, 0, 5] = dt0dz0vz
        else:
            dt0dz = None

        t_to_impact = torch.clip(t_to_impact, max=dt)

        return t_to_impact, dt0dz

    def _check_for_impact(self, x):
        """ Returns True if the transition contains an impact to be handled """
        dt = torch.ones(x.batch_dim, dtype=x.dtype).to(x.device) * self.dt
        x_after = self._predict_freefall(x, dt, with_jacobian=False)
        impact_here_mask = x_after.Xz - self.radius_of_ball < self.table_z
        return impact_here_mask

    def _predict_from_tensor(self, t, has_impact, within_vmap):
        dt_shape = 1 if within_vmap else t.shape[0]
        dt = torch.ones(dt_shape, dtype=t.dtype).to(t.device) * self.dt

        squeeze = False
        if t.dim() == 1:
            squeeze = True
            t = t.unsqueeze(0)

        x_before = self.State(t)

        if has_impact:
            t_to_impact, _ = self._compute_impact_time(
                x_before, with_jacobian=False, within_vmap=within_vmap
            )
            x_at_impact = self._predict_freefall(
                x_before, t_to_impact, with_jacobian=False, within_vmap=within_vmap
            )
            x_after_impact = self._handle_impact(x_at_impact)
            x_after_impact = self._predict_freefall(
                x_after_impact,
                dt - t_to_impact,
                with_jacobian=False,
                within_vmap=within_vmap,
            )
            x_after = x_after_impact
        else:
            x_after = self._predict_freefall(
                x_before, dt, with_jacobian=False, within_vmap=within_vmap
            )

        if squeeze:
            x_after = x_after[0]

        return x_after.t

    def _predict_auto_jacobian(self, x, with_jacobian=False):
        impact_here_mask = self._check_for_impact(x)
        impact_mask = impact_here_mask & (~x.impact_before)

        t_next = torch.zeros_like(x.t)
        J = torch.zeros(x.batch_dim, x.t.shape[-1], x.t.shape[-1], dtype=x.dtype).to(
            x.device
        )

        if torch.any(~impact_mask):
            x_no_impact = x[~impact_mask]
            t_next_no_impact = self._predict_from_tensor(
                x_no_impact.t, has_impact=False, within_vmap=False
            )
            t_next[~impact_mask] = t_next_no_impact
            if with_jacobian:
                jac_fcn = jacfwd(
                    lambda t_: self._predict_from_tensor(
                        t_, has_impact=False, within_vmap=True
                    )
                )
                J_no_impact = vmap(jac_fcn)(x_no_impact.t)
                J[~impact_mask] = J_no_impact

        if torch.any(impact_mask):
            x_with_impact = x[impact_mask]
            t_next_with_impact = self._predict_from_tensor(
                x_with_impact.t, has_impact=True, within_vmap=False
            )
            t_next[impact_mask] = t_next_with_impact
            if with_jacobian:
                jac_fcn = jacfwd(
                    lambda t_: self._predict_from_tensor(
                        t_, has_impact=True, within_vmap=True
                    )
                )
                J_with_impact = vmap(jac_fcn)(x_with_impact.t)
                J[impact_mask] = J_with_impact

        x_next = self.State(t_next)
        x_next.impact_here[impact_here_mask] = True
        x_next.impact_before[impact_here_mask | x.impact_before] = True

        if with_jacobian:
            return x_next, J
        else:
            return x_next

    def _predict_manual_jacobian(self, x, with_jacobian=False):
        dt = torch.ones(x.batch_dim, dtype=x.dtype).to(x.device) * self.dt

        assert x.t.dim() == 2
        # predict step, with potential table impact
        x_before = x
        if with_jacobian:
            x_after, J_z1_z0, _ = self._predict_freefall(x, dt, with_jacobian=True)
        else:
            x_after = self._predict_freefall(x, dt, with_jacobian=False)

        t_to_impact, dt0dz = self._compute_impact_time(
            x_before, with_jacobian=True, within_vmap=False
        )

        impact_here_mask = x_after.Xz - self.radius_of_ball < self.table_z
        # ignore impact if an impact already happened before
        impact_mask = impact_here_mask & (~x.impact_before)

        if torch.any(impact_mask):
            t_to_impact = t_to_impact[impact_mask]
            dt0dz = dt0dz[impact_mask]

            if with_jacobian:
                x_after_impact, J_f_z_z0_t0, J_f_t_z0_t0 = self._predict_freefall(
                    x_before[impact_mask], t_to_impact, with_jacobian=True
                )
                x_after_impact, J_h_zp_zm = self._handle_impact(
                    x_after_impact, with_jacobian=True
                )
                x_after_impact, J_f_z_zp_t1, J_f_t_zp_t1 = self._predict_freefall(
                    x_after_impact, dt[impact_mask] - t_to_impact, with_jacobian=True
                )
                dt1dz0 = -dt0dz
                dzmdz0 = J_f_z_z0_t0 + J_f_t_z0_t0 @ dt0dz
                dzpdz0 = J_h_zp_zm @ dzmdz0
                J_z1_z0_imp = J_f_z_zp_t1 @ dzpdz0 + J_f_t_zp_t1 @ dt1dz0
                J_z1_z0[impact_mask] = J_z1_z0_imp
            else:
                x_after_impact = self._predict_freefall(
                    x_before[impact_mask], t_to_impact, with_jacobian=False
                )
                x_after_impact = self._handle_impact(
                    x_after_impact, with_jacobian=False
                )
                x_after_impact = self._predict_freefall(
                    x_after_impact, dt[impact_mask] - t_to_impact, with_jacobian=False
                )

            x_after[impact_mask] = x_after_impact

        x_after.impact_here[impact_here_mask] = True
        x_after.impact_before[impact_here_mask | x.impact_before] = True

        if with_jacobian:
            return x_after, J_z1_z0
        else:
            return x_after

    def predict(self, x, with_jacobian):
        if self.jacobian_type == "auto":
            return self._predict_auto_jacobian(x, with_jacobian)
        elif self.jacobian_type == "manual":
            return self._predict_manual_jacobian(x, with_jacobian)
        else:
            raise ValueError
