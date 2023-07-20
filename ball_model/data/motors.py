""" 
Mappings from motor actuations s_(tl,tr,b) to revolutions per minute 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import numpy as np
from scipy import interpolate

MOTOR_CURVES = {
    "v2_offset": {
        "rpm_tl": [
            297,
            896,
            1316,
            1726,
            2180,
            2445,
            2735,
            3005,
            3178,
            3337,
            3508,
            3654,
            3826,
            3856,
            3917,
            4123,
            4337,
            4575,
            4614,
            4615,
        ],
        "rpm_tr": [
            0,
            476,
            955,
            1385,
            1864,
            2207,
            2457,
            2716,
            2975,
            3150,
            3332,
            3518,
            3642,
            3739,
            3851,
            3960,
            4060,
            4198,
            4433,
            4596,
        ],
        "rpm_bc": [
            0,
            517,
            1077,
            1594,
            2055,
            2515,
            2851,
            3091,
            3387,
            3572,
            3757,
            3930,
            4060,
            4278,
            4535,
            4816,
            4959,
            4960,
            4961,
            4962,
        ],
        "actuation": [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
        ],
    },
    "v2": {
        "rpm_tl": [
            200,
            747,
            1430,
            1892,
            2373,
            2669,
            3018,
            3238,
            3440,
            3601,
            3781,
            3870,
            3950,
            4170,
            4437,
            4615,
            4616,
        ],
        "rpm_tr": [
            0,
            253,
            963,
            1460,
            1952,
            2346,
            2724,
            2984,
            3235,
            3429,
            3587,
            3744,
            3871,
            3977,
            4115,
            4385,
            4597,
        ],
        "rpm_bc": [
            183,
            603,
            1260,
            1780,
            2275,
            2665,
            3043,
            3305,
            3527,
            3706,
            3865,
            4046,
            4210,
            4484,
            4763,
            4962,
            4964,
        ],
        "actuation": [
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
        ],
    },
}


def _actuation_to_rpm(actuation, motor_name):
    curve_name = "v2_offset"
    curve_rpm = np.array(MOTOR_CURVES[curve_name][f"rpm_{motor_name}"])
    curve_actuation = np.array(MOTOR_CURVES[curve_name]["actuation"])
    if (np.max(actuation) > np.max(curve_actuation)) or (
        np.min(actuation) < np.min(curve_actuation)
    ):
        print("Value out of bounds")
    fcn = interpolate.interp1d(curve_actuation, curve_rpm)
    interp_rpm = fcn(actuation)
    return interp_rpm


def actuation_to_rpm(actuation, motor_name=None):
    if actuation.ndim == 2:
        assert actuation.shape[-1] == 3
        interp_rpm_tl = _actuation_to_rpm(actuation[:, 0], motor_name="tl")
        interp_rpm_tr = _actuation_to_rpm(actuation[:, 1], motor_name="tr")
        interp_rpm_bc = _actuation_to_rpm(actuation[:, 2], motor_name="bc")
        return np.stack((interp_rpm_tl, interp_rpm_tr, interp_rpm_bc), axis=-1)
    elif actuation.ndim == 1:
        return _actuation_to_rpm(actuation, motor_name=motor_name)
    else:
        raise ValueError
