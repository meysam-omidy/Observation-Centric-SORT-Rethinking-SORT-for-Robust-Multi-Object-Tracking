from __future__ import annotations

from copy import deepcopy

import numpy as np
from filterpy.kalman import KalmanFilter as FilterPyKalman
from filterpy.kalman import KalmanFilter as KalmanFilterBase

from utils import (
    batch_speed_direction,
    get_dict_item,
    tlbr_to_z,
    z_to_tlbr,
)


def create_sort_kalman(tlbr: np.ndarray, score: float, use_oru: bool = False) -> FilterPyKalman:
    """
    Standard linear KF for SORT (7-D state: x, y, s, r, vx, vy, vs).

    use_oru=False: plain filterpy predict/update (no gap logic).
    use_oru=True:  OC-SORT Observation-Centric Re-Update — on re-detection after a
                   gap, freeze()/re_update() rewinds to the last real observation and
                   replays the filter through a virtual straight-line trajectory to the
                   new observation, correcting the drifted velocity. Same matrices and
                   confidence-scaled R; only the predict/update mechanics differ.
    """
    kf = KalmanFilter(7, 4) if use_oru else FilterPyKalman(7, 4)
    kf.F = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    kf.H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=float,
    )
    kf.R = np.array(
        [
            [20, 0, -350, 0],
            [0, 20, 700, 0],
            [-350, 700, 2e7, 50],
            [0, 0, 50, 0],
        ],
        dtype=float,
    )
    kf.R[2:, 2:] *= 10
    kf.R *= np.e ** (2 * (1 - float(score)))
    kf.P[4:, 4:] *= 1000
    kf.P *= 10.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01
    z = tlbr_to_z(tlbr)
    kf.x[:4] = z
    if use_oru:
        # seed the observation history so the first re_update has a valid anchor
        kf.history['update'][kf.age] = z
    return kf


def _var_xywh_norm_to_z_diag(
    var_norm: np.ndarray,
    xywh_px: np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Diagonal (x,y,s,r) variances from normalized xywh variances (matching training scale)."""
    iw, ih = float(image_width), float(image_height)
    vn = np.maximum(np.asarray(var_norm, dtype=float).reshape(4), 1e-8)
    vx = vn[0] * iw ** 2
    vy = vn[1] * ih ** 2
    vw = vn[2] * iw ** 2
    vh = vn[3] * ih ** 2
    cx, cy, w, h = np.asarray(xywh_px, dtype=float).reshape(4)
    h = max(h, 1e-3)
    w = max(w, 1e-3)
    vs = (h ** 2) * vw + (w ** 2) * vh
    vr = vw / (h ** 2) + (w ** 2) * vh / (h ** 4)
    return np.array([vx, vy, max(vs, 1e-6), max(vr, 1e-6)])


def learned_process_noise_matrix(
    var_q_norm: np.ndarray,
    xywh_px: np.ndarray,
    image_width: float,
    image_height: float,
    vel_floor: float = 0.01,
) -> np.ndarray:
    """7x7 diagonal Q: learned xywh motion uncertainty mapped to (x,y,s,r) + small velocity floors."""
    d = _var_xywh_norm_to_z_diag(var_q_norm, xywh_px, image_width, image_height)
    q = np.array(
        [d[0], d[1], d[2], d[3], vel_floor, vel_floor, vel_floor],
        dtype=float,
    )
    return np.diag(np.maximum(q, 1e-8))


def learned_measurement_noise_matrix(
    var_r_norm: np.ndarray,
    xywh_px: np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """4x4 diagonal R in z space."""
    d = _var_xywh_norm_to_z_diag(var_r_norm, xywh_px, image_width, image_height)
    return np.diag(np.maximum(d, 1e-8))


class KalmanFilter(KalmanFilterBase):
    def __init__(self, dim_x, dim_z):
        super().__init__(dim_x, dim_z)
        self.time_since_last_update = 0
        self.age = 0
        self.dict = None
        self.history = {
            'update': {},
            'predict': {}
        }

    def update(self, z, R=None, H=None):
        if self.time_since_last_update > 1:
            self.re_update(z, R, H)
        else:
            super().update(z, R, H)
        self.time_since_last_update = 0
        self.history['update'][self.age] = z

    def predict(self, u=None, B=None, F=None, Q=None):
        if self.time_since_last_update == 1:
            self.freeze()
        super().predict(u, B, F, Q)
        self.time_since_last_update += 1
        self.history['predict'][self.age] = self.x
        self.age += 1

    def re_update(self, z, R=None, H=None):
        virtual_z = np.linspace(
            self.history['update'][list(self.history['update'].keys())[-1]],
            z,
            self.time_since_last_update + 1
        )
        self.unfreeze()
        for virtual_z_ in virtual_z[1:]:
            super().update(virtual_z_, R, H)
            super().predict()
        self.time_since_last_update = 0
        self.history['update'][self.age] = z

    def freeze(self):
        self.__dict__.pop('dict')
        self.dict = deepcopy(self.__dict__)
        self.dict.pop('age')
        self.dict.pop('history')
        self.dict.pop('time_since_last_update')

    def unfreeze(self):
        if self.dict:
            self.dict['age'] = self.age
            self.dict['history'] = self.history
            self.dict['time_since_last_update'] = self.time_since_last_update
            self.__dict__ = deepcopy(self.dict)
            self.dict = None

    def k_last_observation(self, delta_t):
        for i in range(delta_t, 0, -1):
            k = self.age - i - 1
            if k in self.history['update']:
                return self.history['update'][k]
        if len(self.history['update']) < 2:
            return np.array([0.5, 0.5, 1, 1])
        return get_dict_item(self.history['update'], -2)

    def speed_direction(self, delta_t):
        if len(self.history['update']) < 2:
            return np.float64(0)
        box_last = z_to_tlbr(get_dict_item(self.history['update'], -1)).reshape(1, 4)
        box_k = z_to_tlbr(self.k_last_observation(delta_t)).reshape(1, 4)
        return batch_speed_direction(box_k, box_last)[0, 0]


def init_kalman_filter(z, c) -> KalmanFilter:
    """Legacy helper on generic z (4,1); kept for compatibility with older call sites."""
    kf = KalmanFilter(7, 4)
    kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
    kf.R = np.array([
        [20, 0, -350, 0],
        [0, 20, 700, 0],
        [-350, 700, 2e7, 50],
        [0, 0, 50, 0]
    ])
    kf.R[2:,2:] *= 10
    kf.R *= np.e ** (2 * (1 - c))
    kf.P[4:,4:] *= 1000
    kf.P *= 10.
    kf.Q[-1,-1] *= 0.01
    kf.Q[4:,4:] *= 0.01
    kf.x[:4] = z
    kf.history['update'][kf.age] = z
    return kf
