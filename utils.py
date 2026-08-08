import numpy as np
from typing import Union
import lap
import time
import pickle
from copy import copy

class BBOX(np.ndarray):
    def __new__(cls, bbox):
        return np.asarray(bbox).view(cls)
    
    def __repr__(self):
        return super().__repr__()
    
    def __str__(self):
        return super().__str__()

    @classmethod
    def from_tlbr(cls, obj : Union[list, np.ndarray]) -> "BBOX":
        bbox = np.array(obj)
        o = np.zeros_like(bbox, dtype=float)
        o[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
        o[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
        o[..., 2] = bbox[..., 2] - bbox[..., 0]
        o[..., 3] = bbox[..., 3] - bbox[..., 1]
        return cls(o)

    @classmethod
    def from_tlwh(cls, obj : Union[list, np.ndarray]) -> "BBOX":
        bbox = np.array(obj)
        o = np.zeros_like(bbox, dtype=float)
        o[..., 0] = bbox[..., 0] + bbox[..., 2] / 2
        o[..., 1] = bbox[..., 1] + bbox[..., 3] / 2
        o[..., 2] = bbox[..., 2]
        o[..., 3] = bbox[..., 3]
        return cls(o)

    @classmethod
    def from_xysa(cls, obj : Union[list, np.ndarray]) -> "BBOX":
        bbox = np.array(obj)
        z = z.reshape(-1)
        o = np.zeros_like(z, dtype=float)
        o[..., 0] = bbox[..., 0]
        o[..., 1] = bbox[..., 1]
        o[..., 2] = sqrt(bbox[..., 2] * bbox[..., 3])
        o[..., 3] = sqrt(bbox[..., 2] / bbox[..., 3])
        return cls(o)
    
    def to_tlbr(self) -> np.ndarray:
        o = np.zeros_like(self, dtype=float)
        o[..., 0] = self[..., 0] - self[..., 2] / 2
        o[..., 1] = self[..., 1] - self[..., 3] / 2
        o[..., 2] = self[..., 0] + self[..., 2] / 2
        o[..., 3] = self[..., 1] + self[..., 3] / 2
        return o
    
    def to_tlwh(self) -> np.ndarray:
        o = np.zeros_like(self, dtype=float)
        o[..., 0] = self[..., 0] - self[..., 2] / 2
        o[..., 1] = self[..., 1] - self[..., 3] / 2
        o[..., 2] = self[..., 2]
        o[..., 3] = self[..., 3]
        return o
    
    def to_xysa(self) -> np.ndarray:
        o = np.zeros_like(self, dtype=float)
        o[..., 0] = self[..., 0]
        o[..., 1] = self[..., 1]
        o[..., 2] = self[..., 2] * self[..., 3]
        o[..., 3] = self[..., 2] / self[..., 3]
        return o
    

def count_time(func):
    
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        print(f"{int((end-start) * 1000)} ms    {func.__name__}    {kwargs.get('cost_matrix').shape if np.any(kwargs.get('cost_matrix', None)) else ''}")
        return result
    return wrapper

def batch_iou(bb1, bb2):
    bb1 = np.expand_dims(bb1, 1)
    bb2 = np.expand_dims(bb2, 0)
    xx1 = np.maximum(bb1[..., 0], bb2[..., 0])
    yy1 = np.maximum(bb1[..., 1], bb2[..., 1])
    xx2 = np.minimum(bb1[..., 2], bb2[..., 2])
    yy2 = np.minimum(bb1[..., 3], bb2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb1[..., 2] - bb1[..., 0]) * (bb1[..., 3] - bb1[..., 1])                                      
        + (bb2[..., 2] - bb2[..., 0]) * (bb2[..., 3] - bb2[..., 1]) - wh)                                              
    return(o) 

def batch_speed_direction(bb1, bb2):
    bb1 = np.expand_dims(bb1, 1)
    bb2 = np.expand_dims(bb2, 0)
    cx1 = (bb1[..., 0] + bb1[..., 2]) / 2
    cy1 = (bb1[..., 1] + bb1[..., 3]) / 2
    cx2 = (bb2[..., 0] + bb2[..., 2]) / 2
    cy2 = (bb2[..., 1] + bb2[..., 3]) / 2
    dx = cx2 - cx1
    dy = cy2 - cy1
    return np.arctan2(dy, dx)

def assignment(cost_matrix):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    # cost, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches.tolist(), unmatched_a.tolist(), unmatched_b.tolist()
    
def select_indices(arr, indices):
    return [arr[index] for index in indices]

def get_dict_item(obj:dict, index:int):
    values = list(obj.values())
    return values[index]

def get_dict_key(obj:dict, index:int):
    keys = list(obj.keys())
    return keys[index]
    
def print_matrix(x, precision=3):
    print(np.array2string(x, precision=precision, suppress_small=True))

def sqrt(x:np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(x, 0))

def compute_motion_features(bboxes):
    bboxes = copy(bboxes)
    n = len(bboxes)
    enhanced = np.zeros((n, 12))
    enhanced[:, :4] = bboxes

    # Velocity (first-order difference)
    if n > 1:
        velocity = np.diff(bboxes, axis=0)
        enhanced[1:, 4:8] = velocity
        # First frame velocity = 0 (no previous frame)

    # Acceleration (second-order difference)
    if n > 2:
        acceleration = np.diff(velocity, axis=0)
        enhanced[2:, 8:12] = acceleration
        # First two frames acceleration = 0 (need at least 3 frames)

    return enhanced


# adaptive Kalman feature layout (see motion-predictor/adaptive_kalman_dataset.py)
ADAPTIVE_KALMAN_FEATURE_DIM = 15
ADAPTIVE_KALMAN_SCORE_IDX = 12
ADAPTIVE_KALMAN_FRAMES_SINCE_IDX = 13
ADAPTIVE_KALMAN_OBSERVED_IDX = 14


def compute_adaptive_kalman_features(bboxes, scores, observed, max_gap_norm=30.0):
    """
    Per-track feature window for the adaptive Kalman Q/R model (15-D):
    [x, y, w, h, vx..vh, ax..ah, det_score, frames_since_obs, is_observed].

    GAP-SAFE CONTRACT — must stay identical to the training builder
    motion-predictor/adaptive_kalman_dataset.py::build_adaptive_kalman_features:
      - position     kept only on observed rows (unobserved -> 0)
      - velocity[i]     only when rows i and i-1 are both observed (else 0)
      - acceleration[i] only when rows i, i-1, i-2 are all observed (else 0)
    Any derivative that would cross a gap is zeroed, so the features depend ONLY on
    real observations and never on whatever box fills a gap (KF prediction here at
    inference, zero in training). This removes the recovery-frame velocity spike and
    gap-boundary garbage that previously differed between train and inference.
    """
    bboxes = np.asarray(bboxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=bool).reshape(-1)
    n = len(bboxes)
    feat = np.zeros((n, ADAPTIVE_KALMAN_FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        if not observed[i]:
            continue
        feat[i, :4] = bboxes[i]
        feat[i, ADAPTIVE_KALMAN_SCORE_IDX] = scores[i]
        if i >= 1 and observed[i - 1]:
            feat[i, 4:8] = bboxes[i] - bboxes[i - 1]
            if i >= 2 and observed[i - 2]:
                feat[i, 8:12] = (bboxes[i] - bboxes[i - 1]) - (bboxes[i - 1] - bboxes[i - 2])

    gap = 0.0
    for i in range(n):
        if observed[i]:
            gap = 0.0
        else:
            gap += 1.0
        feat[i, ADAPTIVE_KALMAN_FRAMES_SINCE_IDX] = min(gap / max_gap_norm, 1.0)
        feat[i, ADAPTIVE_KALMAN_OBSERVED_IDX] = 1.0 if observed[i] else 0.0
    return feat


def tlbr_to_z(tlbr: Union[list, np.ndarray]) -> np.ndarray:
    """Measurement (4,1): center x, y, area s, aspect ratio r = w/h. Input: xyxy top-left bottom-right."""
    bbox = BBOX.from_tlbr(np.asarray(tlbr, dtype=float).reshape(4))
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    s = w * h
    r = w / (h + 1e-6)
    return np.array([[x], [y], [s], [r]])


def z_to_tlbr(z: np.ndarray) -> np.ndarray:
    """Convert SORT measurement z (4,1) or (4,) to tlbr xyxy in pixels."""
    z = np.asarray(z, dtype=float).reshape(-1)
    x, y, s, r = z[0], z[1], max(z[2], 0.0), max(z[3], 1e-6)
    w = np.sqrt(s * r)
    h = np.sqrt(s / r)
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def bbox_xywh_to_z(bbox_xywh: Union[list, np.ndarray]) -> np.ndarray:
    """Column (4,1) z from center-xy wh box (pixels)."""
    x, y, w, h = np.asarray(bbox_xywh, dtype=float).reshape(4)
    s = w * h
    r = w / (h + 1e-6)
    return np.array([[x], [y], [s], [r]])


def z_to_bbox_xywh(z: np.ndarray) -> BBOX:
    """BBOX xywh from measurement z."""
    z = np.asarray(z, dtype=float).reshape(-1)
    x, y, s, r = z[0], z[1], max(z[2], 0.0), max(z[3], 1e-6)
    w = np.sqrt(s * r)
    h = np.sqrt(s / r)
    return BBOX(np.array([x, y, w, h]))


def get_p_matrix(c: float) -> np.ndarray:
    return np.eye(7)


def get_r_matrix(c: float) -> np.ndarray:
    return np.eye(4)
