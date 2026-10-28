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

def _nth(iterable, n: int):
    for i, item in enumerate(iterable):
        if i == n:
            return item
    raise IndexError('dict index out of range')


def get_dict_item(obj:dict, index:int):
    # Track histories grow for the whole sequence; walking from the nearer end
    # keeps the common index=-1/-2 lookups O(1) instead of copying every value.
    if index < 0:
        return _nth(reversed(obj.values()), -index - 1)
    return _nth(obj.values(), index)

def get_dict_key(obj:dict, index:int):
    if index < 0:
        return _nth(reversed(obj.keys()), -index - 1)
    return _nth(obj.keys(), index)
    
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
# Optional per-detection context appended after the base layout (feature_dim 18):
# [max IoU with another detection, overlapping-detection count / 10, touches border].
DETECTION_CONTEXT_DIM = 3
ADAPTIVE_KALMAN_CONTEXT_FEATURE_DIM = ADAPTIVE_KALMAN_FEATURE_DIM + DETECTION_CONTEXT_DIM
BORDER_MARGIN_FRACTION = 0.01


def compute_adaptive_kalman_features(bboxes, scores, observed, max_gap_norm=30.0, context=None):
    """
    Per-track feature window for the adaptive Kalman Q/R model (15-D):
    [x, y, w, h, vx..vh, ax..ah, det_score, frames_since_obs, is_observed].
    With ``context`` (n, 3) the window is 18-D: the detection-context columns of
    ``compute_detection_context`` are appended, kept on observed rows only.

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
    dim = ADAPTIVE_KALMAN_FEATURE_DIM if context is None else ADAPTIVE_KALMAN_CONTEXT_FEATURE_DIM
    feat = np.zeros((n, dim), dtype=np.float32)
    if n == 0:
        return feat
    feat[observed, :4] = bboxes[observed]
    feat[observed, ADAPTIVE_KALMAN_SCORE_IDX] = scores[observed]
    if n >= 2:
        velocity = bboxes[1:] - bboxes[:-1]
        has_velocity = observed[1:] & observed[:-1]
        feat[1:, 4:8][has_velocity] = velocity[has_velocity]
        if n >= 3:
            acceleration = velocity[1:] - velocity[:-1]
            has_acceleration = has_velocity[1:] & has_velocity[:-1]
            feat[2:, 8:12][has_acceleration] = acceleration[has_acceleration]

    # Frames since the last observed row (rows before any observation count from
    # the start of the window, exactly like the original running counter).
    index = np.arange(n)
    last_observed = np.maximum.accumulate(np.where(observed, index, -1))
    gap = (index - last_observed).astype(float)
    feat[:, ADAPTIVE_KALMAN_FRAMES_SINCE_IDX] = np.minimum(gap / max_gap_norm, 1.0)
    feat[:, ADAPTIVE_KALMAN_OBSERVED_IDX] = observed
    if context is not None:
        context = np.asarray(context, dtype=float).reshape(n, DETECTION_CONTEXT_DIM)
        feat[observed, ADAPTIVE_KALMAN_FEATURE_DIM:] = context[observed]
    return feat


def compute_detection_context(detections_tlbr, image_width, image_height):
    """Per-detection context for occlusion-aware Q/R (shape (N, 3)).

    Uses every detection of the frame, whatever its score, so it matches the
    training builder which sees the raw detector file:
      0. maximum IoU with any other detection (partial-occlusion cue)
      1. number of other detections it overlaps, divided by 10 and capped at 1
      2. 1.0 when the box lies within 1% of the image size of a border
    Must stay identical to motion-predictor/adaptive_kalman_dataset.py.
    """
    boxes = np.asarray(detections_tlbr, dtype=float).reshape(-1, 4)
    n = len(boxes)
    context = np.zeros((n, DETECTION_CONTEXT_DIM), dtype=float)
    if n == 0:
        return context
    if n > 1:
        overlap = batch_iou(boxes, boxes)
        overlap = np.nan_to_num(overlap, nan=0.0)
        np.fill_diagonal(overlap, 0.0)
        context[:, 0] = overlap.max(axis=1)
        context[:, 1] = np.minimum((overlap > 0).sum(axis=1) / 10.0, 1.0)
    mx = BORDER_MARGIN_FRACTION * float(image_width)
    my = BORDER_MARGIN_FRACTION * float(image_height)
    context[:, 2] = (
        (boxes[:, 0] <= mx) | (boxes[:, 1] <= my)
        | (boxes[:, 2] >= float(image_width) - mx) | (boxes[:, 3] >= float(image_height) - my)
    ).astype(float)
    return context


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
