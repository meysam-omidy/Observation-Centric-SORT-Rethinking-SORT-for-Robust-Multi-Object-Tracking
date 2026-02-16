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
    print(bb1)
    print(bb2)
    bb1 = np.expand_dims(bb1, 1)
    bb2 = np.expand_dims(bb2, 0)
    cx1 = (bb1[..., 0] + bb1[..., 2]) / 2
    cy1 = (bb1[..., 1] + bb1[..., 3]) / 2
    cx2 = (bb2[..., 0] + bb2[..., 2]) / 2
    cy2 = (bb2[..., 1] + bb2[..., 3]) / 2
    print(cx1, cy1, cx2, cy2)
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
