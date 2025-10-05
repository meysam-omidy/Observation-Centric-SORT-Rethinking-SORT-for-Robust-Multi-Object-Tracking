import numpy as np
import lap
import time
import pickle

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

def associate(tracks, detections, scores, iou_threshold, iou_coefficient, speed_direction_coefficient, logger, phase):
    if len(tracks) == 0:
        return [], [], [i for i in range(len(detections))]
    elif len(detections) == 0:
        return [], [i for i in range(len(tracks))], []
    
    track_speed_directions = np.array([t.speed_direction for t in tracks])
    track_speed_directions = track_speed_directions.repeat(len(detections)).reshape(-1, len(detections))
    track_tlbrs = np.array([t.tlbr for t in tracks])
    track_previous_obs = np.array([t.k_last_observation for t in tracks])
    speed_directions = batch_speed_direction(track_previous_obs, detections)
    speed_directions_cost = np.abs(speed_directions - track_speed_directions)
    speed_directions_cost = np.where(speed_directions_cost > np.pi, 2 * np.pi - speed_directions_cost, speed_directions_cost) / np.pi
    scores_matrix = np.array(scores).reshape(1, -1).repeat(len(tracks), axis=0)
    speed_directions_cost *= scores_matrix
    mask = (track_previous_obs == [0,0,1,1]).all(axis=1).repeat(len(detections)).reshape(-1, len(detections))
    speed_direction_coefficient_matrix = np.ones_like(speed_directions_cost) * speed_direction_coefficient
    speed_direction_coefficient_matrix = np.where(mask, np.zeros_like(speed_directions_cost), speed_direction_coefficient_matrix)

    iou_cost = 1 - batch_iou(track_tlbrs, detections)
    cost = iou_coefficient * iou_cost + speed_direction_coefficient_matrix * speed_directions_cost
    matched_tracks, unmatched_tracks, unmatched_detections = assignment(cost)
    matchs_to_remove = []
    for i, j in matched_tracks:
        if (1 - iou_cost[i,j]) <= iou_threshold:
            matchs_to_remove.append([i,j])
            unmatched_tracks.append(i)
            unmatched_detections.append(j)
    for i,j in matchs_to_remove:
        matched_tracks.remove([i,j])
    if logger:
        track_ids = [t.id for t in tracks]
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        matched_tracks_ = [[track_ids[i], j+1] for i,j in matched_tracks]
        unmatched_detections_ = [i+1 for i in unmatched_detections]
        matchs_to_remove_ = [[track_ids[i], j] for i,j in matchs_to_remove]
        text = f'PHASE {phase}'
        logger.info(f'{"*" * int((150 - len(text)) / 2)}  {text}  {"*" * int((150 - len(text)) / 2)}')
        logger.info(f'tracks')
        logger.info(f'{np.array2string(track_tlbrs, precision=3, suppress_small=True)}')
        logger.info(f'track ids')
        logger.info(f'{track_ids}')
        logger.info(f'track_speed_directions')
        logger.info(f'{np.array2string(track_speed_directions, precision=3, suppress_small=True)}')
        logger.info(f'detections')
        logger.info(f'{np.array2string(np.array(detections), precision=3, suppress_small=True)}',)
        logger.info(f'scores_matrix')
        logger.info(f'{np.array2string(scores_matrix, precision=3, suppress_small=True)}')
        logger.info(f'iou')
        logger.info(f'{np.array2string(1 - iou_cost, precision=3, suppress_small=True)}')
        logger.info(f'speed_directions_cost')
        logger.info(f'{np.array2string(speed_directions_cost, precision=3, suppress_small=True)}')
        logger.info(f'cost')
        logger.info(f'{np.array2string(cost, precision=3, suppress_small=True)}')
        logger.info(f'matchs_to_remove')
        logger.info(f'{matchs_to_remove_}')
        logger.info(f'matched_tracks')
        logger.info(f'{matched_tracks_}')
        logger.info(f'unmatched_track ids')
        logger.info(f'{unmatched_track_ids}')
        logger.info(f'unmatched_detections')
        logger.info(f'{unmatched_detections_}')
    return matched_tracks, unmatched_tracks, unmatched_detections
    
def select_indices(arr, indices):
    return [arr[index] for index in indices]

def get_dict_item(obj:dict, index:int):
    values = list(obj.values())
    return values[index]

def get_r_matrix(c):
    with open('config/r.pickle', 'rb') as f:
    # with open('config/r.pickle', 'rb') as f:
        r_dict = pickle.load(f)
        c_keys = np.array(list(r_dict.keys()))
        c_dists = np.abs(c_keys - c)
        nearest_c =c_keys[c_dists.argmin()]
        return r_dict[nearest_c]

def get_p_matrix(c):
    with open('config/p.pickle', 'rb') as f:
    # with open('config/p.pickle', 'rb') as f:
        p_dict = pickle.load(f)
        c_keys = np.array(list(p_dict.keys()))
        c_dists = np.abs(c_keys - c)
        nearest_c =c_keys[c_dists.argmin()]
        return p_dict[nearest_c]
    
def print_matrix(x, precision=3):
    print(np.array2string(x, precision=precision, suppress_small=True))
    
def tlbr_to_tlwh(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0]
    o[..., 1] = bbox[..., 1]
    o[..., 2] = bbox[..., 2] - bbox[..., 0]
    o[..., 3] = bbox[..., 3] - bbox[..., 1]
    return o

def tlbr_to_xywh(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    o[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    o[..., 2] = bbox[..., 2] - bbox[..., 0]
    o[..., 3] = bbox[..., 3] - bbox[..., 1]
    return o

def tlwh_to_tlbr(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0]
    o[..., 1] = bbox[..., 1]
    o[..., 2] = bbox[..., 0] + bbox[..., 2]
    o[..., 3] = bbox[..., 1] + bbox[..., 3]
    return o

def tlwh_to_xywh(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0] + bbox[..., 2] / 2
    o[..., 1] = bbox[..., 1] + bbox[..., 3] / 2
    o[..., 2] = bbox[..., 2]
    o[..., 3] = bbox[..., 3]
    return o

def xywh_to_tlwh(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0] - bbox[..., 2] / 2
    o[..., 1] = bbox[..., 1] - bbox[..., 3] / 2
    o[..., 2] = bbox[..., 2]
    o[..., 3] = bbox[..., 3]
    return o

def xywh_to_tlbr(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0] - bbox[..., 2] / 2
    o[..., 1] = bbox[..., 1] - bbox[..., 3] / 2
    o[..., 2] = bbox[..., 0] + bbox[..., 2] / 2
    o[..., 3] = bbox[..., 1] + bbox[..., 3] / 2
    return o

def tlwh_to_z(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0] + bbox[..., 2] / 2
    o[..., 1] = bbox[..., 1] + bbox[..., 3] / 2
    o[..., 2] = bbox[..., 2] * bbox[..., 3]
    o[..., 3] = bbox[..., 2] / bbox[..., 3]
    return o.reshape(-1, 1)

def tlbr_to_z(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    o[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    o[..., 2] = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
    o[..., 3] = (bbox[..., 2] - bbox[..., 0]) / (bbox[..., 3] - bbox[..., 1])
    return o.reshape(-1, 1)

def z_to_tlwh(z:np.ndarray) -> np.ndarray:
    z = z.reshape(-1)
    o = np.zeros_like(z, dtype=float)
    o[..., 0] = z[..., 0] - sqrt(z[..., 2] * z[..., 3]) / 2
    o[..., 1] = z[..., 1] - sqrt(z[..., 2] / z[..., 3]) / 2
    o[..., 2] = sqrt(z[..., 2] * z[..., 3])
    o[..., 3] = sqrt(z[..., 2] / z[..., 3])
    return o[:4]

def z_to_tlbr(z:np.ndarray) -> np.ndarray:
    z = z.reshape(-1)
    o = np.zeros_like(z, dtype=float)
    o[..., 0] = z[..., 0] - sqrt(z[..., 2] * z[..., 3]) / 2
    o[..., 1] = z[..., 1] - sqrt(z[..., 2] / z[..., 3]) / 2
    o[..., 2] = z[..., 0] + sqrt(z[..., 2] * z[..., 3]) / 2
    o[..., 3] = z[..., 1] + sqrt(z[..., 2] / z[..., 3]) / 2
    return o[:4]

def z_to_xywh(z:np.ndarray) -> np.ndarray:
    z = z.reshape(-1)
    o = np.zeros_like(z, dtype=float)
    o[..., 0] = z[..., 0]
    o[..., 1] = z[..., 1]
    o[..., 2] = sqrt(z[..., 2] * z[..., 3])
    o[..., 3] = sqrt(z[..., 2] / z[..., 3])
    return o[:4]

def sqrt(x:np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(x, 0))

