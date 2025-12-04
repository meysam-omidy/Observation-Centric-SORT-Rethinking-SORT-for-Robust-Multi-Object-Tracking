from track import Track, TrackHistoryItem
from track_state import TrackState, StateUnconfirmed, StateTracking, StateLost, StateDeleted
from utils import select_indices, batch_iou, batch_speed_direction, assignment, compute_motion_features, get_dict_item, BBOX
from pydantic import BaseModel
from motion_predictor import device as DEVICE, model as MODEL
import numpy as np
import torch
import logging

class OCSORTTrackerConfig(BaseModel):
    max_age : int = 30
    update_window : int = 30
    min_box_area : int = 100
    max_aspect_ratio : float = 1.6
    delta_t : int = 3
    high_score_det_threshold : float = 0.6
    low_score_det_threshold : float = 0.1
    init_track_score_threshold : float = 0.6
    match_high_score_dets_with_confirmed_trks_threshold : float = 0.2
    match_low_score_dets_with_confirmed_trks_threshold : float = 0.5
    match_remained_high_score_dets_with_unconfirmed_trks_threshold : float = 0.3
    association_iou_coefficient : float = 1
    association_speed_direction_coefficient : float = 0.2
    image_width : int = 1920
    image_height : int = 1080
    use_byte : bool = False
    log_path : str = None
    

class OCSORTTracker:
    def __init__(self, config:dict={}):
        self.config = OCSORTTrackerConfig.model_validate(config)
        self.tracks : list[Track] = []
        self.frame_number = 0
        self.id_counter = 1
        if self.config.log_path:
            self.logger = logging.getLogger(f"{self.__class__.__name__}-{id(self)}")
            self.logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(self.config.log_path, 'w')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(message)s")
            file_handler.setFormatter(formatter)
            if not self.logger.handlers:
                self.logger.addHandler(file_handler)
        else:
            self.logger = None

    def update(self, boxes): 
        self.frame_number += 1
        self.predict_tracks()

        if self.config.log_path:
            text = f'FRAME {self.frame_number}'
            self.logger.info('')
            self.logger.info(f'{"#" * int((150 - len(text)) / 2)}  {text}  {"#" * int((150 - len(text)) / 2)}')
            self.logger.info('')

        high_confidence_detections = boxes[boxes[:, 4] >= self.config.high_score_det_threshold][:, :4]
        high_scores = boxes[boxes[:, 4] >= self.config.high_score_det_threshold][:, 4]
        low_confidence_detections = boxes[np.logical_and(boxes[:, 4] <= self.config.high_score_det_threshold, boxes[:, 4] >= self.config.low_score_det_threshold)][:, :4]
        low_scores = boxes[np.logical_and(boxes[:, 4] <= self.config.high_score_det_threshold, boxes[:, 4] >= self.config.low_score_det_threshold)][:, 4]
        confirmed_tracks = self.get_tracks([StateTracking, StateLost])   
        matches, unmatched_confirmed_track_indices, unmatched_high_confidence_detection_indices = self.associate(
            confirmed_tracks, 
            high_confidence_detections, 
            high_scores,
            self.config.match_high_score_dets_with_confirmed_trks_threshold,
            phase=1
        )
        for t_i, d_i in matches:
            confirmed_tracks[t_i].update(
                high_confidence_detections[d_i], 
                score=high_scores[d_i]
            )

        if self.config.use_byte:
            remained_confirmed_tracks = select_indices(confirmed_tracks, unmatched_confirmed_track_indices)
            remained_tracking_tracks = [t for t in remained_confirmed_tracks if t.state == StateTracking]
            matches, unmatched_remained_track_indices, unmatched_low_score_detection_indices = self.associate(
                remained_tracking_tracks, 
                low_confidence_detections, 
                low_scores,
                self.config.match_low_score_dets_with_confirmed_trks_threshold,
                phase=2
            )
            for t_i, d_i in matches:
                remained_tracking_tracks[t_i].update(
                    low_confidence_detections[d_i], 
                    score=low_scores[d_i]
                )

        remained_high_confidence_detections = select_indices(high_confidence_detections, unmatched_high_confidence_detection_indices)
        remained_high_scores = select_indices(high_scores, unmatched_high_confidence_detection_indices)
        unconfirmed_tracks = self.get_tracks([StateUnconfirmed])
        matches, unmatched_unconfirmed_track_indices, unmatched_remained_high_score_detection_indices = self.associate(
            unconfirmed_tracks, 
            remained_high_confidence_detections, 
            remained_high_scores,
            self.config.match_remained_high_score_dets_with_unconfirmed_trks_threshold,
            phase=3
        )
        for t_i, d_i in matches:
            unconfirmed_tracks[t_i].update(
                remained_high_confidence_detections[d_i], 
                score=remained_high_scores[d_i]
            )
        
        unmatched_remained_high_score_detections = select_indices(remained_high_confidence_detections, unmatched_remained_high_score_detection_indices)
        unmatched_remained_high_scores = select_indices(remained_high_scores, unmatched_remained_high_score_detection_indices)
        for d, s in zip(unmatched_remained_high_score_detections, unmatched_remained_high_scores):
            if s < self.config.init_track_score_threshold:
                continue
            self.init_track(d, s)

    def init_track(self, bbox, score):
        track_config = {**self.config.model_dump()}
        if self.frame_number == 1:
            self.tracks.append(Track(bbox, score, self.id_counter, self.frame_number, track_config, StateTracking))
        else:
            self.tracks.append(Track(bbox, score, self.id_counter, self.frame_number, track_config, StateUnconfirmed))
        self.id_counter += 1

    def predict_tracks(self):
        tracks = []
        srcs = []
        for track in self.tracks:
            if track.state != StateDeleted:
                track.predict()
                k_last_updates = track.k_last_updates
                if len(k_last_updates) == 1:
                    track.history.predict[track.current_frame] = k_last_updates[0]
                elif len(k_last_updates) < self.config.update_window:
                    diffs = []
                    for i in range(1, len(k_last_updates)):
                        diffs.append(k_last_updates[i].bbox - k_last_updates[i - 1].bbox)
                    track.history.predict[track.current_frame] = TrackHistoryItem(
                        np.array(diffs).mean(axis=0) + k_last_updates[-1].bbox, 
                        1
                    )
                else:
                    boxes = np.array([k_last_update.bbox for k_last_update in k_last_updates])
                    scores = np.array([k_last_update.score for k_last_update in k_last_updates])
                    boxes[:, 0] /= self.config.image_width
                    boxes[:, 1] /= self.config.image_height
                    boxes[:, 2] /= self.config.image_width
                    boxes[:, 3] /= self.config.image_height
                    src = np.zeros(shape=(len(boxes), 13))
                    src[:, :12] = compute_motion_features(boxes)
                    src[:, 12] = scores
                    tracks.append(track)
                    srcs.append(src)
        if len(tracks) > 0:
            srcs = np.array(srcs)
            srcs = torch.tensor(srcs, dtype=torch.float32).to(DEVICE).reshape(srcs.shape[0], srcs.shape[1], -1)
            # trgs = srcs[:, -1:, :]
            with torch.no_grad():
                preds = MODEL.forward(srcs[:, :-1, :], srcs[:, -1:, :]).reshape(srcs.shape[0], 5).cpu().numpy()
            # preds = MODEL.inference(srcs, trgs, 1).reshape(srcs.shape[0], 4).cpu().numpy()
            preds[:, 0] *= self.config.image_width 
            preds[:, 1] *= self.config.image_height
            preds[:, 2] *= self.config.image_width
            preds[:, 3] *= self.config.image_height
            for i, track in enumerate(tracks):
                # track.history.predict[track.current_frame] = preds[i]
                track.history.predict[track.current_frame] = TrackHistoryItem(
                    BBOX(preds[i][:4]),
                    float(preds[i][4].item())
                )
        for track in self.tracks:
            if track.state != StateDeleted and not track.is_valid:
                track.last_state = track.state
                track.state = StateDeleted
                track.exited_frame = self.frame_number - 1

    def get_tracks(self, included_states : list[TrackState] = []):
        return [track for track in self.tracks if track.state in included_states]

    def get_outputs(self):
        outputs = []
        for track in self.tracks:
            s, a = track.bbox.to_xysa()[2:]
            if all([
                track.state in [StateTracking],
                # track.state == StateTracking,
                s >= self.config.min_box_area,
                a <= self.config.max_aspect_ratio
            ]):
                outputs.append(track.mot_format.format(frame_number=int(self.frame_number)))
        return outputs
    
    def associate(self, tracks : list[Track], detections : np.ndarray, scores : np.ndarray, iou_threshold : float, phase : int):
        if len(tracks) == 0:
            return [], [], [i for i in range(len(detections))]
        elif len(detections) == 0:
            return [], [i for i in range(len(tracks))], []
        
        track_speed_directions = np.array([t.speed_direction for t in tracks])
        track_speed_directions = track_speed_directions.repeat(len(detections)).reshape(-1, len(detections))
        track_tlbrs = np.array([t.bbox.to_tlbr() for t in tracks])
        # track_tlbrs = np.array([t.bbox.to_tlbr() for t in tracks])
        track_previous_obs = np.array([t.k_last_observation for t in tracks])
        speed_directions = batch_speed_direction(track_previous_obs, detections)
        speed_directions_cost = np.abs(speed_directions - track_speed_directions)
        speed_directions_cost = np.where(speed_directions_cost > np.pi, 2 * np.pi - speed_directions_cost, speed_directions_cost) / np.pi
        scores_matrix = np.array(scores).reshape(1, -1).repeat(len(tracks), axis=0)
        speed_directions_cost *= scores_matrix
        mask = (track_previous_obs == [0,0,1,1]).all(axis=1).repeat(len(detections)).reshape(-1, len(detections))
        speed_direction_coefficient_matrix = np.ones_like(speed_directions_cost) * self.config.association_speed_direction_coefficient
        speed_direction_coefficient_matrix = np.where(mask, np.zeros_like(speed_directions_cost), speed_direction_coefficient_matrix)

        iou_cost = 1 - batch_iou(track_tlbrs, detections)
        cost = self.config.association_iou_coefficient * iou_cost + speed_direction_coefficient_matrix * speed_directions_cost
        matched_tracks, unmatched_tracks, unmatched_detections = assignment(cost)
        matchs_to_remove = []
        for i, j in matched_tracks:
            if (1 - iou_cost[i,j]) <= iou_threshold:
                matchs_to_remove.append([i,j])
                unmatched_tracks.append(i)
                unmatched_detections.append(j)
        for i,j in matchs_to_remove:
            matched_tracks.remove([i,j])
        if self.logger:
            track_ids = [t.id for t in tracks]
            unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
            matched_tracks_ = [[track_ids[i], j+1] for i,j in matched_tracks]
            unmatched_detections_ = [i+1 for i in unmatched_detections]
            matchs_to_remove_ = [[track_ids[i], j] for i,j in matchs_to_remove]
            text = f'PHASE {phase}'
            self.logger.info(f'{"*" * int((150 - len(text)) / 2)}  {text}  {"*" * int((150 - len(text)) / 2)}')
            self.logger.info(f'tracks')
            self.logger.info(f'{np.array2string(track_tlbrs, precision=3, suppress_small=True)}')
            self.logger.info(f'track ids')
            self.logger.info(f'{track_ids}')
            self.logger.info(f'track_speed_directions')
            self.logger.info(f'{np.array2string(track_speed_directions, precision=3, suppress_small=True)}')
            self.logger.info(f'detections')
            self.logger.info(f'{np.array2string(np.array(detections), precision=3, suppress_small=True)}',)
            self.logger.info(f'scores_matrix')
            self.logger.info(f'{np.array2string(scores_matrix, precision=3, suppress_small=True)}')
            self.logger.info(f'iou')
            self.logger.info(f'{np.array2string(1 - iou_cost, precision=3, suppress_small=True)}')
            self.logger.info(f'speed_directions_cost')
            self.logger.info(f'{np.array2string(speed_directions_cost, precision=3, suppress_small=True)}')
            self.logger.info(f'cost')
            self.logger.info(f'{np.array2string(cost, precision=3, suppress_small=True)}')
            self.logger.info(f'matchs_to_remove')
            self.logger.info(f'{matchs_to_remove_}')
            self.logger.info(f'matched_tracks')
            self.logger.info(f'{matched_tracks_}')
            self.logger.info(f'unmatched_track ids')
            self.logger.info(f'{unmatched_track_ids}')
            self.logger.info(f'unmatched_detections')
            self.logger.info(f'{unmatched_detections_}')
        return matched_tracks, unmatched_tracks, unmatched_detections