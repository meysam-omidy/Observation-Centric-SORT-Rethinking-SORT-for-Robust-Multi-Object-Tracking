from track import Track
from track_state import STATE_UNCONFIRMED, STATE_TRACKING, STATE_LOST, STATE_DELETED
from utils import associate, select_indices, count_time
from pydantic import BaseModel
import numpy as np

class OCSORTTrackerConfig(BaseModel):
    max_age : int = 30
    min_box_area : int = 10
    max_aspect_ratio : float = 1.6
    high_score_det_threshold : float = 0.6
    low_score_det_threshold : float = 0.1
    init_track_score_threshold : float = 0.6
    match_high_score_dets_with_confirmed_trks_threshold : float = 0.2
    match_low_score_dets_with_confirmed_trks_threshold : float = 0.5
    match_remained_high_score_dets_with_unconfirmed_trks_threshold : float = 0.3
    association_iou_coefficient : float = 1
    association_speed_direction_coefficient : float = 0.3
    delta_t : int = 3
    use_byte : bool = False
    

class OCSORTTracker:
    def __init__(self, config:dict={}):
        self.config = OCSORTTrackerConfig.model_validate(config)
        Track.init(self.config.max_age, self.config.min_box_area, self.config.max_aspect_ratio, self.config.delta_t)

    def update(self, boxes): 
        Track.predict_all()

        high_confidence_detections = boxes[boxes[:, 4] >= self.config.high_score_det_threshold][:, :4]
        high_scores = boxes[boxes[:, 4] >= self.config.high_score_det_threshold][:, 4]
        low_confidence_detections = boxes[np.logical_and(boxes[:, 4] <= self.config.high_score_det_threshold, boxes[:, 4] >= self.config.low_score_det_threshold)][:, :4]
        low_scores = boxes[np.logical_and(boxes[:, 4] <= self.config.high_score_det_threshold, boxes[:, 4] >= self.config.low_score_det_threshold)][:, 4]
        confirmed_tracks = Track.get_tracks([STATE_TRACKING, STATE_LOST])   
        matches, unmatched_confirmed_track_indices, unmatched_high_confidence_detection_indices = associate(
            confirmed_tracks, 
            high_confidence_detections, 
            self.config.match_high_score_dets_with_confirmed_trks_threshold,
            self.config.association_iou_coefficient,
            self.config.association_speed_direction_coefficient
        )
        for t_i, d_i in matches:
            confirmed_tracks[t_i].update(high_confidence_detections[d_i], score=high_scores[d_i])

        if self.config.use_byte:
            remained_confirmed_tracks = select_indices(confirmed_tracks, unmatched_confirmed_track_indices)
            remained_tracking_tracks = [t for t in remained_confirmed_tracks if t.state == STATE_TRACKING]
            matches, unmatched_remained_track_indices, unmatched_low_score_detection_indices = associate(
                remained_tracking_tracks, 
                low_confidence_detections, 
                self.config.match_low_score_dets_with_confirmed_trks_threshold,
                self.config.association_iou_coefficient,
                self.config.association_speed_direction_coefficient
            )
            for t_i, d_i in matches:
                remained_tracking_tracks[t_i].update(low_confidence_detections[d_i], score=low_scores[d_i])

        remained_high_confidence_detections = select_indices(high_confidence_detections, unmatched_high_confidence_detection_indices)
        remained_high_scores = select_indices(high_scores, unmatched_high_confidence_detection_indices)
        unconfirmed_tracks = Track.get_tracks([STATE_UNCONFIRMED])
        matches, unmatched_unconfirmed_track_indices, unmatched_remained_high_score_detection_indices = associate(
            unconfirmed_tracks, 
            remained_high_confidence_detections, 
            self.config.match_remained_high_score_dets_with_unconfirmed_trks_threshold,
            self.config.association_iou_coefficient,
            self.config.association_speed_direction_coefficient
        )
        for t_i, d_i in matches:
            unconfirmed_tracks[t_i].update(remained_high_confidence_detections[d_i], score=remained_high_scores[d_i])
        
        unmatched_remained_high_score_detections = select_indices(remained_high_confidence_detections, unmatched_remained_high_score_detection_indices)
        unmatched_remained_high_scores = select_indices(remained_high_scores, unmatched_remained_high_score_detection_indices)
        for d, s in zip(unmatched_remained_high_score_detections, unmatched_remained_high_scores):
            if s < self.config.init_track_score_threshold:
                continue
            if Track.FRAME_NUMBER == 1:
                Track(d, s, state=STATE_TRACKING)
            else:
                Track(d, s, state=STATE_UNCONFIRMED)
