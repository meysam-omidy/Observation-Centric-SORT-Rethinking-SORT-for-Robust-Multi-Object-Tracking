from track import Track
from track_state import TrackState, StateUnconfirmed, StateTracking, StateLost, StateDeleted
from utils import (
    select_indices, batch_iou, batch_speed_direction, assignment, compute_motion_features,
    compute_adaptive_kalman_features, get_dict_item, tlbr_to_z, BBOX,
)
from pydantic import BaseModel, Field
from motion_predictor import MotionPredictorConfig, MotionPredictorEngine
from typing import Literal
import numpy as np
import logging

class OCSORTTrackerConfig(BaseModel):
    max_age : int = 30
    update_window_start : int = 20
    update_window_end : int = 50
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
    # Optional covariance-aware association. ``use_mahalanobis_association`` is
    # retained as a legacy switch for both behaviours. New experiments should
    # enable the soft cost and hard gate independently.
    use_mahalanobis_association : bool = False
    use_mahalanobis_cost : bool = False
    use_mahalanobis_gate : bool = False
    mahalanobis_cost_coefficient : float = Field(default=1.0, ge=0)
    # Keep cost normalization independent from the optional hard-gate radius.
    # 9.4877 is the 95% chi-square quantile for a 4-D SORT measurement.
    mahalanobis_cost_reference : float = Field(default=9.4877, gt=0)
    mahalanobis_gate_threshold : float = Field(default=9.4877, gt=0)
    image_width : int = 1920
    image_height : int = 1080
    use_byte : bool = False
    use_oru : bool = False
    use_confidence_r : bool = False
    use_learned_q : bool = True   # False -> ignore the model's var_q, keep the KF's fixed Q
    q_scale : float = Field(default=1.0, gt=0)
    r_scale : float = Field(default=1.0, gt=0)
    # Opt-in online extrapolation output for short confirmed-track occlusions.
    # This changes only what is written to MOT output; association/lifecycle stay
    # unchanged and tracks still obey max_age.
    output_lost_tracks : bool = False
    lost_output_max_age : int = Field(default=3, ge=1)
    lost_output_score_decay : float = Field(default=0.7, gt=0, le=1)
    lost_output_min_score : float = Field(default=0.1, ge=0, le=1)
    lost_output_require_inside_frame : bool = True
    # Debug-only, per-frame association matrices. Kept disabled during normal
    # tracking so no extra matrix copies or memory traffic are incurred.
    collect_association_diagnostics : bool = False
    log_path : str | None = None
    reupdate_type : Literal['constant', 'relative', None] = None
    reupdate_constant_weight : float = 1
    motion : MotionPredictorConfig = Field(default_factory=MotionPredictorConfig)   
    

class OCSORTTracker:
    def __init__(self, config:dict={}, motion_engine: MotionPredictorEngine | None = None):
        self.config = OCSORTTrackerConfig.model_validate(config)
        self.tracks : list[Track] = []
        self.frame_number = 0
        self.id_counter = 1
        self.last_association_diagnostics: list[dict] = []
        self.motion_engine = motion_engine
        if self.config.motion.enabled and self.motion_engine is None:
            try:
                self.motion_engine = MotionPredictorEngine(self.config.motion)
            except FileNotFoundError as err:
                print(f'[OCSORTTracker] {err}; using heuristic motion prediction.')
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
        self.predict_tracks()
        # A consumer such as analyze_tracking.py reads this immediately after
        # update. Retaining only the current frame prevents a diagnostic run
        # from accumulating every dense association matrix in tracker memory.
        self.last_association_diagnostics = []

        if self.config.log_path:
            text = f'FRAME {self.frame_number}'
            self.logger.info('')
            self.logger.info(f'{"#" * int((150 - len(text)) / 2)}  {text}  {"#" * int((150 - len(text)) / 2)}')
            self.logger.info('')

        high_mask = boxes[:, 4] >= self.config.high_score_det_threshold
        low_mask = np.logical_and(
            boxes[:, 4] <= self.config.high_score_det_threshold,
            boxes[:, 4] >= self.config.low_score_det_threshold,
        )
        high_detection_indices = np.flatnonzero(high_mask)
        low_detection_indices = np.flatnonzero(low_mask)
        high_confidence_detections = boxes[high_mask][:, :4]
        high_scores = boxes[high_mask][:, 4]
        low_confidence_detections = boxes[low_mask][:, :4]
        low_scores = boxes[low_mask][:, 4]
        confirmed_tracks = self.get_tracks([StateTracking, StateLost])   
        matches, unmatched_confirmed_track_indices, unmatched_high_confidence_detection_indices = self.associate(
            confirmed_tracks, 
            high_confidence_detections, 
            high_scores,
            self.config.match_high_score_dets_with_confirmed_trks_threshold,
            phase=1,
            detection_indices=high_detection_indices,
        )
        self._update_matches(
            confirmed_tracks, high_confidence_detections, high_scores, matches
        )

        if self.config.use_byte:
            remained_confirmed_tracks = select_indices(confirmed_tracks, unmatched_confirmed_track_indices)
            # remained_tracking_tracks = [t for t in remained_confirmed_tracks if t.state in [StateTracking]]
            remained_tracking_tracks = [t for t in remained_confirmed_tracks if t.state in [StateTracking, StateLost]]
            matches, unmatched_remained_track_indices, unmatched_low_score_detection_indices = self.associate(
                remained_tracking_tracks, 
                low_confidence_detections, 
                low_scores,
                self.config.match_low_score_dets_with_confirmed_trks_threshold,
                phase=2,
                detection_indices=low_detection_indices,
            )
            self._update_matches(
                remained_tracking_tracks, low_confidence_detections, low_scores, matches
            )

        remained_high_confidence_detections = select_indices(high_confidence_detections, unmatched_high_confidence_detection_indices)
        remained_high_scores = select_indices(high_scores, unmatched_high_confidence_detection_indices)
        remained_high_detection_indices = select_indices(
            high_detection_indices, unmatched_high_confidence_detection_indices
        )
        unconfirmed_tracks = self.get_tracks([StateUnconfirmed])
        matches, unmatched_unconfirmed_track_indices, unmatched_remained_high_score_detection_indices = self.associate(
            unconfirmed_tracks, 
            remained_high_confidence_detections, 
            remained_high_scores,
            self.config.match_remained_high_score_dets_with_unconfirmed_trks_threshold,
            phase=3,
            detection_indices=np.asarray(remained_high_detection_indices, dtype=int),
        )
        self._update_matches(
            unconfirmed_tracks,
            remained_high_confidence_detections,
            remained_high_scores,
            matches,
        )
        
        unmatched_remained_high_score_detections = select_indices(remained_high_confidence_detections, unmatched_remained_high_score_detection_indices)
        unmatched_remained_high_scores = select_indices(remained_high_scores, unmatched_remained_high_score_detection_indices)
        for d, s in zip(unmatched_remained_high_score_detections, unmatched_remained_high_scores):
            if s < self.config.init_track_score_threshold:
                continue
            self.init_track(d, s)

    def init_track(self, bbox, score):
        track_config = {**self.config.model_dump()}
        track_config['use_kalman'] = self.config.motion.use_kalman
        track_config['kalman_fusion_blend'] = self.config.motion.kalman_fusion_blend
        if self.frame_number == 1:
            self.tracks.append(Track(bbox, score, self.id_counter, self.frame_number, track_config, StateTracking))
        else:
            self.tracks.append(Track(bbox, score, self.id_counter, self.frame_number, track_config, StateUnconfirmed))
        self.id_counter += 1

    @staticmethod
    def _cv_bbox(k_last_updates) -> np.ndarray:
        """Constant-velocity extrapolation: mean frame-to-frame diff + last bbox."""
        diffs = [
            k_last_updates[i].bbox - k_last_updates[i - 1].bbox
            for i in range(1, len(k_last_updates))
        ]
        return np.array(diffs).mean(axis=0) + k_last_updates[-1].bbox

    @property
    def _use_adaptive_kalman(self) -> bool:
        return (
            self.motion_engine is not None
            and self.motion_engine.cfg.model_type == 'adaptive_kalman'
        )

    def _adaptive_context(self, track):
        """Fixed-size completed-frame context matching training exactly."""
        history_len = self.motion_engine.history_len
        items = (
            track.history_items(history_len)
            if hasattr(track, 'history_items')
            else track.k_last_updates[-history_len:]
        )
        if len(items) < history_len:
            return None
        items = items[-history_len:]
        boxes = np.asarray([item.bbox for item in items], dtype=float)
        boxes /= np.array(
            [self.config.image_width, self.config.image_height,
             self.config.image_width, self.config.image_height],
            dtype=float,
        )
        scores = np.asarray([item.score for item in items], dtype=float)
        observed = np.asarray([item.observed for item in items], dtype=bool)
        features = compute_adaptive_kalman_features(
            boxes, scores, observed, self.motion_engine.max_gap_norm
        )
        return boxes, scores, observed, features

    def _measurement_feature(self, context, detection, score):
        boxes, scores, observed, _ = context
        current = np.asarray(BBOX.from_tlbr(detection), dtype=float)
        current /= np.array(
            [self.config.image_width, self.config.image_height,
             self.config.image_width, self.config.image_height],
            dtype=float,
        )
        ext = compute_adaptive_kalman_features(
            np.concatenate([boxes, current[None]], axis=0),
            np.concatenate([scores, [float(score)]]),
            np.concatenate([observed, [True]]),
            self.motion_engine.max_gap_norm,
        )
        return ext[-1]

    def _update_matches(self, tracks, detections, scores, matches):
        """Condition R on each matched current detection, then perform KF update."""
        var_rs = [None] * len(matches)
        if self._use_adaptive_kalman and not self.config.use_confidence_r and matches:
            rows, measurements, valid_positions = [], [], []
            for pos, (t_i, d_i) in enumerate(matches):
                context = self._adaptive_context(tracks[t_i])
                if context is None:
                    continue
                rows.append(context[3])
                measurements.append(
                    self._measurement_feature(
                        context, detections[d_i], float(scores[d_i])
                    )
                )
                valid_positions.append(pos)
            if rows:
                batched = np.stack(rows).astype(np.float32, copy=False)
                lens = [self.motion_engine.history_len] * len(rows)
                predicted = self.motion_engine.predict_r_batch(
                    batched, lens, np.stack(measurements)
                ).cpu().numpy()
                for pos, var_r in zip(valid_positions, predicted):
                    var_rs[pos] = var_r

        for pos, (t_i, d_i) in enumerate(matches):
            tracks[t_i].update(
                detections[d_i],
                score=float(scores[d_i]),
                var_r=var_rs[pos],
            )

    def predict_tracks(self):
        self.frame_number += 1
        active_tracks = self.get_tracks([StateTracking, StateLost, StateUnconfirmed])
        for track in active_tracks:
            track.predict()

        # A motion window only exists to provide a model with enough history.  If
        # no model is available, bypass feature construction/batching entirely.
        # Two completed states are sufficient for the CV fallback and, with the
        # normal Kalman blend of zero, the KF prediction remains the final box.
        if self.motion_engine is None:
            for track in active_tracks:
                recent_items = track.history_items(2)
                if len(recent_items) == 1:
                    bbox = recent_items[0].bbox
                    score = float(recent_items[0].score)
                else:
                    bbox = self._cv_bbox(recent_items)
                    score = track.score
                track.set_prediction_from_motion(
                    bbox,
                    score,
                    var_q=None,
                    var_r=None,
                )
            self._delete_invalid_tracks()
            return

        use_adaptive = self._use_adaptive_kalman

        tracks_batch = []
        cv_bboxes = []
        srcs = []
        valid_lens = []
        prediction_gaps = []
        max_len = 0
        for track in active_tracks:
            k_last_updates = track.k_last_updates
            adaptive_context = self._adaptive_context(track) if use_adaptive else None
            if len(k_last_updates) == 1:
                it0 = k_last_updates[0]
                track.set_prediction_from_motion(
                    it0.bbox,
                    float(it0.score),
                    var_q=None,
                    var_r=None,
                )
            elif (
                adaptive_context is None
                if use_adaptive
                else len(k_last_updates) < self.config.update_window_start
            ):
                track.set_prediction_from_motion(
                    self._cv_bbox(k_last_updates),
                    track.score,
                    var_q=None,
                    var_r=None,
                )
            else:
                tracks_batch.append(track)
                cv_bboxes.append(self._cv_bbox(k_last_updates))
                if use_adaptive:
                    srcs.append(adaptive_context[3])
                    L = self.motion_engine.history_len
                    # track.predict() has already incremented age for this frame.
                    prediction_gaps.append(
                        min(track.age / self.motion_engine.max_gap_norm, 1.0)
                    )
                else:
                    boxes = np.array([item.bbox for item in k_last_updates])
                    scores = np.array([item.score for item in k_last_updates])
                    boxes[:, 0] /= self.config.image_width
                    boxes[:, 1] /= self.config.image_height
                    boxes[:, 2] /= self.config.image_width
                    boxes[:, 3] /= self.config.image_height
                    L = len(boxes)
                    srcs.append((compute_motion_features(boxes), scores))
                valid_lens.append(L)
                max_len = max(max_len, L)

        if len(tracks_batch) > 0:
            if use_adaptive:
                # Fixed length from the checkpoint: positional encodings no longer
                # depend on which other tracks happen to share this frame's batch.
                batched = np.stack(srcs).astype(np.float32, copy=False)
            else:
                batched = np.zeros(
                    shape=(len(tracks_batch), max_len, 13), dtype=np.float32
                )
                for i, L in enumerate(valid_lens):
                    motion12, scores = srcs[i]
                    batched[i, :L, :12] = motion12
                    batched[i, :L, 12] = scores

            if use_adaptive:
                preds_t = None
                var_q_t = self.motion_engine.predict_q_batch(
                    batched, valid_lens, np.asarray(prediction_gaps, dtype=np.float32)
                )
                var_r_t = None
            else:
                preds_t, var_q_t, var_r_t = (
                    self.motion_engine.predict_batch(batched, valid_lens)
                    if self.motion_engine is not None
                    else (None, None, None)
                )
            preds = None
            if preds_t is not None:
                preds = preds_t.cpu().numpy()
                preds[:, 0] *= self.config.image_width
                preds[:, 1] *= self.config.image_height
                preds[:, 2] *= self.config.image_width
                preds[:, 3] *= self.config.image_height
            var_q_np = var_q_t.cpu().numpy() if var_q_t is not None else None
            var_r_np = var_r_t.cpu().numpy() if var_r_t is not None else None

            for i, track in enumerate(tracks_batch):
                vq = None if (var_q_np is None or not self.config.use_learned_q) else var_q_np[i]
                vr = None if var_r_np is None else var_r_np[i]
                if preds is not None:
                    xywh, score = preds[i][:4], float(preds[i][4].item())
                else:
                    xywh, score = cv_bboxes[i], track.score
                track.set_prediction_from_motion(xywh, score, var_q=vq, var_r=vr)
        self._delete_invalid_tracks()

    def _delete_invalid_tracks(self):
        """Mark tracks invalidated by the current prediction step as deleted."""
        for track in self.tracks:
            if track.state != StateDeleted and not track.is_valid:
                track.last_state = track.state
                track.state = StateDeleted
                track.exited_frame = self.frame_number - 1

    def get_tracks(self, included_states : list[TrackState] = []):
        return [track for track in self.tracks if track.state in included_states]
    
    def show_tracks(self):
        for t in self.tracks:
            print(t.clean_format)

    def get_outputs(self):
        outputs = []
        for track in self.tracks:
            s, a = track.bbox.to_xysa()[2:]
            is_current_detection = track.current_frame in track.history.update
            if track.state == StateTracking and (
                is_current_detection or not self.config.output_lost_tracks
            ):
                # With the feature disabled, normal output remains byte-for-byte
                # equivalent to the old path (including the first missed frame).
                output = track.mot_format
            elif (
                self.config.output_lost_tracks
                and track.state in [StateTracking, StateLost]
                and not is_current_detection
                and 1 <= track.age <= self.config.lost_output_max_age
            ):
                # track.bbox is this frame's KF/CV prediction. Decay the last real
                # detection confidence each missed frame, and stop once it becomes
                # too weak to be a credible online output.
                predicted_score = track.score * (
                    self.config.lost_output_score_decay ** track.age
                )
                if predicted_score < self.config.lost_output_min_score:
                    continue
                if self.config.lost_output_require_inside_frame:
                    x1, y1, x2, y2 = track.bbox.to_tlbr()
                    if x1 < 0 or y1 < 0 or x2 > self.config.image_width or y2 > self.config.image_height:
                        # A partial out-of-frame prediction likely represents an
                        # exit, not an occlusion. Do not emit a ghost detection.
                        continue
                output = track.mot_format_for(track.bbox, predicted_score)
            else:
                continue

            if s >= self.config.min_box_area and a <= self.config.max_aspect_ratio:
                outputs.append(output)
        return outputs

    def _mahalanobis_distances(
        self,
        tracks: list[Track],
        detections: np.ndarray,
        scores: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> np.ndarray:
        """Squared innovation distance for every track/detection candidate.

        ``Track.set_prediction_from_motion`` has already advanced each KF, so
        ``x`` and ``P`` here are its current predicted state and covariance.
        The confidence-R mode uses the same candidate-specific R used by
        ``Track.update``.  Learned R is produced only after association in the
        current model contract, so its last/fixed KF R remains the fallback here.
        """
        distances = np.full((len(tracks), len(detections)), np.nan, dtype=float)
        measurements = np.asarray([tlbr_to_z(box).reshape(4) for box in detections])
        fixed_r_tracks = []
        confidence_r_tracks = []
        for track_index, track in enumerate(tracks):
            candidate_indices = np.flatnonzero(candidate_mask[track_index])
            if len(candidate_indices) == 0:
                continue
            if track.kf is None:
                distances[track_index, candidate_indices] = 0.0
                continue
            if track.config.use_confidence_r:
                confidence_r_tracks.append((track_index, track, candidate_indices))
            else:
                fixed_r_tracks.append((track_index, track, candidate_indices))

        if fixed_r_tracks:
            # Fixed R is by far the common learned-Kalman path.  Batch all
            # tracks into one linear-algebra call rather than factorizing once
            # per track in Python.
            indices = np.asarray([item[0] for item in fixed_r_tracks])
            kfs = [item[1].kf for item in fixed_r_tracks]
            try:
                predicted_measurements = np.stack(
                    [(kf.H @ kf.x).reshape(4) for kf in kfs]
                )
                predicted_covariances = np.stack(
                    [kf.H @ kf.P @ kf.H.T + kf.R for kf in kfs]
                )
                predicted_covariances = 0.5 * (
                    predicted_covariances
                    + np.swapaxes(predicted_covariances, -1, -2)
                )
                if not np.isfinite(predicted_covariances).all():
                    raise np.linalg.LinAlgError("non-finite innovation covariance")
                predicted_covariances += np.eye(4) * 1e-6
                innovations = measurements[None, :, :] - predicted_measurements[:, None, :]
                solved = np.linalg.solve(
                    predicted_covariances,
                    np.swapaxes(innovations, 1, 2),
                )
                solved = np.swapaxes(solved, 1, 2)
                all_distances = np.einsum('...i,...i->...', innovations, solved)
                all_distances = np.where(
                    np.isfinite(all_distances) & (all_distances >= 0),
                    all_distances,
                    np.inf,
                )
                distances[indices] = np.where(
                    candidate_mask[indices], all_distances, np.nan
                )
            except np.linalg.LinAlgError:
                for track_index, _, candidate_indices in fixed_r_tracks:
                    distances[track_index, candidate_indices] = np.inf

        for track_index, track, candidate_indices in confidence_r_tracks:
            # Candidate R changes by score, but its candidate matrices are
            # still solved in a single NumPy batch for this track.
            kf = track.kf
            innovations = measurements[candidate_indices] - (kf.H @ kf.x).reshape(4)
            try:
                predicted_covariance = kf.H @ kf.P @ kf.H.T
                base_r = np.diag([1.0, 1.0, 10.0, 10.0])
                scales = np.exp(2.0 * (1.0 - scores[candidate_indices]))
                innovation_covariances = (
                    predicted_covariance[None, :, :]
                    + scales[:, None, None] * base_r
                )
                innovation_covariances = 0.5 * (
                    innovation_covariances
                    + np.swapaxes(innovation_covariances, -1, -2)
                )
                if not np.isfinite(innovation_covariances).all():
                    raise np.linalg.LinAlgError("non-finite innovation covariance")
                innovation_covariances += np.eye(4) * 1e-6
                solved = np.linalg.solve(
                    innovation_covariances, innovations[..., None]
                ).squeeze(-1)
                candidate_distances = np.einsum('ij,ij->i', innovations, solved)
                distances[track_index, candidate_indices] = np.where(
                    np.isfinite(candidate_distances) & (candidate_distances >= 0),
                    candidate_distances,
                    np.inf,
                )
            except np.linalg.LinAlgError:
                distances[track_index, candidate_indices] = np.inf
        return distances
    
    def _record_association_diagnostic(
        self,
        phase: int,
        tracks: list[Track],
        detections: np.ndarray,
        scores: np.ndarray,
        detection_indices: np.ndarray,
        iou: np.ndarray,
        direction_cost: np.ndarray,
        cost: np.ndarray,
        valid_pairs: np.ndarray,
        matches: list[list[int]],
    ) -> None:
        """Save the current phase's matrices for an opt-in diagnostics consumer."""
        if not self.config.collect_association_diagnostics:
            return
        self.last_association_diagnostics.append({
            "frame": int(self.frame_number),
            "phase": int(phase),
            "track_ids": [int(track.id) for track in tracks],
            "track_states": [track.state.name for track in tracks],
            # Indices refer to the unfiltered ``boxes`` argument of update().
            "detection_indices": np.asarray(detection_indices, dtype=int).copy(),
            "scores": np.asarray(scores, dtype=float).copy(),
            "iou": np.asarray(iou, dtype=float).copy(),
            "direction_cost": np.asarray(direction_cost, dtype=float).copy(),
            "cost": np.asarray(cost, dtype=float).copy(),
            "valid_pairs": np.asarray(valid_pairs, dtype=bool).copy(),
            "matches": [list(map(int, match)) for match in matches],
        })

    def associate(
        self,
        tracks: list[Track],
        detections: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        phase: int,
        detection_indices: np.ndarray | None = None,
    ):
        detections = np.asarray(detections, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if detection_indices is None:
            detection_indices = np.arange(len(detections), dtype=int)
        else:
            detection_indices = np.asarray(detection_indices, dtype=int).reshape(-1)
        if len(detection_indices) != len(detections):
            raise ValueError("detection_indices must align with detections")
        if len(tracks) == 0:
            matches, unmatched_tracks, unmatched_detections = [], [], list(range(len(detections)))
            self._record_association_diagnostic(
                phase, tracks, detections, scores, detection_indices,
                np.empty((0, len(detections))), np.empty((0, len(detections))),
                np.empty((0, len(detections))), np.zeros((0, len(detections)), dtype=bool),
                matches,
            )
            return matches, unmatched_tracks, unmatched_detections
        elif len(detections) == 0:
            matches, unmatched_tracks, unmatched_detections = [], list(range(len(tracks))), []
            self._record_association_diagnostic(
                phase, tracks, detections, scores, detection_indices,
                np.empty((len(tracks), 0)), np.empty((len(tracks), 0)),
                np.empty((len(tracks), 0)), np.zeros((len(tracks), 0), dtype=bool),
                matches,
            )
            return matches, unmatched_tracks, unmatched_detections
        
        track_speed_directions = np.array([t.speed_direction for t in tracks])
        track_speed_directions = track_speed_directions.repeat(len(detections)).reshape(-1, len(detections))
        track_tlbrs = np.array([t.bbox.to_tlbr() for t in tracks])
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
        iou_valid_pairs = (1 - iou_cost) > iou_threshold
        cost = self.config.association_iou_coefficient * iou_cost + speed_direction_coefficient_matrix * speed_directions_cost
        # The original flag remains a compatibility alias for the old coupled
        # behaviour.  New callers can evaluate the ranking cost and rejection
        # gate separately.
        use_mahalanobis_cost = (
            self.config.use_mahalanobis_association
            or self.config.use_mahalanobis_cost
        )
        use_mahalanobis_gate = (
            self.config.use_mahalanobis_association
            or self.config.use_mahalanobis_gate
        )
        mahalanobis_distances = None
        invalid_mahalanobis_pairs = np.zeros_like(iou_cost, dtype=bool)
        if use_mahalanobis_cost or use_mahalanobis_gate:
            mahalanobis_distances = self._mahalanobis_distances(
                tracks, detections, scores, iou_valid_pairs
            )
            if use_mahalanobis_gate:
                valid_mahalanobis_pairs = iou_valid_pairs & (
                    mahalanobis_distances <= self.config.mahalanobis_gate_threshold
                )
                # A fixed SORT covariance can be temporarily miscalibrated after
                # sharp motion/scale change. If it rejects every geometrically
                # viable candidate for a track, retain the original association
                # row rather than creating short-lived replacement tracks.
                enforce_mahalanobis_gate = valid_mahalanobis_pairs.any(axis=1)
                invalid_mahalanobis_pairs = (
                    mahalanobis_distances > self.config.mahalanobis_gate_threshold
                ) & enforce_mahalanobis_gate[:, None]
            if use_mahalanobis_cost:
                # A non-finite innovation distance means this covariance could
                # not score the pair reliably. In soft-cost mode it must not
                # turn into NumPy's maximum float (an accidental hard gate);
                # fall back to the existing IoU/direction association instead.
                # Explicit gate mode still handles non-finite values as invalid
                # through ``invalid_mahalanobis_pairs`` above.
                mahalanobis_cost = np.where(
                    np.isfinite(mahalanobis_distances),
                    mahalanobis_distances,
                    0.0,
                )
                # A fixed statistical reference keeps the cost coefficient
                # interpretable when the optional gate threshold is swept.
                cost += (
                    self.config.mahalanobis_cost_coefficient
                    * mahalanobis_cost
                    / self.config.mahalanobis_cost_reference
                )

        invalid_pairs = ~iou_valid_pairs
        invalid_pairs |= invalid_mahalanobis_pairs
        # The assignment must still be solved globally; invalid assignments are
        # removed below.  A large cost prevents them displacing valid candidates.
        cost = np.where(invalid_pairs, 1e6, cost)
        matched_tracks, unmatched_tracks, unmatched_detections = assignment(cost)
        matchs_to_remove = []
        for i, j in matched_tracks:
            if invalid_pairs[i, j]:
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
            matchs_to_remove_ = [[track_ids[i], j+1] for i,j in matchs_to_remove]
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
            if mahalanobis_distances is not None:
                self.logger.info('mahalanobis_squared')
                self.logger.info(f'{np.array2string(mahalanobis_distances, precision=3, suppress_small=True)}')
                self.logger.info('mahalanobis_gated_pairs')
                self.logger.info(f'{np.array2string(invalid_mahalanobis_pairs)}')
                self.logger.info('mahalanobis_cost')
                self.logger.info(f'{np.array2string(mahalanobis_cost)}')
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
        self._record_association_diagnostic(
            phase,
            tracks,
            detections,
            scores,
            detection_indices,
            1 - iou_cost,
            speed_directions_cost,
            cost,
            ~invalid_pairs,
            matched_tracks,
        )
        return matched_tracks, unmatched_tracks, unmatched_detections
