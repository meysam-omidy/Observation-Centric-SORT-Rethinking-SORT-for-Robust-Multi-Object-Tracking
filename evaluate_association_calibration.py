"""Validate pre-association Kalman covariance calibration on DanceTrack.

This evaluator replays the production adaptive-Kalman tracker with *oracle*
one-to-one GT/detection correspondences.  Ground truth is used only after a
causal prediction to identify the correct current detection; it is never fed to
the motion model.  For every real matched detection, it records the squared
2-D center-position innovation distance:

    d2 = (z_xy - H_xy x_pred)^T (H_xy P_pred H_xy^T + R_xy)^-1
         (z_xy - H_xy x_pred)

For calibrated two-dimensional Gaussian innovations, d2 follows chi-square(2):
mean=2, median=1.386, and 95% of matches should be <=5.991.

Example (run from this directory):
  python evaluate_association_calibration.py \
    --weights_path ../motion-predictor/checkpoints/.../best_model.pth \
    --seqs dancetrack0001 dancetrack0002 \
    --calibration_seqs dancetrack0001 \
    --evaluation_seqs dancetrack0002
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import lap
import numpy as np

from kalman_filter import learned_measurement_noise_matrix
from motion_predictor import MotionPredictorConfig, MotionPredictorEngine
from ocsort import OCSORTTracker
from track_state import StateDeleted, StateLost, StateTracking, StateUnconfirmed
from utils import BBOX, batch_iou, batch_speed_direction, tlbr_to_z


# Chi-square quantiles for 2 degrees of freedom.  Keeping these constants avoids
# requiring scipy just to generate a calibration report.
CHI2_2D = {
    "mean": 2.0,
    "q50": 1.3862943611,
    "q90": 4.6051701860,
    "q95": 5.9914645471,
    "q99": 9.2103403720,
}


# These methods answer a deliberately local question: given one causal track
# prediction and the detections that the production phase-1 association would
# consider for it, does a cost rank the oracle-correct detection ahead of its
# alternatives?  Global Hungarian effects are intentionally excluded here.
RANKING_METHODS = (
    "iou_only",
    "tracker_iou_direction",
    "mahalanobis_xy",
    "tracker_plus_capped_mahalanobis_xy",
)


def _tlwh_to_tlbr(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float).reshape(-1, 4)
    out = rows.copy()
    out[:, 2] += out[:, 0]
    out[:, 3] += out[:, 1]
    return out


def _one_to_one_matches(
    gt_boxes: np.ndarray,
    det_boxes: np.ndarray,
    min_iou: float,
) -> list[tuple[int, int]]:
    """Hungarian-like one-to-one IoU matching, then reject weak pairs."""
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return []
    ious = batch_iou(gt_boxes, det_boxes)
    _, gt_to_det, _ = lap.lapjv(1.0 - ious, extend_cost=True)
    return [
        (gt_index, int(det_index))
        for gt_index, det_index in enumerate(gt_to_det)
        if det_index >= 0 and ious[gt_index, det_index] >= min_iou
    ]


def _load_gt_by_frame(sequence_dir: Path) -> dict[int, list[tuple[int, np.ndarray]]]:
    rows = np.loadtxt(sequence_dir / "gt" / "gt.txt", delimiter=",")
    if rows.ndim == 1:
        rows = rows[None, :]
    # The MOT-format DanceTrack labels use conf/class columns.  Retain the same
    # considered-object convention as the real-detection training dataset.
    if rows.shape[1] >= 7:
        rows = rows[rows[:, 6] == 1]
    if rows.shape[1] >= 8:
        rows = rows[rows[:, 7] == 1]

    by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for row in rows:
        frame, gt_id = int(row[0]), int(row[1])
        x, y, w, h = row[2:6]
        by_frame[frame].append(
            (gt_id, np.array([x, y, x + w, y + h], dtype=float))
        )
    return dict(by_frame)


def _load_detections_by_frame(path: Path) -> dict[int, np.ndarray]:
    rows = np.loadtxt(path, delimiter=",")
    if rows.ndim == 1:
        rows = rows[None, :]
    by_frame: dict[int, np.ndarray] = {}
    for frame in np.unique(rows[:, 0].astype(int)):
        frame_rows = rows[rows[:, 0].astype(int) == frame]
        # Stored files are frame,x1,y1,x2,y2,score.
        by_frame[frame] = frame_rows[:, 1:6].astype(float, copy=False)
    return by_frame


def _crowding_by_gt_id(gt_items: list[tuple[int, np.ndarray]]) -> dict[int, float]:
    """Maximum GT-box overlap with another object at this frame."""
    if len(gt_items) < 2:
        return {gt_id: 0.0 for gt_id, _ in gt_items}
    boxes = np.stack([box for _, box in gt_items])
    ious = batch_iou(boxes, boxes)
    np.fill_diagonal(ious, 0.0)
    return {
        gt_id: float(ious[index].max())
        for index, (gt_id, _) in enumerate(gt_items)
    }


def _bucket_gap(age: int) -> str:
    if age <= 1:
        return "gap_1"
    if age <= 3:
        return "gap_2_3"
    return "gap_4_plus"


def _bucket_score(score: float) -> str:
    if score < 0.5:
        return "score_lt_0_5"
    if score < 0.75:
        return "score_0_5_0_75"
    return "score_ge_0_75"


def _summary(values: Iterable[float], scale: float = 1.0) -> dict:
    d2 = np.asarray(list(values), dtype=float)
    d2 = d2[np.isfinite(d2)] / float(scale)
    if len(d2) == 0:
        return {"count": 0}
    return {
        "count": int(len(d2)),
        "mean_d2": float(d2.mean()),
        "median_d2": float(np.median(d2)),
        "q90_d2": float(np.quantile(d2, 0.90)),
        "q95_d2": float(np.quantile(d2, 0.95)),
        "q99_d2": float(np.quantile(d2, 0.99)),
        "coverage_50": float(np.mean(d2 <= CHI2_2D["q50"])),
        "coverage_90": float(np.mean(d2 <= CHI2_2D["q90"])),
        "coverage_95": float(np.mean(d2 <= CHI2_2D["q95"])),
        "coverage_99": float(np.mean(d2 <= CHI2_2D["q99"])),
        "expected": {
            "mean_d2": CHI2_2D["mean"],
            "median_d2": CHI2_2D["q50"],
            "q90_d2": CHI2_2D["q90"],
            "q95_d2": CHI2_2D["q95"],
            "q99_d2": CHI2_2D["q99"],
            "coverage_50": 0.50,
            "coverage_90": 0.90,
            "coverage_95": 0.95,
            "coverage_99": 0.99,
        },
    }


def _group_summaries(records: list[dict], key: str, scale: float = 1.0) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record["d2_xy"])
    return {name: _summary(values, scale=scale) for name, values in sorted(groups.items())}


def _rank_costs(costs: np.ndarray, correct_index: int) -> dict | None:
    """Summarize the oracle candidate's rank for one lower-is-better cost."""
    costs = np.asarray(costs, dtype=float).reshape(-1)
    if len(costs) == 0 or not np.isfinite(costs[correct_index]):
        return None
    correct_cost = float(costs[correct_index])
    other_indices = np.arange(len(costs)) != int(correct_index)
    other_costs = costs[other_indices]
    # A tiny absolute tolerance prevents insignificant floating-point noise
    # from turning a real tie into an arbitrary strict preference.
    tolerance = 1e-12
    better = int(np.sum(other_costs < correct_cost - tolerance))
    tied = int(np.sum(np.abs(other_costs - correct_cost) <= tolerance))
    rank = better + 1
    finite_other_costs = other_costs[np.isfinite(other_costs)]
    margin = (
        float(finite_other_costs.min() - correct_cost)
        if len(finite_other_costs)
        else None
    )
    top_index = int(np.argmin(costs))
    return {
        # ``rank`` is tie-inclusive: rank=1 means no candidate is strictly
        # preferred. ``strict_top_1`` is the stronger, unambiguous condition.
        "rank": rank,
        "top_1": bool(rank == 1),
        "strict_top_1": bool(rank == 1 and tied == 0),
        "reciprocal_rank": 1.0 / rank,
        # Ties receive half credit in pairwise comparisons.
        "pairwise_preference": float(
            (np.sum(other_costs > correct_cost + tolerance) + 0.5 * tied)
            / max(len(other_costs), 1)
        ),
        "margin_to_nearest_competitor": margin,
        "top_candidate_local_index": top_index,
    }


def _ranking_summary(records: list[dict]) -> dict:
    """Aggregate local-ranking metrics without writing every candidate to JSON."""
    if not records:
        return {"count": 0}
    result = {
        "count": len(records),
        "mean_candidate_count": float(np.mean([r["candidate_count"] for r in records])),
        "candidate_count_ge_2_fraction": float(
            np.mean([r["candidate_count"] >= 2 for r in records])
        ),
        "methods": {},
    }
    for method in RANKING_METHODS:
        values = [r["methods"][method] for r in records if r["methods"][method] is not None]
        if not values:
            result["methods"][method] = {"count": 0}
            continue
        margins = [v["margin_to_nearest_competitor"] for v in values]
        finite_margins = [m for m in margins if m is not None and math.isfinite(m)]
        result["methods"][method] = {
            "count": len(values),
            "top_1_rate": float(np.mean([v["top_1"] for v in values])),
            "strict_top_1_rate": float(np.mean([v["strict_top_1"] for v in values])),
            "mean_rank": float(np.mean([v["rank"] for v in values])),
            "median_rank": float(np.median([v["rank"] for v in values])),
            "mean_reciprocal_rank": float(np.mean([v["reciprocal_rank"] for v in values])),
            "pairwise_preference_rate": float(
                np.mean([v["pairwise_preference"] for v in values])
            ),
            "mean_margin_to_nearest_competitor": (
                float(np.mean(finite_margins)) if finite_margins else None
            ),
        }
    return result


def _ranking_group_summaries(records: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {
        name: _ranking_summary(group_records)
        for name, group_records in sorted(groups.items())
    }


def _ranking_examples(records: list[dict], limit: int) -> dict:
    """Keep a few actionable ranking disagreements without bloating the report."""
    def method(record: dict, name: str) -> dict | None:
        return record["methods"].get(name)

    def rescued_by(candidate_method: str) -> list[dict]:
        examples = [
            record for record in records
            if method(record, "tracker_iou_direction") is not None
            and method(record, candidate_method) is not None
            and not method(record, "tracker_iou_direction")["top_1"]
            and method(record, candidate_method)["top_1"]
        ]
        return sorted(
            examples,
            key=lambda r: (
                -r["methods"]["tracker_iou_direction"]["rank"],
                -r["candidate_count"],
                -r["crowding_iou"],
            ),
        )[:limit]

    def harmed_by(candidate_method: str) -> list[dict]:
        examples = [
            record for record in records
            if method(record, "tracker_iou_direction") is not None
            and method(record, candidate_method) is not None
            and method(record, "tracker_iou_direction")["top_1"]
            and not method(record, candidate_method)["top_1"]
        ]
        return sorted(
            examples,
            key=lambda r: (
                -r["methods"][candidate_method]["rank"],
                -r["candidate_count"],
                -r["crowding_iou"],
            ),
        )[:limit]

    return {
        "mahalanobis_rescues": rescued_by("mahalanobis_xy"),
        "mahalanobis_harms": harmed_by("mahalanobis_xy"),
        "combined_rescues": rescued_by("tracker_plus_capped_mahalanobis_xy"),
        "combined_harms": harmed_by("tracker_plus_capped_mahalanobis_xy"),
    }


def _tracker_config(args, image_width: int, image_height: int, motion: MotionPredictorConfig) -> dict:
    return {
        "image_width": image_width,
        "image_height": image_height,
        "max_age": args.max_age,
        "update_window_start": args.update_window_start,
        "update_window_end": args.update_window_end,
        "high_score_det_threshold": args.high_score_det_threshold,
        "low_score_det_threshold": args.low_score_det_threshold,
        "init_track_score_threshold": args.init_track_score_threshold,
        "delta_t": args.delta_t,
        "match_high_score_dets_with_confirmed_trks_threshold": (
            args.match_high_score_dets_with_confirmed_trks_threshold
        ),
        "match_low_score_dets_with_confirmed_trks_threshold": (
            args.match_low_score_dets_with_confirmed_trks_threshold
        ),
        "match_remained_high_score_dets_with_unconfirmed_trks_threshold": (
            args.match_remained_high_score_dets_with_unconfirmed_trks_threshold
        ),
        "association_iou_coefficient": args.association_iou_coefficient,
        "association_speed_direction_coefficient": (
            args.association_speed_direction_coefficient
        ),
        "use_byte": args.use_byte,
        "use_oru": args.use_oru,
        "use_confidence_r": args.use_confidence_r,
        "use_learned_q": args.use_learned_q,
        "q_scale": args.q_scale,
        "r_scale": args.r_scale,
        "reupdate_type": args.reupdate_type,
        "reupdate_constant_weight": args.reupdate_constant_weight,
        "motion": motion.model_dump(),
    }


def _candidate_r_matrix(
    tracker: OCSORTTracker,
    track,
    detection: np.ndarray,
    score: float,
) -> np.ndarray | None:
    """Return the exact R that production ``Track.update`` will use."""
    if track.kf is None:
        return None
    if tracker.config.use_confidence_r:
        r = np.diag([1.0, 1.0, 10.0, 10.0])
        return r * np.exp(2.0 * (1.0 - float(score)))
    context = tracker._adaptive_context(track)
    if context is None:
        # This matches the tracker warm-up path: no learned R until a full
        # completed-frame history exists, so Track.update falls back to kf.R.
        return track.kf.R.copy()
    measurement = tracker._measurement_feature(context, detection, score)
    var_r = tracker.motion_engine.predict_r_batch(
        context[3][None].astype(np.float32, copy=False),
        [tracker.motion_engine.history_len],
        measurement[None].astype(np.float32, copy=False),
    ).cpu().numpy()[0]
    xywh = BBOX.from_tlbr(detection)
    return tracker.config.r_scale * learned_measurement_noise_matrix(
        var_r, xywh, tracker.config.image_width, tracker.config.image_height
    )


def _record_pre_update_distance(
    tracker: OCSORTTracker,
    track,
    detection: np.ndarray,
    score: float,
    seq: str,
    frame: int,
    gt_id: int,
    crowd_iou: float,
) -> dict | None:
    # Warm-up samples use the fixed SORT covariance, not the learned adaptive
    # Q/R contract; exclude them from learned-covariance calibration statistics.
    if tracker._adaptive_context(track) is None or track.kf is None:
        return None
    r = _candidate_r_matrix(tracker, track, detection, score)
    if r is None:
        return None

    h_xy = track.kf.H[:2]
    innovation = tlbr_to_z(detection)[:2] - h_xy @ track.kf.x
    s_xy = h_xy @ track.kf.P @ h_xy.T + r[:2, :2]
    s_xy = 0.5 * (s_xy + s_xy.T)
    if not np.isfinite(s_xy).all() or not np.isfinite(innovation).all():
        return None
    try:
        solved = np.linalg.solve(s_xy + np.eye(2) * 1e-6, innovation)
    except np.linalg.LinAlgError:
        return None
    d2 = float((innovation.T @ solved).squeeze())
    if not math.isfinite(d2) or d2 < 0:
        return None
    return {
        "seq": seq,
        "frame": frame,
        "gt_id": gt_id,
        "d2_xy": d2,
        "gap": _bucket_gap(track.age),
        "score": _bucket_score(float(score)),
        "crowding": "crowded_iou_ge_0_3" if crowd_iou >= 0.3 else "separate_iou_lt_0_3",
        "crowding_iou": float(crowd_iou),
        "score_value": float(score),
        "p_xy_trace": float(np.trace(h_xy @ track.kf.P @ h_xy.T)),
        "r_xy_trace": float(np.trace(r[:2, :2])),
    }


def _phase_1_candidate_request(
    tracker: OCSORTTracker,
    track,
    detections: np.ndarray,
    correct_detection_index: int,
    context,
) -> dict | None:
    """Build one exact phase-1 candidate set for a confirmed oracle track.

    Phase 1 is the important case for direct identity theft: all confirmed
    tracks compete for high-confidence detections with the configured IoU
    threshold.  The result keeps original detection indices so summaries and
    examples can point back to a frame in the detection file.
    """
    correct_score = float(detections[correct_detection_index, 4])
    if correct_score <= tracker.config.high_score_det_threshold:
        return None
    high_indices = np.flatnonzero(
        detections[:, 4] > tracker.config.high_score_det_threshold
    )
    if len(high_indices) == 0:
        return None
    candidate_boxes = detections[high_indices, :4]
    ious = batch_iou(track.bbox.to_tlbr()[None], candidate_boxes)[0]
    viable = ious > tracker.config.match_high_score_dets_with_confirmed_trks_threshold
    candidate_indices = high_indices[viable]
    if correct_detection_index not in candidate_indices:
        return None
    correct_local_index = int(
        np.flatnonzero(candidate_indices == correct_detection_index)[0]
    )
    return {
        "track": track,
        "context": context,
        "candidate_detection_indices": candidate_indices,
        "candidate_boxes": detections[candidate_indices, :4],
        "candidate_scores": detections[candidate_indices, 4],
        "correct_local_index": correct_local_index,
    }


def _candidate_d2_xy_batch(tracker: OCSORTTracker, requests: list[dict]) -> None:
    """Attach learned-R 2-D d2 values to all candidate pairs in one batch.

    The target's P matrix is already causal because ``predict_tracks`` has run.
    R is candidate-specific for the learned measurement model, so it is
    predicted for every viable pair.  Batching pairs across a frame keeps this
    diagnostic practical on long DanceTrack sequences.
    """
    if not requests:
        return

    pair_request_indices: list[int] = []
    pair_local_indices: list[int] = []
    pair_boxes: list[np.ndarray] = []
    pair_scores: list[float] = []
    for request_index, request in enumerate(requests):
        for local_index, (box, score) in enumerate(
            zip(request["candidate_boxes"], request["candidate_scores"])
        ):
            pair_request_indices.append(request_index)
            pair_local_indices.append(local_index)
            pair_boxes.append(box)
            pair_scores.append(float(score))
    if not pair_boxes:
        return

    pair_boxes_array = np.asarray(pair_boxes, dtype=float)
    pair_scores_array = np.asarray(pair_scores, dtype=float)
    n_pairs = len(pair_boxes_array)
    r_matrices = np.empty((n_pairs, 4, 4), dtype=float)

    if tracker.config.use_confidence_r:
        base_r = np.diag([1.0, 1.0, 10.0, 10.0])
        scales = np.exp(2.0 * (1.0 - pair_scores_array))
        r_matrices[:] = scales[:, None, None] * base_r
    elif tracker.motion_engine is None:
        r_matrices[:] = np.stack(
            [requests[index]["track"].kf.R for index in pair_request_indices]
        )
    else:
        contexts = [requests[index]["context"] for index in pair_request_indices]
        features = np.stack([context[3] for context in contexts]).astype(
            np.float32, copy=False
        )
        measurements = np.stack([
            tracker._measurement_feature(context, box, score)
            for context, box, score in zip(contexts, pair_boxes_array, pair_scores_array)
        ]).astype(np.float32, copy=False)
        var_rs = tracker.motion_engine.predict_r_batch(
            features,
            [tracker.motion_engine.history_len] * n_pairs,
            measurements,
        ).cpu().numpy()
        for pair_index, (request_index, box, var_r) in enumerate(
            zip(pair_request_indices, pair_boxes_array, var_rs)
        ):
            xywh = BBOX.from_tlbr(box)
            r_matrices[pair_index] = tracker.config.r_scale * learned_measurement_noise_matrix(
                var_r,
                xywh,
                tracker.config.image_width,
                tracker.config.image_height,
            )

    h_xy = np.asarray([request["track"].kf.H[:2] for request in requests])
    predicted_means = np.asarray([
        request["track"].kf.H[:2] @ request["track"].kf.x for request in requests
    ]).reshape(len(requests), 2)
    p_xy = np.asarray([
        request["track"].kf.H[:2] @ request["track"].kf.P @ request["track"].kf.H[:2].T
        for request in requests
    ])
    request_indices = np.asarray(pair_request_indices, dtype=int)
    innovations = np.asarray([
        tlbr_to_z(box)[:2].reshape(2) for box in pair_boxes_array
    ]) - predicted_means[request_indices]
    innovation_covariances = p_xy[request_indices] + r_matrices[:, :2, :2]
    innovation_covariances = 0.5 * (
        innovation_covariances + np.swapaxes(innovation_covariances, -1, -2)
    )
    innovation_covariances += np.eye(2) * 1e-6
    try:
        solved = np.linalg.solve(innovation_covariances, innovations[..., None]).squeeze(-1)
        d2_values = np.einsum("ij,ij->i", innovations, solved)
        d2_values = np.where(
            np.isfinite(d2_values) & (d2_values >= 0), d2_values, np.inf
        )
    except np.linalg.LinAlgError:
        d2_values = np.full(n_pairs, np.inf, dtype=float)

    for request in requests:
        request["candidate_d2_xy"] = np.full(
            len(request["candidate_boxes"]), np.inf, dtype=float
        )
    for request_index, local_index, d2 in zip(
        pair_request_indices, pair_local_indices, d2_values
    ):
        requests[request_index]["candidate_d2_xy"][local_index] = d2


def _association_costs_for_candidates(
    tracker: OCSORTTracker,
    track,
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return IoU and the exact local IoU+direction association cost."""
    ious = batch_iou(track.bbox.to_tlbr()[None], candidate_boxes)[0]
    directions = batch_speed_direction(
        track.k_last_observation.reshape(1, 4), candidate_boxes
    )[0]
    direction_cost = np.abs(directions - track.speed_direction)
    direction_cost = np.where(
        direction_cost > np.pi, 2.0 * np.pi - direction_cost, direction_cost
    ) / np.pi
    direction_cost *= candidate_scores
    if np.array_equal(track.k_last_observation, np.array([0, 0, 1, 1])):
        direction_cost *= 0.0
    association_cost = (
        tracker.config.association_iou_coefficient * (1.0 - ious)
        + tracker.config.association_speed_direction_coefficient * direction_cost
    )
    return ious, association_cost


def _ranking_record(
    args,
    tracker: OCSORTTracker,
    request: dict,
    seq: str,
    frame: int,
    gt_id: int,
    crowd_iou: float,
) -> dict:
    """Build a compact ranking record for an oracle-correct phase-1 pair."""
    track = request["track"]
    candidate_boxes = request["candidate_boxes"]
    candidate_scores = request["candidate_scores"]
    correct_index = request["correct_local_index"]
    ious, tracker_cost = _association_costs_for_candidates(
        tracker, track, candidate_boxes, candidate_scores
    )
    d2_xy = request["candidate_d2_xy"] / args.ranking_covariance_scale
    capped_mahalanobis_cost = args.ranking_mahalanobis_coefficient * np.minimum(
        d2_xy / args.ranking_mahalanobis_reference,
        args.ranking_mahalanobis_cap,
    )
    costs = {
        "iou_only": 1.0 - ious,
        "tracker_iou_direction": tracker_cost,
        "mahalanobis_xy": d2_xy,
        "tracker_plus_capped_mahalanobis_xy": (
            tracker_cost + capped_mahalanobis_cost
        ),
    }
    method_records = {
        name: _rank_costs(cost, correct_index) for name, cost in costs.items()
    }
    top_candidates = {}
    for name, ranking in method_records.items():
        if ranking is None:
            top_candidates[name] = None
            continue
        top_index = ranking["top_candidate_local_index"]
        top_candidates[name] = {
            "detection_index": int(request["candidate_detection_indices"][top_index]) + 1,
            "iou": float(ious[top_index]),
            "d2_xy": float(d2_xy[top_index]),
            "score": float(candidate_scores[top_index]),
        }
    return {
        "seq": seq,
        "frame": frame,
        "gt_id": gt_id,
        "gap": _bucket_gap(track.age),
        "score": _bucket_score(float(candidate_scores[correct_index])),
        "crowding": "crowded_iou_ge_0_3" if crowd_iou >= 0.3 else "separate_iou_lt_0_3",
        "crowding_iou": float(crowd_iou),
        "candidate_count": int(len(candidate_boxes)),
        "correct_detection_index": int(request["candidate_detection_indices"][correct_index]) + 1,
        "correct": {
            "iou": float(ious[correct_index]),
            "d2_xy": float(d2_xy[correct_index]),
            "score": float(candidate_scores[correct_index]),
        },
        "methods": method_records,
        "top_candidates": top_candidates,
    }


def evaluate_sequence(
    args,
    seq: str,
    motion: MotionPredictorConfig,
    motion_engine: MotionPredictorEngine,
) -> tuple[list[dict], list[dict], dict]:
    sequence_dir = Path(args.dataset_dir) / args.split / seq
    detection_path = Path(args.detections_dir) / f"{seq}.txt"
    if not detection_path.is_file():
        raise FileNotFoundError(f"missing detections: {detection_path}")
    parser = configparser.ConfigParser()
    parser.read(sequence_dir / "seqinfo.ini")
    width = int(parser["Sequence"]["imWidth"])
    height = int(parser["Sequence"]["imHeight"])
    length = int(parser["Sequence"]["seqLength"])
    gt_by_frame = _load_gt_by_frame(sequence_dir)
    det_by_frame = _load_detections_by_frame(detection_path)
    tracker = OCSORTTracker(_tracker_config(args, width, height, motion), motion_engine)

    track_by_gt: dict[int, object] = {}
    records: list[dict] = []
    ranking_records: list[dict] = []
    diagnostics = {
        "oracle_matches": 0,
        "learned_covariance_samples": 0,
        "new_tracks": 0,
        "ranking_confirmed_high_score_targets": 0,
        "ranking_warmup_targets": 0,
        "ranking_correct_detection_not_iou_viable": 0,
        "ranking_records": 0,
    }

    for frame in range(1, length + 1):
        gt_items = gt_by_frame.get(frame, [])
        detections = det_by_frame.get(frame, np.empty((0, 5), dtype=float))
        gt_boxes = (
            np.stack([box for _, box in gt_items])
            if gt_items else np.empty((0, 4), dtype=float)
        )
        oracle_pairs = _one_to_one_matches(gt_boxes, detections[:, :4], args.match_iou)
        matched_by_gt = {
            gt_items[gt_index][0]: (
                int(det_index), detections[det_index, :4], float(detections[det_index, 4])
            )
            for gt_index, det_index in oracle_pairs
            if detections[det_index, 4] >= args.low_score_det_threshold
        }
        diagnostics["oracle_matches"] += len(matched_by_gt)
        crowding = _crowding_by_gt_id(gt_items)

        # This is the production causal prediction stage. Its batched Q inference
        # and Q->P conversion are shared with normal tracking.
        tracker.predict_tracks()
        active_tracks = tracker.get_tracks([StateTracking, StateLost, StateUnconfirmed])
        active_index = {id(track): index for index, track in enumerate(active_tracks)}

        update_detections: list[np.ndarray] = []
        update_scores: list[float] = []
        update_matches: list[list[int]] = []
        ranking_requests: list[tuple[dict, dict]] = []
        for gt_id, (detection_index, detection, score) in matched_by_gt.items():
            track = track_by_gt.get(gt_id)
            if track is None or track.state == StateDeleted or id(track) not in active_index:
                # A new/reappearing GT identity gets the same initialization rule
                # as the tracker. It cannot yield a pre-update calibration sample
                # until it has enough causal history.
                if score >= args.init_track_score_threshold:
                    tracker.init_track(detection, score)
                    track_by_gt[gt_id] = tracker.tracks[-1]
                diagnostics["new_tracks"] += 1
                continue

            # The ranking diagnostic intentionally mirrors only the confirmed
            # high-confidence phase. It is where direct association theft occurs;
            # Byte's recovery and unconfirmed-track phases are separate questions.
            if (
                track.state in (StateTracking, StateLost)
                and score > args.high_score_det_threshold
            ):
                diagnostics["ranking_confirmed_high_score_targets"] += 1
                context = tracker._adaptive_context(track)
                if context is None:
                    diagnostics["ranking_warmup_targets"] += 1
                else:
                    request = _phase_1_candidate_request(
                        tracker, track, detections, detection_index, context
                    )
                    if request is None:
                        diagnostics["ranking_correct_detection_not_iou_viable"] += 1
                    else:
                        ranking_requests.append((
                            {"gt_id": gt_id, "crowd_iou": crowding.get(gt_id, 0.0)},
                            request,
                        ))

            record = _record_pre_update_distance(
                tracker, track, detection, score, seq, frame, gt_id,
                crowding.get(gt_id, 0.0),
            )
            if record is not None:
                records.append(record)
                diagnostics["learned_covariance_samples"] += 1
            update_matches.append([active_index[id(track)], len(update_detections)])
            update_detections.append(detection)
            update_scores.append(score)

        # All ranking costs are measured before any oracle update at this frame.
        # Batched R inference makes their candidate-specific uncertainty feasible.
        if ranking_requests:
            requests = [request for _, request in ranking_requests]
            _candidate_d2_xy_batch(tracker, requests)
            for metadata, request in ranking_requests:
                ranking_records.append(_ranking_record(
                    args,
                    tracker,
                    request,
                    seq,
                    frame,
                    metadata["gt_id"],
                    metadata["crowd_iou"],
                ))
            diagnostics["ranking_records"] += len(ranking_requests)

        if update_matches:
            tracker._update_matches(
                active_tracks,
                np.asarray(update_detections, dtype=float),
                np.asarray(update_scores, dtype=float),
                update_matches,
            )

    diagnostics["frames"] = length
    diagnostics["records"] = len(records)
    return records, ranking_records, diagnostics


def _available_sequences(args) -> list[str]:
    if args.seqs:
        return list(dict.fromkeys(args.seqs))
    root = Path(args.dataset_dir) / args.split
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def main(args) -> None:
    if args.reupdate_type == "none":
        args.reupdate_type = None
    if args.ranking_covariance_scale <= 0:
        raise ValueError("--ranking_covariance_scale must be positive")
    if args.ranking_mahalanobis_reference <= 0:
        raise ValueError("--ranking_mahalanobis_reference must be positive")
    if args.ranking_mahalanobis_coefficient < 0:
        raise ValueError("--ranking_mahalanobis_coefficient must be non-negative")
    if args.ranking_mahalanobis_cap < 0:
        raise ValueError("--ranking_mahalanobis_cap must be non-negative")
    if args.ranking_example_limit < 0:
        raise ValueError("--ranking_example_limit must be non-negative")
    sequences = _available_sequences(args)
    if not sequences:
        raise ValueError("no sequences selected")

    motion = MotionPredictorConfig(
        enabled=True,
        model_type="adaptive_kalman",
        weights_path=args.weights_path,
        device=args.device,
        use_kalman=True,
        kalman_fusion_blend=args.kalman_fusion_blend,
        max_gap_norm=args.max_gap_norm,
    )
    motion_engine = MotionPredictorEngine(motion)
    all_records: list[dict] = []
    all_ranking_records: list[dict] = []
    sequence_diagnostics: dict[str, dict] = {}
    for seq in sequences:
        print(f"[{seq}] replaying oracle GT/detection matches...")
        records, ranking_records, diagnostics = evaluate_sequence(
            args, seq, motion, motion_engine
        )
        all_records.extend(records)
        all_ranking_records.extend(ranking_records)
        sequence_diagnostics[seq] = diagnostics
        print(
            f"[{seq}] learned-covariance samples: {len(records)} | "
            f"phase-1 ranking samples: {len(ranking_records)}"
        )

    raw_d2 = [record["d2_xy"] for record in all_records]
    report = {
        "description": (
            "Pre-update 2-D center innovation calibration. Ground truth selects "
            "oracle one-to-one detections only after causal tracker prediction."
        ),
        "checkpoint": str(Path(args.weights_path).resolve()),
        "sequences": sequences,
        "chi_square_2d_reference": CHI2_2D,
        "raw": {
            "overall": _summary(raw_d2),
            "by_sequence": _group_summaries(all_records, "seq"),
            "by_gap": _group_summaries(all_records, "gap"),
            "by_score": _group_summaries(all_records, "score"),
            "by_crowding": _group_summaries(all_records, "crowding"),
        },
        "sequence_diagnostics": sequence_diagnostics,
        "candidate_ranking": {
            "description": (
                "Local, causal phase-1 candidate ranking. For each confirmed "
                "oracle track with an oracle-correct high-confidence detection, "
                "the candidate list is exactly its high-confidence detections "
                "passing the production high-score IoU threshold. This diagnoses "
                "cost discrimination only; it does not simulate global Hungarian "
                "assignment conflicts."
            ),
            "settings": {
                "phase": "confirmed tracks vs high-confidence detections",
                "iou_threshold": args.match_high_score_dets_with_confirmed_trks_threshold,
                "association_iou_coefficient": args.association_iou_coefficient,
                "association_speed_direction_coefficient": (
                    args.association_speed_direction_coefficient
                ),
                "mahalanobis_dimensions": "center_xy (2-D)",
                "covariance_scale": args.ranking_covariance_scale,
                "capped_mahalanobis_coefficient": args.ranking_mahalanobis_coefficient,
                "capped_mahalanobis_reference": args.ranking_mahalanobis_reference,
                "capped_mahalanobis_cap": args.ranking_mahalanobis_cap,
            },
            "diagnostics": {
                "eligible_samples": len(all_ranking_records),
                "ambiguous_samples_candidate_count_ge_2": sum(
                    record["candidate_count"] >= 2 for record in all_ranking_records
                ),
            },
            "overall": _ranking_summary(all_ranking_records),
            "ambiguous_candidate_count_ge_2": _ranking_summary([
                record for record in all_ranking_records
                if record["candidate_count"] >= 2
            ]),
            "by_sequence": _ranking_group_summaries(all_ranking_records, "seq"),
            "by_gap": _ranking_group_summaries(all_ranking_records, "gap"),
            "by_score": _ranking_group_summaries(all_ranking_records, "score"),
            "by_crowding": _ranking_group_summaries(all_ranking_records, "crowding"),
            "examples": _ranking_examples(all_ranking_records, args.ranking_example_limit),
        },
    }

    calibration_seqs = set(args.calibration_seqs or [])
    if calibration_seqs:
        unknown = calibration_seqs - set(sequences)
        if unknown:
            raise ValueError(f"--calibration_seqs not selected by --seqs: {sorted(unknown)}")
        evaluation_seqs = set(args.evaluation_seqs or (set(sequences) - calibration_seqs))
        if not evaluation_seqs:
            raise ValueError("select disjoint --evaluation_seqs, or leave it empty to use remaining --seqs")
        overlap = calibration_seqs & evaluation_seqs
        if overlap:
            raise ValueError(f"calibration/evaluation sequences overlap: {sorted(overlap)}")
        cal_records = [record for record in all_records if record["seq"] in calibration_seqs]
        eval_records = [record for record in all_records if record["seq"] in evaluation_seqs]
        if not cal_records or not eval_records:
            raise ValueError("both calibration and evaluation sequence sets need learned-covariance samples")
        # For d2 ~ alpha*chi2(2), MLE of a global covariance scale is mean(d2)/2.
        scale = float(np.mean([record["d2_xy"] for record in cal_records]) / CHI2_2D["mean"])
        report["held_out_scale_validation"] = {
            "calibration_sequences": sorted(calibration_seqs),
            "evaluation_sequences": sorted(evaluation_seqs),
            "fitted_global_covariance_scale": scale,
            "calibration_raw": _summary([record["d2_xy"] for record in cal_records]),
            "evaluation_raw": _summary([record["d2_xy"] for record in eval_records]),
            "evaluation_after_global_scaling": {
                "overall": _summary([record["d2_xy"] for record in eval_records], scale=scale),
                "by_gap": _group_summaries(eval_records, "gap", scale=scale),
                "by_score": _group_summaries(eval_records, "score", scale=scale),
                "by_crowding": _group_summaries(eval_records, "crowding", scale=scale),
            },
        }
        calibration_ranking_records = [
            record for record in all_ranking_records
            if record["seq"] in calibration_seqs
        ]
        evaluation_ranking_records = [
            record for record in all_ranking_records
            if record["seq"] in evaluation_seqs
        ]
        report["candidate_ranking"]["held_out_validation"] = {
            "calibration_sequences": sorted(calibration_seqs),
            "evaluation_sequences": sorted(evaluation_seqs),
            # The covariance scale used by combined candidate ranking remains
            # explicit in --ranking_covariance_scale. This prevents a scale
            # fitted on calibration matches from being silently applied to the
            # calibration side of the comparison as well.
            "ranking_covariance_scale_used": args.ranking_covariance_scale,
            "calibration": _ranking_summary(calibration_ranking_records),
            "evaluation": {
                "overall": _ranking_summary(evaluation_ranking_records),
                "ambiguous_candidate_count_ge_2": _ranking_summary([
                    record for record in evaluation_ranking_records
                    if record["candidate_count"] >= 2
                ]),
                "by_gap": _ranking_group_summaries(
                    evaluation_ranking_records, "gap"
                ),
                "by_crowding": _ranking_group_summaries(
                    evaluation_ranking_records, "crowding"
                ),
            },
        }

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    overall = report["raw"]["overall"]
    print(
        "\nRaw 2-D calibration: "
        f"n={overall.get('count', 0)}, mean d2={overall.get('mean_d2', float('nan')):.3f} "
        f"(expected 2.000), coverage@95={overall.get('coverage_95', float('nan')):.3f} "
        "(expected 0.950)"
    )
    ranking = report["candidate_ranking"]["ambiguous_candidate_count_ge_2"]
    base = ranking.get("methods", {}).get("tracker_iou_direction", {})
    maha = ranking.get("methods", {}).get("mahalanobis_xy", {})
    print(
        "Ambiguous phase-1 candidate ranking: "
        f"n={ranking.get('count', 0)}, "
        f"tracker top-1={base.get('top_1_rate', float('nan')):.3f}, "
        f"Mahalanobis top-1={maha.get('top_1_rate', float('nan')):.3f}"
    )
    print(f"Report written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate held-out 2-D Kalman innovation calibration for association."
    )
    parser.add_argument("--weights_path", required=True, help="adaptive-Kalman best_model.pth")
    parser.add_argument("--dataset_dir", default="C:/Projects/.Datasets/DanceTrack")
    parser.add_argument("--detections_dir", default="C:/Projects/.Detections/YOLOXx/DanceTrack")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seqs", nargs="*", default=None, help="default: every sequence in --split")
    parser.add_argument("--calibration_seqs", nargs="*", default=None)
    parser.add_argument("--evaluation_seqs", nargs="*", default=None)
    parser.add_argument("--output_path", default="eval/association_calibration/report.json")
    parser.add_argument("--device", default=None, help="cuda | cpu | mps; default=auto")
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--update_window_start", type=int, default=30)
    parser.add_argument("--update_window_end", type=int, default=90)
    parser.add_argument("--high_score_det_threshold", type=float, default=0.6)
    parser.add_argument("--low_score_det_threshold", type=float, default=0.1)
    parser.add_argument("--init_track_score_threshold", type=float, default=0.6)
    parser.add_argument("--delta_t", type=int, default=3)
    parser.add_argument("--match_high_score_dets_with_confirmed_trks_threshold", type=float, default=0.2)
    parser.add_argument("--match_low_score_dets_with_confirmed_trks_threshold", type=float, default=0.5)
    parser.add_argument("--match_remained_high_score_dets_with_unconfirmed_trks_threshold", type=float, default=0.3)
    parser.add_argument("--association_iou_coefficient", type=float, default=1.0)
    parser.add_argument("--association_speed_direction_coefficient", type=float, default=0.3)
    parser.add_argument(
        "--ranking_covariance_scale", type=float, default=1.0,
        help="positive calibration scale applied to P+R before candidate d2 ranking",
    )
    parser.add_argument(
        "--ranking_mahalanobis_coefficient", type=float, default=0.02,
        help="maximum weak covariance contribution to the combined ranking cost",
    )
    parser.add_argument(
        "--ranking_mahalanobis_reference", type=float, default=CHI2_2D["q95"],
        help="2-D d2 reference used to normalize the capped covariance cost",
    )
    parser.add_argument(
        "--ranking_mahalanobis_cap", type=float, default=1.0,
        help="cap on normalized d2 before applying the weak covariance coefficient",
    )
    parser.add_argument(
        "--ranking_example_limit", type=int, default=20,
        help="maximum examples per candidate-ranking disagreement category",
    )
    parser.add_argument("--use_byte", action="store_true", default=True)
    parser.add_argument("--no_use_byte", action="store_false", dest="use_byte")
    parser.add_argument("--use_oru", action="store_true", default=False)
    parser.add_argument("--no_use_oru", action="store_false", dest="use_oru")
    parser.add_argument("--use_confidence_r", action="store_true", default=False)
    parser.add_argument("--use_learned_q", action="store_true", default=True)
    parser.add_argument("--no_use_learned_q", action="store_false", dest="use_learned_q")
    parser.add_argument("--q_scale", type=float, default=1.0)
    parser.add_argument("--r_scale", type=float, default=1.0)
    parser.add_argument("--kalman_fusion_blend", type=float, default=0.0)
    parser.add_argument("--max_gap_norm", type=float, default=None)
    parser.add_argument("--reupdate_type", default="none", choices=["constant", "relative", "none"])
    parser.add_argument("--reupdate_constant_weight", type=float, default=0.8)
    main(parser.parse_args())
