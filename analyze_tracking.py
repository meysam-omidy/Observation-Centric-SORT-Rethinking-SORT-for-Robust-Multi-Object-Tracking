"""
Diagnostic harness: run a tracking config, capture the full per-track history, then
cross-reference against GT to pinpoint WHY each ID switch / fragmentation happened.

Like ablation.py it pickles the tracker (all track histories) and writes the per-frame
association log. Beyond that, it produces a human-readable report that, for every ID
switch, shows the boxes / var_q / var_r / states of the tracks involved in the frames
around the switch and classifies the cause:
  - DRIFT   : the old track's predicted box drifted off the GT (IoU collapsed) before the swap
  - THEFT   : a different existing track grabbed the detection during a crossing
  - BIRTH   : a brand-new track id took over (old track died / went unmatched)
  - REVIVE  : an old lost track re-attached to the wrong identity

Usage (defaults = the current 'new implementation': adaptive_kalman real model):
  python analyze_tracking.py --seqs dancetrack0079
  python analyze_tracking.py --seqs dancetrack0079 --no_motion --use_confidence_r --use_oru   # heuristic
  python analyze_tracking.py --dataset MOT17 --seqs MOT17-04-FRCNN
"""
from __future__ import annotations
import argparse, configparser, json, os, pickle
from collections import Counter
import numpy as np
import lap

from ocsort import OCSORTTracker
from track_state import StateTracking, StateLost, StateDeleted, StateUnconfirmed
from utils import batch_iou


# ----------------------------- geometry -----------------------------
def _tlbr(b):  # accepts tlbr already; helper for arrays
    return np.asarray(b, dtype=float)


def iou_1v1(a, b):  # a,b tlbr
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
    inter = w * h
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def tlwh_to_tlbr(t):
    return np.array([t[0], t[1], t[0]+t[2], t[1]+t[3]], dtype=float)


# ----------------------------- data -----------------------------
def load_gt(seq_path, is_mot):
    """{frame: [(gid, tlbr), ...]} — MOT filtered to considered pedestrians."""
    import pandas as pd
    gt = pd.read_csv(os.path.join(seq_path, "gt", "gt.txt"), header=None).to_numpy()
    out = {}
    for row in gt:
        fr, gid, x, y, w, h = int(row[0]), int(row[1]), row[2], row[3], row[4], row[5]
        if is_mot and gt.shape[1] >= 8 and not (int(row[6]) == 1 and int(row[7]) == 1):
            continue  # ignore region / non-pedestrian
        out.setdefault(fr, []).append((gid, np.array([x, y, x+w, y+h], dtype=float)))
    return out


def build_config(args, iw, ih):
    return {
        "image_width": iw, "image_height": ih,
        "max_age": args.max_age, "update_window_start": args.update_window_start,
        "update_window_end": args.update_window_end, "min_box_area": args.min_box_area,
        "max_aspect_ratio": args.max_aspect_ratio,
        "delta_t": args.delta_t, "high_score_det_threshold": args.high_score_det_threshold,
        "low_score_det_threshold": args.low_score_det_threshold,
        "init_track_score_threshold": args.init_track_score_threshold,
        "match_high_score_dets_with_confirmed_trks_threshold": args.match_high_score_dets_with_confirmed_trks_threshold,
        "match_low_score_dets_with_confirmed_trks_threshold": args.match_low_score_dets_with_confirmed_trks_threshold,
        "match_remained_high_score_dets_with_unconfirmed_trks_threshold": args.match_remained_high_score_dets_with_unconfirmed_trks_threshold,
        "association_iou_coefficient": args.association_iou_coefficient,
        "association_speed_direction_coefficient": args.association_speed_direction_coefficient,
        "use_mahalanobis_association": args.use_mahalanobis_association,
        "use_mahalanobis_cost": args.use_mahalanobis_cost,
        "use_mahalanobis_gate": args.use_mahalanobis_gate,
        "mahalanobis_cost_coefficient": args.mahalanobis_cost_coefficient,
        "mahalanobis_gate_threshold": args.mahalanobis_gate_threshold,
        "use_learned_association": args.use_learned_association,
        "association_weights_path": args.association_weights_path,
        "association_device": args.association_device,
        "association_cost_weight": args.association_cost_weight,
        "association_residual_clip": args.association_residual_clip,
        "use_byte": args.use_byte, "use_oru": args.use_oru,
        "use_confidence_r": args.use_confidence_r,
        "use_learned_q": args.use_learned_q,
        "q_scale": args.q_scale, "r_scale": args.r_scale,
        "output_lost_tracks": args.output_lost_tracks,
        "lost_output_max_age": args.lost_output_max_age,
        "lost_output_score_decay": args.lost_output_score_decay,
        "lost_output_min_score": args.lost_output_min_score,
        "lost_output_require_inside_frame": args.lost_output_require_inside_frame,
        "suppress_duplicate_track_births": args.suppress_duplicate_track_births,
        "cleanup_duplicate_tracks": args.cleanup_duplicate_tracks,
        "duplicate_track_iou_threshold": args.duplicate_track_iou_threshold,
        "duplicate_track_min_observations": args.duplicate_track_min_observations,
        "duplicate_track_overlap_frames": args.duplicate_track_overlap_frames,
        "prioritize_mature_tracks": args.prioritize_mature_tracks,
        "mature_track_min_observations": args.mature_track_min_observations,
        "collect_association_diagnostics": (
            args.association_attribution or args.oracle_association
        ),
        "reupdate_type": args.reupdate_type,
        "reupdate_constant_weight": args.reupdate_constant_weight,
        "log_path": None,  # set per-seq
        "motion": {
            "enabled": args.motion_enabled, "model_type": args.model_type,
            "weights_path": args.weights_path, "device": args.device,
            "use_kalman": args.use_kalman,
            "kalman_fusion_blend": args.kalman_fusion_blend,
            "max_gap_norm": args.max_gap_norm,
        },
    }


# ----------------------------- run -----------------------------
def detection_file_path(args, seq):
    """Return the detector-specific MOT detection file for a sequence.

    Passing an empty ``--detector_name`` retains support for the former flat
    ``<detections_dir>/<dataset>/<sequence>.txt`` layout.
    """
    parts = [args.detections_dir]
    if args.detector_name:
        parts.append(args.detector_name)
    parts.extend([args.dataset, f"{seq}.txt"])
    return os.path.join(*parts)


def run_and_capture(args, seq, out_dir, gt_by_frame=None):
    seq_dir = f"{args.datasets_dir}/{args.dataset}/{args.split}/{seq}"
    cfg = configparser.ConfigParser(); cfg.read(f"{seq_dir}/seqinfo.ini")
    iw, ih = cfg["Sequence"]["imWidth"], cfg["Sequence"]["imHeight"]
    seqlen = int(cfg["Sequence"]["seqLength"])

    conf = build_config(args, iw, ih)
    conf["log_path"] = os.path.join(out_dir, f"{seq}.assoc.log")
    tracker = OCSORTTracker(conf)

    det_path = detection_file_path(args, seq)
    dets = np.loadtxt(det_path, delimiter=",")
    attribution = (
        AssociationAttribution(args.match_iou)
        if (args.association_attribution or args.oracle_association) else None
    )
    oracle_policy = OracleAssociationPolicy() if args.oracle_association else None

    outputs_by_frame = {}   # frame -> [(tid, tlbr, score)]
    mot_output_lines = []
    for fr in range(1, seqlen + 1):
        frame_dets = dets[dets[:, 0] == fr][:, 1:]
        if oracle_policy is not None:
            gt_to_det, _ = oracle_gt_detection_matches(
                (gt_by_frame or {}).get(fr, []), frame_dets, args.match_iou
            )
            oracle_policy.begin_frame(fr, attribution.track_to_gt, gt_to_det)
        tracker.update(frame_dets, association_override=oracle_policy)
        if attribution is not None:
            attribution.consume_frame(
                fr,
                (gt_by_frame or {}).get(fr, []),
                frame_dets,
                tracker.last_association_diagnostics,
                tracker,
            )
        rows = []
        for line in tracker.get_outputs():
            mot_output_lines.append(line)
            p = line.split(",")
            f_, tid = int(p[0]), int(p[1])
            x, y, w, h, sc = map(float, (p[2], p[3], p[4], p[5], p[6]))
            rows.append((tid, tlwh_to_tlbr([x, y, w, h]), sc))
        outputs_by_frame[fr] = rows

    tracker.motion_engine = None  # drop the torch model so the pickle is small/portable
    tracker.last_association_diagnostics = []
    with open(os.path.join(out_dir, f"{seq}.tracker.pkl"), "wb") as f:
        pickle.dump(tracker, f)
    return tracker, outputs_by_frame, seqlen, attribution, oracle_policy, mot_output_lines


# ----------------------------- analyze -----------------------------
def greedy_match(gts, outs, thr):
    """gts:[(gid,tlbr)] outs:[(tid,tlbr,score)] -> {gid: tid} by descending IoU."""
    pairs = []
    for gi, (gid, gb) in enumerate(gts):
        for oi, (tid, ob, _) in enumerate(outs):
            i = iou_1v1(gb, ob)
            if i >= thr:
                pairs.append((i, gi, oi, gid, tid))
    pairs.sort(reverse=True)
    used_g, used_o, m = set(), set(), {}
    for i, gi, oi, gid, tid in pairs:
        if gi in used_g or oi in used_o:
            continue
        used_g.add(gi); used_o.add(oi); m[gid] = tid
    return m


# ----------------------------- association attribution -----------------------------
def oracle_gt_detection_matches(gts, detections, threshold):
    """Return one-to-one GT/detection indices for a frame using Hungarian IoU."""
    if not gts or len(detections) == 0:
        return {}, {}
    gt_boxes = np.stack([box for _, box in gts]).astype(float, copy=False)
    det_boxes = np.asarray(detections, dtype=float).reshape(-1, 5)[:, :4]
    ious = batch_iou(gt_boxes, det_boxes)
    _, gt_to_det_indices, _ = lap.lapjv(1.0 - ious, extend_cost=True)
    gt_to_det, det_to_gt = {}, {}
    for gt_index, detection_index in enumerate(gt_to_det_indices):
        if detection_index < 0 or ious[gt_index, detection_index] < threshold:
            continue
        gt_id = int(gts[gt_index][0])
        detection_index = int(detection_index)
        gt_to_det[gt_id] = detection_index
        det_to_gt[detection_index] = gt_id
    return gt_to_det, det_to_gt


class AssociationAttribution:
    """Causally label real tracker association outcomes against oracle GT/dets.

    ``track_to_gt`` is updated only after each production association. Ground
    truth never changes tracking; it only lets us ask whether the track that
    represented GT identity g before the frame received g's oracle detection.
    """

    def __init__(self, match_iou):
        self.match_iou = float(match_iou)
        self.track_to_gt: dict[int, int] = {}
        self.counts = Counter()
        self.global_conflict_counts = Counter()
        self.failures: list[dict] = []
        self._by_frame_track: dict[tuple[int, int, int], dict] = {}
        self.frames = 0

    @staticmethod
    def _as_index(value):
        return None if value is None else int(value) + 1

    @staticmethod
    def _compact_number(value):
        return None if value is None or not np.isfinite(value) else round(float(value), 6)

    def _record(self, record):
        label = record["label"]
        self.counts[label] += 1
        if label == "GLOBAL_DISPLACEMENT" and record.get("conflict"):
            self.global_conflict_counts[record["conflict"]["type"]] += 1
        self._by_frame_track[
            (record["frame"], record["track_id"], record["source_gt_id"])
        ] = record
        if label != "CORRECT_ASSOCIATION":
            self.failures.append(record)

    def _pair_cost(self, event, row, detection_index):
        """Return the exact row/column score if that detection was offered."""
        if detection_index is None:
            return None
        cols = np.flatnonzero(event["detection_indices"] == detection_index)
        if not len(cols):
            return {
                "detection_index": self._as_index(detection_index),
                "offered": False,
                "valid": False,
                "cost": None,
            }
        col = int(cols[0])
        return {
            "detection_index": self._as_index(detection_index),
            "offered": True,
            "valid": bool(event["valid_pairs"][row, col]),
            "cost": self._compact_number(event["cost"][row, col]),
        }

    def _global_conflict(
        self,
        event,
        source_track_id,
        source_gt_id,
        expected_detection_a,
        assigned_detection,
        expected_owner_track_id,
        previous_track_to_gt,
        gt_to_det,
    ):
        """Describe the counterpart that won A's locally best detection."""
        if expected_owner_track_id is None:
            return {
                "type": "UNRESOLVED_OWNER",
                "reason": "No matched owner was found for A's expected detection.",
            }
        owner_rows = np.flatnonzero(
            np.asarray(event["track_ids"], dtype=int) == expected_owner_track_id
        )
        if not len(owner_rows):
            return {
                "type": "UNRESOLVED_OWNER",
                "reason": "Expected-detection owner is absent from the phase matrix.",
                "track_b_id": int(expected_owner_track_id),
            }
        owner_row = int(owner_rows[0])
        source_gt_b = previous_track_to_gt.get(expected_owner_track_id)
        expected_detection_b = (
            None if source_gt_b is None else gt_to_det.get(source_gt_b)
        )
        assigned_b = assigned_detection.get(expected_owner_track_id)
        assigned_detection_b = None if assigned_b is None else assigned_b[0]

        if source_gt_b is None:
            conflict_type = "WINNER_SOURCE_UNMAPPED"
            reason = "The winning track had no causal GT identity before this frame."
        elif source_gt_b == source_gt_id:
            conflict_type = "DUPLICATE_TRACK_CONFLICT"
            reason = "Both tracks represented the same GT identity before this frame."
        elif expected_detection_b is None:
            conflict_type = "WINNER_GT_DETECTION_MISSING"
            reason = "The winning track's own GT had no oracle detector match."
        elif (
            assigned_detection_b == expected_detection_a
            and assigned_detection.get(source_track_id, (None,))[0] == expected_detection_b
        ):
            conflict_type = "TWO_WAY_SWAP"
            reason = "The two tracks exchanged their oracle expected detections."
        elif assigned_detection_b == expected_detection_a:
            conflict_type = "ONE_WAY_THEFT"
            reason = "The winning track took A's expected detection without a direct swap."
        else:
            conflict_type = "COMPLEX_MULTITRACK_CONFLICT"
            reason = "The expected detection changed owners through a larger assignment conflict."

        source_rows = np.flatnonzero(
            np.asarray(event["track_ids"], dtype=int) == source_track_id
        )
        source_row = int(source_rows[0]) if len(source_rows) else None
        return {
            "type": conflict_type,
            "reason": reason,
            "track_a": {
                "track_id": int(source_track_id),
                "source_gt_id": int(source_gt_id),
                "expected_detection_index": self._as_index(expected_detection_a),
                "assigned_detection_index": self._as_index(
                    assigned_detection.get(source_track_id, (None,))[0]
                ),
            },
            "track_b": {
                "track_id": int(expected_owner_track_id),
                "source_gt_id": source_gt_b,
                "expected_detection_index": self._as_index(expected_detection_b),
                "assigned_detection_index": self._as_index(assigned_detection_b),
            },
            "costs": {
                "a_to_expected_a": self._pair_cost(
                    event, source_row, expected_detection_a
                ) if source_row is not None else None,
                "a_to_expected_b": self._pair_cost(
                    event, source_row, expected_detection_b
                ) if source_row is not None else None,
                "b_to_expected_a": self._pair_cost(
                    event, owner_row, expected_detection_a
                ),
                "b_to_expected_b": self._pair_cost(
                    event, owner_row, expected_detection_b
                ),
            },
        }

    def consume_frame(self, frame, gt_items, detections, events, tracker):
        """Consume current-frame tracker diagnostics after ``tracker.update``."""
        self.frames += 1
        detections = np.asarray(detections, dtype=float).reshape(-1, 5)
        gt_to_det, det_to_gt = oracle_gt_detection_matches(
            gt_items, detections, self.match_iou
        )
        previous_track_to_gt = dict(self.track_to_gt)

        # Actual per-track assignments after all phases. A confirmed track
        # appears in phase 1 and, only if unmatched there, may appear in phase
        # 2. The last stored assignment is therefore its final one.
        assigned_detection: dict[int, tuple[int, int]] = {}
        events_by_track: dict[int, list[tuple[dict, int]]] = {}
        for event in events:
            track_ids = event["track_ids"]
            for row, track_id in enumerate(track_ids):
                events_by_track.setdefault(track_id, []).append((event, row))
            for row, col in event["matches"]:
                track_id = track_ids[row]
                assigned_detection[track_id] = (
                    int(event["detection_indices"][col]), int(event["phase"])
                )

        for track_id, source_gt_id in previous_track_to_gt.items():
            expected_detection = gt_to_det.get(source_gt_id)
            actual = assigned_detection.get(track_id)
            actual_detection = None if actual is None else actual[0]
            actual_phase = None if actual is None else actual[1]
            source_events = events_by_track.get(track_id, [])

            # A deleted track retains its causal identity through a GT/detector
            # miss gap. Once its GT has a usable detector match again, report one
            # lifecycle failure rather than losing the eventual BIRTH/REVIVE
            # switch context altogether.
            if not source_events and expected_detection is None:
                continue
            record = {
                "frame": int(frame),
                "track_id": int(track_id),
                "source_gt_id": int(source_gt_id),
                "expected_detection_index": self._as_index(expected_detection),
                "assigned_detection_index": self._as_index(actual_detection),
                "assigned_gt_id": (
                    None if actual_detection is None else det_to_gt.get(actual_detection)
                ),
                "assigned_phase": actual_phase,
                "label": None,
                "reason": None,
            }

            if expected_detection is None:
                record.update(
                    label="GT_DETECTION_MISSING",
                    reason="No one-to-one detector match for the source GT at this frame.",
                )
                self._record(record)
                continue

            if not source_events:
                record.update(
                    label="TRACK_NOT_ACTIVE",
                    reason="Source track was no longer active when its GT detection returned.",
                )
                self._record(record)
                # Do not report the same deleted track on every later frame.
                self.track_to_gt.pop(track_id, None)
                continue

            expected_event = None
            expected_row = expected_col = None
            offered_phases = []
            for event, row in source_events:
                offered_phases.append(int(event["phase"]))
                cols = np.flatnonzero(event["detection_indices"] == expected_detection)
                if len(cols):
                    expected_event, expected_row, expected_col = event, row, int(cols[0])
                    break

            if actual_detection == expected_detection:
                record.update(
                    label="CORRECT_ASSOCIATION",
                    reason="The source track received its oracle GT detection.",
                )
                self._record(record)
                continue

            if expected_event is None:
                if offered_phases:
                    reason = (
                        "Expected detection was not offered in phases "
                        f"{sorted(set(offered_phases))}; score/phase priority or an "
                        "earlier match prevented this candidate."
                    )
                    label = "CANDIDATE_REJECTED"
                else:
                    reason = "Source track was not active in any association phase."
                    label = "TRACK_NOT_ACTIVE"
                record.update(label=label, reason=reason, offered_phases=sorted(set(offered_phases)))
                self._record(record)
                continue

            valid_pairs = expected_event["valid_pairs"]
            costs = expected_event["cost"]
            ious = expected_event["iou"]
            direction_cost = expected_event["direction_cost"]
            record.update(
                phase=int(expected_event["phase"]),
                expected_iou=self._compact_number(ious[expected_row, expected_col]),
                expected_direction_cost=self._compact_number(
                    direction_cost[expected_row, expected_col]
                ),
                expected_cost=self._compact_number(costs[expected_row, expected_col]),
            )
            if not valid_pairs[expected_row, expected_col]:
                record.update(
                    label="CANDIDATE_REJECTED",
                    reason="Expected detection failed the phase's valid-pair gate.",
                )
                self._record(record)
                continue

            row_valid = valid_pairs[expected_row]
            row_costs = costs[expected_row]
            best_col = int(np.argmin(np.where(row_valid, row_costs, np.inf)))
            expected_cost = float(row_costs[expected_col])
            local_rank = int(np.sum(row_valid & (row_costs < expected_cost - 1e-12))) + 1
            expected_owner = None
            for owner_row, owner_col in expected_event["matches"]:
                if owner_col == expected_col:
                    expected_owner = int(expected_event["track_ids"][owner_row])
                    break
            record.update(
                local_rank=local_rank,
                row_best_detection_index=self._as_index(
                    int(expected_event["detection_indices"][best_col])
                ),
                row_best_cost=self._compact_number(row_costs[best_col]),
                expected_detection_owner_track_id=expected_owner,
            )
            if local_rank > 1:
                record.update(
                    label="LOCAL_RANKING_FAILURE",
                    reason="A different valid detection had lower local association cost.",
                )
            else:
                record.update(
                    label="GLOBAL_DISPLACEMENT",
                    reason=(
                        "Expected detection was locally best but was assigned to another "
                        "track or displaced by global assignment."
                    ),
                )
                record["conflict"] = self._global_conflict(
                    expected_event,
                    track_id,
                    source_gt_id,
                    expected_detection,
                    assigned_detection,
                    expected_owner,
                    previous_track_to_gt,
                    gt_to_det,
                )
            self._record(record)

        # Association updates are the only causal source of a track's identity
        # mapping. Unmatched tracks retain their last identity through gaps.
        for track_id, (detection_index, _) in assigned_detection.items():
            gt_id = det_to_gt.get(detection_index)
            if gt_id is not None:
                self.track_to_gt[track_id] = gt_id

        # First-frame / post-phase-3 births have no existing source identity.
        # Attach them after the tracker has initialized them so their next frame
        # can be attributed causally.
        used_detection_indices = set(assigned_detection.values())
        used_detection_indices = {item[0] for item in used_detection_indices}
        for track in tracker.tracks:
            if track.id in self.track_to_gt:
                continue
            update = track.history.update.get(frame)
            if update is None:
                continue
            box = update.bbox.to_tlbr()
            candidates = np.flatnonzero(np.all(np.isclose(detections[:, :4], box), axis=1))
            for detection_index in candidates:
                if detection_index in used_detection_indices:
                    continue
                gt_id = det_to_gt.get(int(detection_index))
                if gt_id is not None:
                    self.track_to_gt[track.id] = gt_id
                    used_detection_indices.add(int(detection_index))
                    break

        # Align the next frame's source identity with the same GT/output IoU
        # convention used by this script's ID-switch analysis. This is still
        # strictly post-association diagnostic bookkeeping: it never feeds back
        # into the tracker or the cost matrices captured above.
        output_rows = []
        for line in tracker.get_outputs():
            values = line.split(",")
            track_id = int(values[1])
            x, y, w, h, score = map(float, values[2:7])
            output_rows.append((track_id, tlwh_to_tlbr([x, y, w, h]), score))
        for gt_id, track_id in greedy_match(gt_items, output_rows, self.match_iou).items():
            self.track_to_gt[track_id] = gt_id


    def switch_record(self, frame, old_track_id, gt_id):
        return self._by_frame_track.get((int(frame), int(old_track_id), int(gt_id)))

    def summary(self):
        return {
            "frames": self.frames,
            "classified_track_frames": int(sum(self.counts.values())),
            "label_counts": dict(sorted(self.counts.items())),
            "global_conflict_type_counts": dict(sorted(self.global_conflict_counts.items())),
            "failure_count": len(self.failures),
        }

    def write(self, seq, out_dir):
        path = os.path.join(out_dir, f"{seq}.association_attribution.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"seq": seq, "summary": self.summary(), "failures": self.failures},
                handle,
                indent=2,
            )
        return path


class OracleAssociationPolicy:
    """GT-only causal association upper-bound policy for offline evaluation.

    The policy receives the production track/detection candidates after normal
    gates have been applied. It can force only a valid pair whose source track
    was mapped to the same GT identity on the preceding frame. GT never
    contributes a box, a prediction, a new track, or an invalid association.
    """

    def __init__(self):
        self.frame = None
        self.previous_track_to_gt = {}
        self.gt_to_detection = {}
        self.counts = Counter()
        self.forced_by_phase = Counter()
        self._seen_missing_detection = set()
        self._seen_not_offered = set()
        self._seen_invalid = set()

    def begin_frame(self, frame, previous_track_to_gt, gt_to_detection):
        self.frame = int(frame)
        self.previous_track_to_gt = dict(previous_track_to_gt)
        self.gt_to_detection = dict(gt_to_detection)
        self.counts["frames"] += 1
        self._seen_missing_detection.clear()
        self._seen_not_offered.clear()
        self._seen_invalid.clear()

    def __call__(self, *, phase, tracks, detection_indices, valid_pairs, cost):
        """Return disjoint valid local row/column pairs for this phase."""
        self.counts["association_calls"] += 1
        detection_indices = np.asarray(detection_indices, dtype=int)
        valid_pairs = np.asarray(valid_pairs, dtype=bool)
        cost = np.asarray(cost, dtype=float)
        candidates_by_detection = {}

        for row, track in enumerate(tracks):
            track_id = int(track.id)
            source_gt_id = self.previous_track_to_gt.get(track_id)
            if source_gt_id is None:
                continue
            expected_detection = self.gt_to_detection.get(source_gt_id)
            source_key = (track_id, int(source_gt_id))
            if expected_detection is None:
                if source_key not in self._seen_missing_detection:
                    self.counts["source_gt_has_no_detector_match"] += 1
                    self._seen_missing_detection.add(source_key)
                continue

            cols = np.flatnonzero(detection_indices == int(expected_detection))
            if len(cols) == 0:
                # The expected detector box may be in a different score phase;
                # count once per track/frame rather than once per callback.
                if source_key not in self._seen_not_offered:
                    self.counts["expected_detection_not_offered"] += 1
                    self._seen_not_offered.add(source_key)
                continue
            col = int(cols[0])
            if not valid_pairs[row, col]:
                if source_key not in self._seen_invalid:
                    self.counts["expected_pair_failed_tracker_gate"] += 1
                    self._seen_invalid.add(source_key)
                continue
            self.counts["eligible_valid_pairs"] += 1
            candidates_by_detection.setdefault(int(expected_detection), []).append(
                (float(cost[row, col]), track_id, row, col)
            )

        forced_matches = []
        for expected_detection, candidates in candidates_by_detection.items():
            # A prior mapping can contain aliases after an earlier identity
            # error. Use the lowest production cost, then older ID, rather than
            # forcing two tracks onto one detector box.
            candidates.sort(key=lambda item: (item[0], item[1]))
            _, _, row, col = candidates[0]
            forced_matches.append((row, col))
            self.counts["forced_matches"] += 1
            self.forced_by_phase[int(phase)] += 1
            if len(candidates) > 1:
                self.counts["same_gt_track_conflicts"] += len(candidates) - 1
        return forced_matches

    def summary(self):
        return {
            "counts": dict(sorted(self.counts.items())),
            "forced_by_phase": {
                str(phase): int(count)
                for phase, count in sorted(self.forced_by_phase.items())
            },
        }

    def write(self, seq, out_dir):
        path = os.path.join(out_dir, f"{seq}.oracle_association.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"seq": seq, **self.summary()}, handle, indent=2)
        return path


def track_snapshot(track, fr, gt_tlbr):
    """Per-frame view of a track for the report."""
    pred = track.history.predict.get(fr)
    upd = track.history.update.get(fr)
    st = track.history.state.get(fr, "-")
    snap = {"state": st, "matched_det": upd is not None}
    if pred is not None:
        pb = pred.bbox.to_tlbr()
        snap["pred_tlbr"] = [round(float(v), 1) for v in pb]
        snap["iou_pred_gt"] = round(iou_1v1(pb, gt_tlbr), 3) if gt_tlbr is not None else None
        snap["var_q"] = None if pred.var_q is None else [round(float(v), 6) for v in np.ravel(pred.var_q)[:4]]
        snap["var_r"] = None if pred.var_r is None else [round(float(v), 6) for v in np.ravel(pred.var_r)[:4]]
    if upd is not None:
        snap["det_score"] = round(float(upd.score), 3)
    return snap


def analyze(
    seq, tracker, outputs_by_frame, gt_by_frame, seqlen, thr, out_dir,
    ctx=2, attribution=None,
):
    tracks_by_id = {t.id: t for t in tracker.tracks}

    # per-frame GT<->tracker assignment
    gt_to_tid = {}  # frame -> {gid: tid}
    for fr in range(1, seqlen + 1):
        gts = gt_by_frame.get(fr, [])
        outs = outputs_by_frame.get(fr, [])
        gt_to_tid[fr] = greedy_match(gts, outs, thr) if gts and outs else {}

    # detect ID switches per GT id
    switches, frag = [], 0
    gt_ids = sorted({gid for f in gt_by_frame.values() for gid, _ in f})
    for gid in gt_ids:
        last_tid, last_fr = None, None
        for fr in range(1, seqlen + 1):
            tid = gt_to_tid[fr].get(gid)
            if tid is None:
                continue
            if last_tid is not None and tid != last_tid:
                switches.append({"frame": fr, "gt_id": gid, "old_tid": last_tid,
                                 "new_tid": tid, "gap": fr - last_fr - 1})
                if fr - last_fr - 1 > 0:
                    frag += 1
            last_tid, last_fr = tid, fr

    # build a context dump + cause classification for each switch
    def gt_box_at(gid, fr):
        for g, b in gt_by_frame.get(fr, []):
            if g == gid:
                return b
        return None

    cases = []
    for sw in switches:
        fr, gid, ot, nt = sw["frame"], sw["gt_id"], sw["old_tid"], sw["new_tid"]
        old_t, new_t = tracks_by_id.get(ot), tracks_by_id.get(nt)
        timeline = []
        for f in range(max(1, fr - ctx), min(seqlen, fr + ctx) + 1):
            gb = gt_box_at(gid, f)
            timeline.append({
                "frame": f, "is_switch": f == fr,
                "gt_tlbr": None if gb is None else [round(float(v), 1) for v in gb],
                "old": track_snapshot(old_t, f, gb) if old_t else None,
                "new": track_snapshot(new_t, f, gb) if new_t else None,
            })
        # classify cause
        cause = "UNKNOWN"
        old_pre = [t for t in timeline if t["frame"] < fr and t["old"]]
        old_iou_before = old_pre[-1]["old"].get("iou_pred_gt") if old_pre else None
        new_born = new_t is not None and new_t.entered_frame >= fr - 1
        old_state_at = old_t.history.state.get(fr) if old_t else None
        if new_born:
            cause = "BIRTH (old track lost/unmatched -> new id created)"
        elif old_iou_before is not None and old_iou_before < 0.3:
            cause = f"DRIFT (old pred IoU with GT fell to {old_iou_before} before swap)"
        elif old_state_at in ("Lost", "Deleted"):
            cause = "REVIVE (old track went Lost; GT re-attached to another id)"
        else:
            cause = "THEFT (another live track grabbed the detection during crossing)"
        case = {**sw, "cause": cause, "timeline": timeline}
        if attribution is not None:
            case["association_attribution"] = attribution.switch_record(fr, ot, gid)
        cases.append(case)

    summary = {
        "seq": seq, "frames": seqlen,
        "gt_ids": len(gt_ids),
        "tracker_ids": len({t.id for t in tracker.tracks}),
        "id_switches": len(switches),
        "fragmentations": frag,
        "cause_counts": {},
    }
    for c in cases:
        key = c["cause"].split(" ")[0]
        summary["cause_counts"][key] = summary["cause_counts"].get(key, 0) + 1
    if attribution is not None:
        switch_records = [
            case["association_attribution"] for case in cases
            if case.get("association_attribution") is not None
        ]
        summary["association_attribution"] = {
            **attribution.summary(),
            "switches_with_attribution": len(switch_records),
            "switch_label_counts": dict(sorted(Counter(
                record["label"] for record in switch_records
            ).items())),
            "switch_global_conflict_type_counts": dict(sorted(Counter(
                record["conflict"]["type"]
                for record in switch_records
                if record["label"] == "GLOBAL_DISPLACEMENT" and record.get("conflict")
            ).items())),
        }

    with open(os.path.join(out_dir, f"{seq}.analysis.json"), "w") as f:
        json.dump({"summary": summary, "switches": cases}, f, indent=2)
    _write_report(seq, summary, cases, out_dir)
    return summary


def _write_report(seq, summary, cases, out_dir, max_cases=25):
    L = []
    L.append("=" * 78)
    L.append(f"  TRACKING ANALYSIS — {seq}")
    L.append("=" * 78)
    L.append(f"  frames={summary['frames']}  GT ids={summary['gt_ids']}  "
             f"tracker ids={summary['tracker_ids']}")
    L.append(f"  ID switches={summary['id_switches']}  fragmentations={summary['fragmentations']}")
    L.append(f"  cause breakdown: {summary['cause_counts']}")
    L.append("")
    L.append(f"  Showing up to {max_cases} switches (full detail in {seq}.analysis.json)")
    L.append("-" * 78)
    for c in cases[:max_cases]:
        L.append(f"\n  SWITCH @ frame {c['frame']}  GT id {c['gt_id']}:  "
                 f"track {c['old_tid']} -> {c['new_tid']}  (gap {c['gap']})   [{c['cause']}]")
        L.append(f"    {'frame':>6} {'':1} {'GT tlbr':>26} | old: state/IoU(pred,GT)/detScore | new: state/IoU/detScore")
        for t in c["timeline"]:
            mark = "*" if t["is_switch"] else " "
            gt = t["gt_tlbr"]
            gt_s = "-" if gt is None else str(gt)
            def fmt(x):
                if x is None:
                    return "        -        "
                return f"{x['state'][:5]:>5}/{str(x.get('iou_pred_gt')):>5}/{str(x.get('det_score','-')):>5}"
            L.append(f"    {t['frame']:>6} {mark} {gt_s:>26} | {fmt(t['old'])} | {fmt(t['new'])}")
    txt = "\n".join(L)
    with open(os.path.join(out_dir, f"{seq}.report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


def main(args):
    is_mot = "MOT" in args.dataset
    out_dir = os.path.join("analysis", args.name)
    os.makedirs(out_dir, exist_ok=True)
    seqs = args.seqs or os.listdir(f"{args.datasets_dir}/{args.dataset}/{args.split}/")
    all_sum = []
    oracle_summaries = []
    for seq in seqs:
        print(f"\n### running {seq} ...")
        gt_by_frame = load_gt(f"{args.datasets_dir}/{args.dataset}/{args.split}/{seq}", is_mot)
        tracker, outputs_by_frame, seqlen, attribution, oracle_policy, mot_output_lines = run_and_capture(
            args, seq, out_dir, gt_by_frame
        )
        s = analyze(
            seq, tracker, outputs_by_frame, gt_by_frame, seqlen, args.match_iou,
            out_dir, attribution=attribution,
        )
        if attribution is not None:
            path = attribution.write(seq, out_dir)
            print(f"  association attribution: {path}")
        if oracle_policy is not None:
            output_dir = os.path.join("outputs", args.oracle_tracker_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{seq}.txt")
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(mot_output_lines))
                if mot_output_lines:
                    handle.write("\n")
            path = oracle_policy.write(seq, out_dir)
            oracle_summaries.append({"seq": seq, **oracle_policy.summary()})
            print(f"  oracle MOT output: {output_path}")
            print(f"  oracle headroom report: {path}")
        all_sum.append(s)
    print("\n=== ALL SEQUENCES ===")
    for s in all_sum:
        print(f"  {s['seq']}: IDSW={s['id_switches']} frag={s['fragmentations']} "
              f"causes={s['cause_counts']}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(all_sum, f, indent=2)
    if oracle_summaries:
        with open(os.path.join(out_dir, "oracle_association_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(oracle_summaries, handle, indent=2)
    if args.evaluate_oracle:
        if not args.oracle_association:
            raise ValueError("--evaluate_oracle requires --oracle_association")
        seqmap_dir = os.path.join("trackeval", "seqmap", args.dataset.lower())
        os.makedirs(seqmap_dir, exist_ok=True)
        with open(os.path.join(seqmap_dir, "custom.txt"), "w", encoding="utf-8") as handle:
            handle.write("name\n")
            for seq in seqs:
                handle.write(f"{seq}\n")
        from evaluate import evaluate
        print("\nevaluating oracle-association upper bound...")
        evaluate(
            args.dataset,
            args.split,
            trackers_to_eval=[args.oracle_tracker_name],
            datasets_dir=args.datasets_dir,
        )
    print(f"\nOutputs in {out_dir}/ : *.report.txt (readable), *.analysis.json (full), "
          f"*.tracker.pkl (histories), *.assoc.log (per-frame association)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze a tracking run against GT to locate ID-switch causes")
    p.add_argument("--dataset", default="DanceTrack", choices=["MOT17", "MOT20", "DanceTrack", "SportsMOT"])
    p.add_argument("--split", default="val")
    p.add_argument("--seqs", nargs="*", default=["dancetrack0065"])
    p.add_argument("--name", default="new_impl", help="output subfolder under analysis/")
    p.add_argument("--datasets_dir", default="C:/Projects/.Datasets")
    p.add_argument(
        "--detections_dir",
        default="C:/Projects/.Detections",
        help="Root directory containing detector-specific detection folders.",
    )
    p.add_argument(
        "--detector_name",
        default="YOLOXx",
        help="Detector subfolder under --detections_dir (e.g. YOLO11x, YOLO26x, YOLOXx).",
    )
    p.add_argument("--match_iou", type=float, default=0.5)
    p.add_argument(
        "--association_attribution", action="store_true", default=False,
        help=(
            "capture production association matrices and label each stable-track "
            "failure as local ranking, global displacement, candidate rejection, "
            "or missing GT detection"
        ),
    )
    p.add_argument(
        "--oracle_association", action="store_true", default=False,
        help=(
            "GT-only causal upper bound: force only prior-track/current-detection "
            "pairs that production association already offers and considers valid"
        ),
    )
    p.add_argument(
        "--oracle_tracker_name", default="ocsort-oracle-association",
        help="output subfolder under outputs/ when --oracle_association is enabled",
    )
    p.add_argument(
        "--evaluate_oracle", action="store_true", default=False,
        help="run TrackEval for the selected oracle-association sequences",
    )
    # Tracker options: kept in parity with run_tracker.py so an analysis run
    # reproduces the same tracker behaviour.
    p.add_argument("--max_age", type=int, default=30)
    p.add_argument("--update_window_start", type=int, default=30)
    p.add_argument("--update_window_end", type=int, default=90)
    p.add_argument("--min_box_area", type=int, default=100)
    p.add_argument("--max_aspect_ratio", type=float, default=1.6)
    p.add_argument("--delta_t", type=int, default=3)
    p.add_argument("--high_score_det_threshold", type=float, default=0.6)
    p.add_argument("--low_score_det_threshold", type=float, default=0.1)
    p.add_argument("--init_track_score_threshold", type=float, default=0.6)
    p.add_argument("--match_high_score_dets_with_confirmed_trks_threshold", type=float, default=0.2)
    p.add_argument("--match_low_score_dets_with_confirmed_trks_threshold", type=float, default=0.5)
    p.add_argument("--match_remained_high_score_dets_with_unconfirmed_trks_threshold", type=float, default=0.3)
    p.add_argument("--association_iou_coefficient", type=float, default=1.0)
    p.add_argument("--association_speed_direction_coefficient", type=float, default=0.3)
    p.add_argument("--use_mahalanobis_association", action="store_true", default=False,
                   help="legacy alias: enable both Mahalanobis soft cost and hard gate")
    p.add_argument("--use_mahalanobis_cost", action="store_true", default=False,
                   help="add a soft Mahalanobis cost without rejecting candidates")
    p.add_argument("--use_mahalanobis_gate", action="store_true", default=False,
                   help="reject covariance-improbable candidates without adding Mahalanobis cost")
    p.add_argument("--mahalanobis_cost_coefficient", type=float, default=1.0)
    p.add_argument("--mahalanobis_gate_threshold", type=float, default=9.4877)
    p.add_argument("--use_learned_association", action="store_true", default=False,
                   help="add a trained no-ReID association residual to valid candidate costs")
    p.add_argument("--association_weights_path", default=None,
                   help="checkpoint produced by train_association_model.py")
    p.add_argument("--association_device", default=None)
    p.add_argument("--association_cost_weight", type=float, default=0.10)
    p.add_argument("--association_residual_clip", type=float, default=0.50)
    p.add_argument("--use_byte", action="store_true", default=True)
    p.add_argument("--no_use_byte", action="store_false", dest="use_byte")
    p.add_argument("--use_oru", action="store_true", default=True)
    p.add_argument("--no_use_oru", action="store_false", dest="use_oru")
    p.add_argument("--use_confidence_r", action="store_true", default=False)
    p.add_argument("--use_learned_q", action="store_true", default=True)
    p.add_argument("--no_use_learned_q", action="store_false", dest="use_learned_q")
    p.add_argument("--q_scale", type=float, default=1.0)
    p.add_argument("--r_scale", type=float, default=1.0)
    p.add_argument("--output_lost_tracks", action="store_true", default=False)
    p.add_argument("--lost_output_max_age", type=int, default=3)
    p.add_argument("--lost_output_score_decay", type=float, default=0.7)
    p.add_argument("--lost_output_min_score", type=float, default=0.3)
    p.add_argument("--lost_output_require_inside_frame", action="store_true", default=True)
    p.add_argument("--lost_output_allow_partial_outside", action="store_false",
                   dest="lost_output_require_inside_frame")
    p.add_argument("--suppress_duplicate_track_births", action="store_true", default=False,
                   help="do not initialize a high-score detection that overlaps a mature live/lost track")
    p.add_argument("--cleanup_duplicate_tracks", action="store_true", default=False,
                   help="retire a weaker track after persistent high-IoU overlap with a mature observed track")
    p.add_argument("--duplicate_track_iou_threshold", type=float, default=0.85)
    p.add_argument("--duplicate_track_min_observations", type=int, default=3)
    p.add_argument("--duplicate_track_overlap_frames", type=int, default=3)
    p.add_argument("--prioritize_mature_tracks", action="store_true", default=False,
                   help="associate mature currently tracking identities before younger or lost tracks in phase 1")
    p.add_argument("--mature_track_min_observations", type=int, default=3)
    p.add_argument("--reupdate_type", default="constant", choices=["constant", "relative", "none"])
    p.add_argument("--reupdate_constant_weight", type=float, default=0.8)
    p.add_argument("--motion_enabled", action="store_true", default=True)
    p.add_argument("--no_motion", action="store_false", dest="motion_enabled")
    p.add_argument("--model_type", default="adaptive_kalman",
                   choices=["transformer", "transformer_learned", "lstm", "lstm_learned", "adaptive_kalman"])
    p.add_argument("--weights_path", default="../motion-predictor/checkpoints/adaptive_kalman_real_low_data/best_model.pth")
    p.add_argument("--device", default=None, help="cuda | cpu | mps; default = auto")
    p.add_argument("--use_kalman", action="store_true", default=True)
    p.add_argument("--no_use_kalman", action="store_false", dest="use_kalman")
    p.add_argument("--kalman_fusion_blend", type=float, default=0.0)
    p.add_argument("--max_gap_norm", type=float, default=None,
                   help="adaptive_kalman only; None = read from checkpoint (falls back to 30.0)")
    args = p.parse_args()
    if args.reupdate_type == "none":
        args.reupdate_type = None
    main(args)
