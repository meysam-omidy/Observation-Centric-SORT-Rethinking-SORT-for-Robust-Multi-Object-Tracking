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
import numpy as np

from ocsort import OCSORTTracker
from track_state import StateTracking, StateLost, StateDeleted, StateUnconfirmed


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
        "use_byte": args.use_byte, "use_oru": args.use_oru,
        "use_confidence_r": args.use_confidence_r,
        "use_learned_q": args.use_learned_q,
        "q_scale": args.q_scale, "r_scale": args.r_scale,
        "output_lost_tracks": args.output_lost_tracks,
        "lost_output_max_age": args.lost_output_max_age,
        "lost_output_score_decay": args.lost_output_score_decay,
        "lost_output_min_score": args.lost_output_min_score,
        "lost_output_require_inside_frame": args.lost_output_require_inside_frame,
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


def run_and_capture(args, seq, out_dir):
    seq_dir = f"{args.datasets_dir}/{args.dataset}/{args.split}/{seq}"
    cfg = configparser.ConfigParser(); cfg.read(f"{seq_dir}/seqinfo.ini")
    iw, ih = cfg["Sequence"]["imWidth"], cfg["Sequence"]["imHeight"]
    seqlen = int(cfg["Sequence"]["seqLength"])

    conf = build_config(args, iw, ih)
    conf["log_path"] = os.path.join(out_dir, f"{seq}.assoc.log")
    tracker = OCSORTTracker(conf)

    det_path = detection_file_path(args, seq)
    dets = np.loadtxt(det_path, delimiter=",")

    outputs_by_frame = {}   # frame -> [(tid, tlbr, score)]
    for fr in range(1, seqlen + 1):
        frame_dets = dets[dets[:, 0] == fr][:, 1:]
        tracker.update(frame_dets)
        rows = []
        for line in tracker.get_outputs():
            p = line.split(",")
            f_, tid = int(p[0]), int(p[1])
            x, y, w, h, sc = map(float, (p[2], p[3], p[4], p[5], p[6]))
            rows.append((tid, tlwh_to_tlbr([x, y, w, h]), sc))
        outputs_by_frame[fr] = rows

    tracker.motion_engine = None  # drop the torch model so the pickle is small/portable
    with open(os.path.join(out_dir, f"{seq}.tracker.pkl"), "wb") as f:
        pickle.dump(tracker, f)
    return tracker, outputs_by_frame, seqlen


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


def analyze(seq, tracker, outputs_by_frame, gt_by_frame, seqlen, thr, out_dir, ctx=2):
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
        cases.append({**sw, "cause": cause, "timeline": timeline})

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
    for seq in seqs:
        print(f"\n### running {seq} ...")
        tracker, outputs_by_frame, seqlen = run_and_capture(args, seq, out_dir)
        gt_by_frame = load_gt(f"{args.datasets_dir}/{args.dataset}/{args.split}/{seq}", is_mot)
        s = analyze(seq, tracker, outputs_by_frame, gt_by_frame, seqlen, args.match_iou, out_dir)
        all_sum.append(s)
    print("\n=== ALL SEQUENCES ===")
    for s in all_sum:
        print(f"  {s['seq']}: IDSW={s['id_switches']} frag={s['fragmentations']} "
              f"causes={s['cause_counts']}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(all_sum, f, indent=2)
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
