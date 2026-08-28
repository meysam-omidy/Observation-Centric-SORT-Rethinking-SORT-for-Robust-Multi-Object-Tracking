"""
Run OC-SORT over a sequence split, with the adaptive Kalman Q/R model (trained in
../motion-predictor via train_adaptive_kalman.py) driving the Kalman filter's noise.

Example:
  python run_tracker.py \\
    --dataset MOT17 --split val --detection_folder bytetrack_x_mot17 \\
    --model_type adaptive_kalman \\
    --weights_path ../motion-predictor/checkpoints/adaptive_kalman_3/best_model.pth \\
    --kalman_fusion_blend 0 \\
    --evaluate
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
import time

import numpy as np
import configparser

from ocsort import OCSORTTracker
from association_model import AssociationScorerEngine
from evaluate import evaluate
from motion_predictor import MotionPredictorConfig, MotionPredictorEngine


def collect_seqs(args) -> list[str]:
    if args.seqs:
        return args.seqs
    return sorted(os.listdir(f'{args.datasets_dir}/{args.dataset}/{args.split}/'))


def motion_config(args) -> dict:
    return {
        'enabled': args.motion_enabled,
        'model_type': args.model_type,
        'weights_path': args.weights_path,
        'device': args.device,
        'use_kalman': args.use_kalman,
        'kalman_fusion_blend': args.kalman_fusion_blend,
        'max_gap_norm': args.max_gap_norm,
    }


class LockedMotionPredictorEngine:
    """Share one read-only model safely between sequence worker threads."""

    def __init__(self, engine: MotionPredictorEngine):
        self._engine = engine
        self._inference_lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def predict_q_batch(self, *args, **kwargs):
        with self._inference_lock:
            return self._engine.predict_q_batch(*args, **kwargs)

    def predict_r_batch(self, *args, **kwargs):
        with self._inference_lock:
            return self._engine.predict_r_batch(*args, **kwargs)

    def predict_batch(self, *args, **kwargs):
        with self._inference_lock:
            return self._engine.predict_batch(*args, **kwargs)


class LockedAssociationScorerEngine:
    """Share the tiny read-only scorer without reloading it per sequence."""

    def __init__(self, engine: AssociationScorerEngine):
        self._engine = engine
        self._inference_lock = threading.Lock()

    def predict_logits(self, *args, **kwargs):
        with self._inference_lock:
            return self._engine.predict_logits(*args, **kwargs)


def tracker_config(
    args,
    image_width: str,
    image_height: str,
    log_path: str | None = None,
) -> dict:
    return {
        'image_width': image_width,
        'image_height': image_height,
        'max_age': args.max_age,
        'update_window_start': args.update_window_start,
        'update_window_end': args.update_window_end,
        'min_box_area': args.min_box_area,
        'max_aspect_ratio': args.max_aspect_ratio,
        'delta_t': args.delta_t,
        'high_score_det_threshold': args.high_score_det_threshold,
        'low_score_det_threshold': args.low_score_det_threshold,
        'init_track_score_threshold': args.init_track_score_threshold,
        'match_high_score_dets_with_confirmed_trks_threshold': args.match_high_score_dets_with_confirmed_trks_threshold,
        'match_low_score_dets_with_confirmed_trks_threshold': args.match_low_score_dets_with_confirmed_trks_threshold,
        'match_remained_high_score_dets_with_unconfirmed_trks_threshold': args.match_remained_high_score_dets_with_unconfirmed_trks_threshold,
        'association_iou_coefficient': args.association_iou_coefficient,
        'association_speed_direction_coefficient': args.association_speed_direction_coefficient,
        'legacy_post_assignment_iou_gate': getattr(args, 'legacy_post_assignment_iou_gate', False),
        'use_mahalanobis_association': args.use_mahalanobis_association,
        'use_mahalanobis_cost': args.use_mahalanobis_cost,
        'use_mahalanobis_gate': args.use_mahalanobis_gate,
        'mahalanobis_cost_coefficient': args.mahalanobis_cost_coefficient,
        'mahalanobis_gate_threshold': args.mahalanobis_gate_threshold,
        'use_learned_association': getattr(args, 'use_learned_association', False),
        'association_weights_path': getattr(args, 'association_weights_path', None),
        'association_device': getattr(args, 'association_device', None),
        'association_cost_weight': getattr(args, 'association_cost_weight', 0.10),
        'association_residual_clip': getattr(args, 'association_residual_clip', 0.50),
        'use_byte': args.use_byte,
        'use_oru': args.use_oru,
        'use_confidence_r': args.use_confidence_r,
        'use_learned_q': args.use_learned_q,
        'q_scale': args.q_scale,
        'r_scale': args.r_scale,
        'output_lost_tracks': args.output_lost_tracks,
        'lost_output_max_age': args.lost_output_max_age,
        'lost_output_score_decay': args.lost_output_score_decay,
        'lost_output_min_score': args.lost_output_min_score,
        'lost_output_require_inside_frame': args.lost_output_require_inside_frame,
        'suppress_duplicate_track_births': args.suppress_duplicate_track_births,
        'cleanup_duplicate_tracks': args.cleanup_duplicate_tracks,
        'duplicate_track_iou_threshold': args.duplicate_track_iou_threshold,
        'duplicate_track_min_observations': args.duplicate_track_min_observations,
        'duplicate_track_overlap_frames': args.duplicate_track_overlap_frames,
        'prioritize_mature_tracks': args.prioritize_mature_tracks,
        'mature_track_min_observations': args.mature_track_min_observations,
        'reupdate_type': args.reupdate_type,
        'reupdate_constant_weight': args.reupdate_constant_weight,
        'log_path': log_path,
        'motion': motion_config(args),
    }


def detection_file_path(args, seq: str) -> str:
    """Resolve <detections_dir>/<detector_name>/<dataset>/<sequence>.txt."""
    parts = [args.detections_dir]
    if args.detector_name:
        parts.append(args.detector_name)
    parts.extend([args.dataset, f'{seq}.txt'])
    return os.path.join(*parts)


def sequence_log_path(args, seq: str) -> str | None:
    """Resolve a separate association log for each sequence.

    ``--log_path logs`` writes ``logs/<sequence>.assoc.log``. A path containing
    ``{seq}`` is treated as a template, while a filename ending in ``.log`` is
    used directly for a single sequence and receives a ``-<sequence>`` suffix
    for a multi-sequence run.
    """
    requested_path = args.log_path
    if not requested_path:
        return None

    if '{seq}' in requested_path:
        path = requested_path.replace('{seq}', seq)
    elif os.path.splitext(requested_path)[1].lower() == '.log':
        seqs = args.seqs or []
        if len(set(seqs)) <= 1:
            path = requested_path
        else:
            root, extension = os.path.splitext(requested_path)
            path = f'{root}-{seq}{extension}'
    else:
        path = os.path.join(requested_path, f'{seq}.assoc.log')

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def run(seq: str, args, motion_engine=None, association_engine=None) -> None:
    print(f'[{seq}] starting')
    started = time.perf_counter()
    detections = np.loadtxt(detection_file_path(args, seq), delimiter=',')
    config = configparser.ConfigParser()
    config.read(f'{args.datasets_dir}/{args.dataset}/{args.split}/{seq}/seqinfo.ini')
    frame_count = int(config['Sequence']['seqLength'])
    tracker = OCSORTTracker(
        tracker_config(
            args,
            config['Sequence']['imWidth'],
            config['Sequence']['imHeight'],
            log_path=sequence_log_path(args, seq),
        ),
        motion_engine=motion_engine,
        association_engine=association_engine,
    )
    os.makedirs(f'outputs/{args.tracker_name}', exist_ok=True)
    output_path = f'outputs/{args.tracker_name}/{seq}.txt'
    temporary_path = f'{output_path}.tmp-{os.getpid()}-{threading.get_ident()}'
    try:
        with open(temporary_path, 'w') as file:
            for frame_number in range(1, frame_count + 1):
                dets = detections[detections[:, 0] == frame_number][:, 1:]
                tracker.update(dets)
                for output in tracker.get_outputs():
                    file.write(f'{output}\n')
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    elapsed = time.perf_counter() - started
    fps = frame_count / elapsed if elapsed > 0 else float('inf')
    print(f'[{seq}] completed in {elapsed:.2f}s ({fps:.1f} FPS)')


def build_shared_motion_engine(args, workers: int):
    if not args.motion_enabled:
        return None
    try:
        engine = MotionPredictorEngine(MotionPredictorConfig.model_validate(motion_config(args)))
    except FileNotFoundError as err:
        print(f'[run_tracker] {err}; using heuristic motion prediction.')
        args.motion_enabled = False
        return None
    return LockedMotionPredictorEngine(engine) if workers > 1 else engine


def build_shared_association_engine(args, workers: int):
    if not getattr(args, 'use_learned_association', False):
        return None
    if not getattr(args, 'association_weights_path', None):
        raise ValueError("--use_learned_association requires --association_weights_path")
    engine = AssociationScorerEngine(
        args.association_weights_path, getattr(args, 'association_device', None)
    )
    return LockedAssociationScorerEngine(engine) if workers > 1 else engine


def main(args) -> None:
    # Preserve user ordering while preventing two workers from targeting the same file.
    seqs = list(dict.fromkeys(collect_seqs(args)))
    if not seqs:
        raise ValueError('no sequences found')
    workers = min(args.sequence_workers, len(seqs))
    seqmap_dir = f'./trackeval/seqmap/{args.dataset.lower()}'
    os.makedirs(seqmap_dir, exist_ok=True)
    with open(f'{seqmap_dir}/custom.txt', 'w') as seqmap:
        seqmap.write('name\n')
        for seq in seqs:
            seqmap.write(f'{seq}\n')

    os.makedirs(f'outputs/{args.tracker_name}', exist_ok=True)
    motion_engine = build_shared_motion_engine(args, workers)
    association_engine = build_shared_association_engine(args, workers)

    print(f'tracking {len(seqs)} sequence(s) with {workers} worker(s)...')
    def run_sequence(seq):
        # Retain the long-standing three-argument run call when the optional
        # scorer is disabled (also keeps external scripts/tests compatible).
        if association_engine is None:
            return run(seq, args, motion_engine)
        return run(seq, args, motion_engine, association_engine)

    if workers == 1:
        for seq in seqs:
            run_sequence(seq)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ocsort-seq') as executor:
            futures = {
                executor.submit(run_sequence, seq): seq
                for seq in seqs
            }
            for future in as_completed(futures):
                # Propagate worker failures and do not evaluate partial output.
                future.result()

    if args.evaluate:
        print('evaluating...')
        evaluate(
            args.dataset, args.split,
            trackers_to_eval=[args.tracker_name, 'ocsort-self-v', 'oc-sort', 'ocsort-self-wbrt', 'official-ocsort-yoloxx'],
            datasets_dir=args.datasets_dir,
        )


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run OC-SORT with the adaptive Kalman Q/R motion model')

    p.add_argument('--dataset', type=str, default='MOT17', choices=['MOT17', 'MOT20', 'DanceTrack'])
    p.add_argument('--split', type=str, default='val')
    p.add_argument('--seqs', type=str, nargs='*', default=None, help='specific sequence names; default = whole split')
    p.add_argument('--datasets_dir', type=str, default='C:/Projects/.Datasets')
    p.add_argument('--detections_dir', type=str, default='C:/Projects/.Detections',
                   help='root directory containing detector subfolders')
    p.add_argument('--detector_name', type=str, default='YOLOXx',
                   help='detector subfolder under --detections_dir, e.g. YOLO11x, YOLO26x, YOLOXx')
    p.add_argument('--tracker_name', type=str, default='ocsort-self', help='output subfolder under outputs/')
    p.add_argument(
        '--log_path', type=str, default=None,
        help=(
            'association-log directory, .log file, or path template containing {seq}; '
            'multi-sequence runs always receive separate logs'
        ),
    )
    p.add_argument('--evaluate', action='store_true', help='run trackeval (HOTA/CLEAR/Identity) after tracking')
    p.add_argument(
        '--sequence_workers', type=int, default=1,
        help='number of independent dataset sequences to track concurrently',
    )

    p.add_argument('--max_age', type=int, default=30)
    p.add_argument('--update_window_start', type=int, default=30)
    p.add_argument('--update_window_end', type=int, default=50)
    p.add_argument('--min_box_area', type=int, default=100)
    p.add_argument('--max_aspect_ratio', type=float, default=1.6)
    p.add_argument('--delta_t', type=int, default=3)
    p.add_argument('--high_score_det_threshold', type=float, default=0.6)
    p.add_argument('--low_score_det_threshold', type=float, default=0.1)
    p.add_argument('--init_track_score_threshold', type=float, default=0.6)
    p.add_argument('--match_high_score_dets_with_confirmed_trks_threshold', type=float, default=0.2)
    p.add_argument('--match_low_score_dets_with_confirmed_trks_threshold', type=float, default=0.5)
    p.add_argument('--match_remained_high_score_dets_with_unconfirmed_trks_threshold', type=float, default=0.3)
    p.add_argument('--association_iou_coefficient', type=float, default=1.0)
    p.add_argument('--association_speed_direction_coefficient', type=float, default=0.3)
    p.add_argument(
        '--legacy_post_assignment_iou_gate', action='store_true', default=False,
        help='compatibility mode: run Hungarian on raw association cost, then remove IoU-invalid matches',
    )
    p.add_argument('--use_mahalanobis_association', action='store_true', default=False,
                   help='legacy alias: enable both Mahalanobis soft cost and hard gate')
    p.add_argument('--use_mahalanobis_cost', action='store_true', default=False,
                   help='add a soft KF innovation-distance cost without rejecting candidates')
    p.add_argument('--use_mahalanobis_gate', action='store_true', default=False,
                   help='reject covariance-improbable candidates without adding Mahalanobis cost')
    p.add_argument('--mahalanobis_cost_coefficient', type=float, default=1.0,
                   help='weight of squared Mahalanobis distance normalized by the fixed 4-D 95%% reference')
    p.add_argument('--mahalanobis_gate_threshold', type=float, default=9.4877,
                   help='maximum squared Mahalanobis distance when --use_mahalanobis_gate is enabled')
    p.add_argument('--use_learned_association', action='store_true', default=False,
                   help='add a trained no-ReID association residual to valid OC-SORT candidate costs')
    p.add_argument('--association_weights_path', type=str, default=None,
                   help='checkpoint produced by train_association_model.py')
    p.add_argument('--association_device', type=str, default=None,
                   help='device for association scorer; default = auto')
    p.add_argument('--association_cost_weight', type=float, default=0.10,
                   help='non-negative multiplier for bounded learned association residual')
    p.add_argument('--association_residual_clip', type=float, default=0.50,
                   help='absolute residual cap before association_cost_weight is applied')
    p.add_argument('--use_byte', action='store_true', default=True)
    p.add_argument('--no_use_byte', action='store_false', dest='use_byte')
    p.add_argument('--use_oru', action='store_true', default=False,
                   help='OC-SORT Observation-Centric Re-Update: replay virtual observations through the KF on re-detection after a gap')
    p.add_argument('--no_use_oru', action='store_false', dest='use_oru')
    p.add_argument('--use_confidence_r', action='store_true', default=False,
                   help='wbrt-style simple per-frame confidence R (diag[1,1,10,10]*e^(2(1-conf))); trusts measured scale, overrides learned var_r')
    p.add_argument('--use_learned_q', action='store_true', default=True,
                   help="use the model's var_q for process noise Q")
    p.add_argument('--no_use_learned_q', action='store_false', dest='use_learned_q',
                   help="ignore the model's var_q; keep the KF's fixed Q (pairs well with learned R)")
    p.add_argument(
        '--q_scale', type=float, default=1,
        help='positive multiplier applied to the final learned Kalman Q matrix',
    )
    p.add_argument(
        '--r_scale', type=float, default=1,
        help='positive multiplier applied to the final learned Kalman R matrix',
    )
    p.add_argument('--output_lost_tracks', action='store_true', default=False,
                   help='emit short-lived predicted boxes for confirmed tracks that miss detections')
    p.add_argument('--lost_output_max_age', type=int, default=3,
                   help='maximum consecutive missed frames to emit when --output_lost_tracks is enabled')
    p.add_argument('--lost_output_score_decay', type=float, default=0.7,
                   help='per-missed-frame multiplier for a predicted track output score')
    p.add_argument('--lost_output_min_score', type=float, default=0.3,
                   help='do not emit a predicted track once its decayed score is below this value')
    p.add_argument('--lost_output_require_inside_frame', action='store_true', default=True,
                   help='only emit a lost prediction when its whole box remains inside the frame')
    p.add_argument('--lost_output_allow_partial_outside', action='store_false',
                   dest='lost_output_require_inside_frame',
                   help='allow lost-track predictions whose boxes partially leave the frame')
    p.add_argument(
        '--suppress_duplicate_track_births', action='store_true', default=False,
        help='do not initialize an unmatched high-score detection that overlaps a mature live/lost track',
    )
    p.add_argument(
        '--cleanup_duplicate_tracks', action='store_true', default=False,
        help='retire a weaker track after persistent high-IoU overlap with a mature observed track',
    )
    p.add_argument(
        '--duplicate_track_iou_threshold', type=float, default=0.85,
        help='IoU required by duplicate-birth suppression and duplicate-track cleanup',
    )
    p.add_argument(
        '--duplicate_track_min_observations', type=int, default=3,
        help='real detector updates required before a track participates in duplicate control',
    )
    p.add_argument(
        '--duplicate_track_overlap_frames', type=int, default=3,
        help='consecutive observed overlapping frames required before duplicate cleanup retires a track',
    )
    p.add_argument(
        '--prioritize_mature_tracks', action='store_true', default=False,
        help='associate mature currently tracking identities before younger or lost tracks in phase 1',
    )
    p.add_argument(
        '--mature_track_min_observations', type=int, default=3,
        help='real detector updates required for phase-1 mature-track priority',
    )
    p.add_argument('--reupdate_type', type=str, default='constant', choices=['constant', 'relative', 'none'])
    p.add_argument('--reupdate_constant_weight', type=float, default=0.8)

    p.add_argument('--motion_enabled', action='store_true', default=True)
    p.add_argument('--no_motion', action='store_false', dest='motion_enabled')
    p.add_argument(
        '--model_type', type=str, default='adaptive_kalman',
        choices=['transformer', 'transformer_learned', 'lstm', 'lstm_learned', 'adaptive_kalman'],
    )
    p.add_argument(
        '--weights_path', type=str,
        default='../motion-predictor/checkpoints/adaptive_kalman_real_low_data/best_model.pth',
        # default='../motion-predictor/checkpoints/adaptive_kalman_2/best_model.pth',
    )
    p.add_argument('--device', type=str, default=None, help='cuda | cpu | mps; default = auto')
    p.add_argument('--use_kalman', action='store_true', default=True)
    p.add_argument('--no_use_kalman', action='store_false', dest='use_kalman')
    p.add_argument(
        '--kalman_fusion_blend', type=float, default=0.0,
        help='0 = pure Kalman/CV prediction (adaptive_kalman has no bbox head); >0 blends in the model bbox (legacy model types only)',
    )
    p.add_argument(
        '--max_gap_norm', type=float, default=None,
        help='adaptive_kalman only; None = read from checkpoint (falls back to 30.0)',
    )

    args = p.parse_args()
    if args.sequence_workers < 1:
        p.error('--sequence_workers must be at least 1')
    if args.reupdate_type == 'none':
        args.reupdate_type = None
    if args.log_path == 'none':
        args.log_path = None
    main(args)
