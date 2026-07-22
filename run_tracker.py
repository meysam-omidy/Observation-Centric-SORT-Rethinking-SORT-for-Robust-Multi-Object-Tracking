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
import os

import numpy as np
import configparser

from ocsort import OCSORTTracker
from evaluate import evaluate
from utils import count_time


def collect_seqs(args) -> list[str]:
    if args.seqs:
        return args.seqs
    return sorted(os.listdir(f'{args.datasets_dir}/{args.dataset}/{args.split}/'))


def tracker_config(args, image_width: str, image_height: str) -> dict:
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
        'use_byte': args.use_byte,
        'use_oru': args.use_oru,
        'reupdate_type': args.reupdate_type,
        'reupdate_constant_weight': args.reupdate_constant_weight,
        'motion': {
            'enabled': args.motion_enabled,
            'model_type': args.model_type,
            'weights_path': args.weights_path,
            'device': args.device,
            'use_kalman': args.use_kalman,
            'kalman_fusion_blend': args.kalman_fusion_blend,
            'max_gap_norm': args.max_gap_norm,
        },
    }


@count_time
def run(seq: str, args) -> None:
    print(seq)
    detections = np.loadtxt(f'{args.detections_dir}/{args.dataset}/{seq}.txt', delimiter=',')
    config = configparser.ConfigParser()
    config.read(f'{args.datasets_dir}/{args.dataset}/{args.split}/{seq}/seqinfo.ini')
    tracker = OCSORTTracker(tracker_config(
        args,
        config['Sequence']['imWidth'],
        config['Sequence']['imHeight'],
    ))
    os.makedirs(f'outputs/{args.tracker_name}', exist_ok=True)
    with open(f'outputs/{args.tracker_name}/{seq}.txt', 'w') as file:
        for frame_number in range(1, int(config['Sequence']['seqLength']) + 1):
            dets = detections[detections[:, 0] == frame_number][:, 1:]
            tracker.update(dets)
            for output in tracker.get_outputs():
                file.write(f'{output}\n')


def main(args) -> None:
    seqs = collect_seqs(args)
    seqmap_dir = f'./trackeval/seqmap/{args.dataset.lower()}'
    os.makedirs(seqmap_dir, exist_ok=True)
    with open(f'{seqmap_dir}/custom.txt', 'w') as seqmap:
        seqmap.write('name\n')
        for seq in seqs:
            seqmap.write(f'{seq}\n')

    print('tracking...')
    for seq in seqs:
        run(seq, args)

    if args.evaluate:
        print('evaluating...')
        evaluate(
            args.dataset, args.split,
            trackers_to_eval=[args.tracker_name],
            datasets_dir=args.datasets_dir,
        )


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run OC-SORT with the adaptive Kalman Q/R motion model')

    p.add_argument('--dataset', type=str, default='MOT17', choices=['MOT17', 'MOT20', 'DanceTrack'])
    p.add_argument('--split', type=str, default='val')
    p.add_argument('--seqs', type=str, nargs='*', default=None, help='specific sequence names; default = whole split')
    p.add_argument('--datasets_dir', type=str, default='C:/Projects/.Datasets')
    p.add_argument('--detections_dir', type=str, default='C:/Projects/.Detections')
    p.add_argument('--tracker_name', type=str, default='ocsort-self', help='output subfolder under outputs/')
    p.add_argument('--evaluate', action='store_true', help='run trackeval (HOTA/CLEAR/Identity) after tracking')

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
    p.add_argument('--use_byte', action='store_true', default=True)
    p.add_argument('--no_use_byte', action='store_false', dest='use_byte')
    p.add_argument('--use_oru', action='store_true', default=False,
                   help='OC-SORT Observation-Centric Re-Update: replay virtual observations through the KF on re-detection after a gap')
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
        default='../motion-predictor/checkpoints/adaptive_kalman_2/best_model.pth',
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
    if args.reupdate_type == 'none':
        args.reupdate_type = None
    main(args)
