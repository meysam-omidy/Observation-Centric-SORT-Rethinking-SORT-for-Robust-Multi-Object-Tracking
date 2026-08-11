"""Run reproducible adaptive-Kalman Q/R scale experiments.

The default staged search first sweeps R with Q=1, selects the best HOTA, then
sweeps Q at that R.  ``--full_grid`` evaluates the complete Cartesian product.
Every run has a unique tracker/output name and the final metrics are collected in
``results/<prefix>-summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from evaluate import evaluate


DEFAULT_Q_SCALES = (0.1, 0.25, 0.5, 1.0)
DEFAULT_R_SCALES = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('scale values must be greater than zero')
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('worker count must be at least one')
    return number


def scale_tag(value: float) -> str:
    """Filesystem-friendly, stable representation such as 0p02 or 1."""
    return format(float(value), '.12g').replace('.', 'p')


@dataclass(frozen=True)
class Experiment:
    name: str
    mode: str
    q_scale: float | None
    r_scale: float | None


def learned_experiment(prefix: str, q_scale: float, r_scale: float) -> Experiment:
    name = f'{prefix}-q{scale_tag(q_scale)}-r{scale_tag(r_scale)}'
    return Experiment(name, 'learned_qr', q_scale, r_scale)


def fixed_confidence_experiment(prefix: str) -> Experiment:
    return Experiment(f'{prefix}-fixedq-confr', 'fixed_q_confidence_r', None, None)


def run_experiment(experiment: Experiment, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name('run_tracker.py')),
        '--dataset', args.dataset,
        '--split', args.split,
        '--datasets_dir', args.datasets_dir,
        '--detections_dir', args.detections_dir,
        '--model_type', 'adaptive_kalman',
        '--weights_path', args.weights_path,
        '--kalman_fusion_blend', '0',
        '--sequence_workers', str(args.sequence_workers),
        '--tracker_name', experiment.name,
    ]
    if args.seqs:
        command.extend(['--seqs', *args.seqs])
    if args.device is not None:
        command.extend(['--device', args.device])
    if experiment.mode == 'fixed_q_confidence_r':
        command.extend(['--no_use_learned_q', '--use_confidence_r'])
    else:
        command.extend([
            '--q_scale', str(experiment.q_scale),
            '--r_scale', str(experiment.r_scale),
        ])
    extra_args = args.tracker_args[1:] if args.tracker_args[:1] == ['--'] else args.tracker_args
    command.extend(extra_args)

    print(f'\n=== Tracking {experiment.name} ===', flush=True)
    subprocess.run(command, check=True, cwd=Path(__file__).parent)


def read_result(experiment: Experiment) -> dict[str, str]:
    result_path = Path(__file__).parent / 'results' / f'{experiment.name}-results.txt'
    metrics: dict[str, str] = {}
    with result_path.open(encoding='utf-8') as result_file:
        for line in result_file:
            key, separator, value = line.partition(':')
            if separator:
                metrics[key.strip().upper()] = value.strip()
    return metrics


def evaluate_experiments(experiments: list[Experiment], args: argparse.Namespace) -> None:
    evaluate(
        args.dataset,
        args.split,
        trackers_to_eval=[experiment.name for experiment in experiments],
        datasets_dir=args.datasets_dir,
    )


def write_summary(experiments: list[Experiment], args: argparse.Namespace) -> Path:
    rows = []
    for experiment in experiments:
        metrics = read_result(experiment)
        rows.append({
            'tracker': experiment.name,
            'mode': experiment.mode,
            'q_scale': '' if experiment.q_scale is None else experiment.q_scale,
            'r_scale': '' if experiment.r_scale is None else experiment.r_scale,
            'HOTA': metrics.get('HOTA', ''),
            'ASSA': metrics.get('ASSA', ''),
            'DETA': metrics.get('DETA', ''),
            'IDF1': metrics.get('IDF1', ''),
            'MOTA': metrics.get('MOTA', ''),
            'IDSW': metrics.get('IDSW', ''),
        })
    rows.sort(key=lambda row: float(row['HOTA'] or '-inf'), reverse=True)

    path = Path(__file__).parent / 'results' / f'{args.prefix}-summary.csv'
    with path.open('w', newline='', encoding='utf-8') as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print('\n=== Q/R scale ranking (HOTA) ===')
    for row in rows:
        print(
            f"{row['tracker']:<36} HOTA={row['HOTA']:<10} "
            f"AssA={row['ASSA']:<10} IDF1={row['IDF1']:<10} IDSW={row['IDSW']}"
        )
    print(f'\nSummary written to {path}')
    return path


def unique_experiments(experiments: list[Experiment]) -> list[Experiment]:
    return list(dict.fromkeys(experiments))


def main(args: argparse.Namespace) -> None:
    controls = [] if args.no_fixed_confidence_control else [
        fixed_confidence_experiment(args.prefix)
    ]

    if args.full_grid:
        learned = [
            learned_experiment(args.prefix, q_scale, r_scale)
            for q_scale in args.q_scales
            for r_scale in args.r_scales
        ]
        experiments = unique_experiments(controls + learned)
        for experiment in experiments:
            run_experiment(experiment, args)
        evaluate_experiments(experiments, args)
        write_summary(experiments, args)
        return

    r_experiments = unique_experiments([
        learned_experiment(args.prefix, 1.0, r_scale)
        for r_scale in args.r_scales
    ])
    first_stage = unique_experiments(controls + r_experiments)
    for experiment in first_stage:
        run_experiment(experiment, args)
    evaluate_experiments(first_stage, args)

    best_r_experiment = max(
        r_experiments,
        key=lambda experiment: float(read_result(experiment)['HOTA']),
    )
    best_r = best_r_experiment.r_scale
    print(f'\nBest R scale with Q=1: {best_r}')

    q_experiments = unique_experiments([
        learned_experiment(args.prefix, q_scale, best_r)
        for q_scale in args.q_scales
    ])
    already_run = {experiment.name for experiment in first_stage}
    for experiment in q_experiments:
        if experiment.name not in already_run:
            run_experiment(experiment, args)

    experiments = unique_experiments(first_stage + q_experiments)
    evaluate_experiments(experiments, args)
    write_summary(experiments, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sweep final learned Q/R matrix scales and rank DanceTrack HOTA.',
    )
    parser.add_argument('--dataset', default='DanceTrack', choices=['MOT17', 'MOT20', 'DanceTrack'])
    parser.add_argument('--seqs', type=str, nargs='*', default=None, help='specific sequence names; default = whole split')
    parser.add_argument('--split', default='val')
    parser.add_argument('--datasets_dir', default='C:/Projects/.Datasets')
    parser.add_argument('--detections_dir', default='C:/Projects/.Detections')
    parser.add_argument(
        '--weights_path',
        default=(
            '../motion-predictor/checkpoints/'
            'kf_adaptive_kalman_real_low_data_light_transformer_lr_new_kftrackoff2/'
            'best_model.pth'
        ),
    )
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--sequence_workers', type=positive_int, default=4,
        help='parallel sequence workers inside each tracker experiment',
    )
    parser.add_argument('--prefix', default='adaptive-qr-scale')
    parser.add_argument('--q_scales', nargs='+', type=positive_float, default=list(DEFAULT_Q_SCALES))
    parser.add_argument('--r_scales', nargs='+', type=positive_float, default=list(DEFAULT_R_SCALES))
    parser.add_argument('--full_grid', action='store_true', help='run all Q/R combinations instead of the staged sweep')
    parser.add_argument('--no_fixed_confidence_control', action='store_true')
    parser.add_argument(
        'tracker_args', nargs=argparse.REMAINDER,
        help='extra run_tracker.py arguments; place them after --',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
