"""Run reproducible OC-SORT ablations on DanceTrack validation.

The experiments are intentionally one-factor-at-a-time: every experiment starts
from the same adaptive-Kalman baseline and changes exactly one association or
track-management decision.  This makes the final ranking useful for selecting
parameters instead of mixing several causes in one result.

Examples
--------
Quick comparison on the curated difficult sequences::

    python run_dancetrack_ablations.py --scope hard --profile core

Select parameters on the complete DanceTrack validation set::

    python run_dancetrack_ablations.py --scope val --profile extended \
        --sequence_workers 4 --device cuda

Results are kept in ``ablation_results/<run_name>``.  ``ranking.csv`` and
``best_params.json`` are the primary selection artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate import evaluate


# Sequences with frequent close interactions, occlusion, or previously observed
# association failures.  Change this list with --seqs without editing the file.
HARD_DANCETRACK_VAL_SEQS = (
    'dancetrack0026', 'dancetrack0034', 'dancetrack0041', 'dancetrack0043', 'dancetrack0081', 'dancetrack0094'
)
# HARD_DANCETRACK_VAL_SEQS = (
#     'dancetrack0035', 'dancetrack0041', 'dancetrack0043', 'dancetrack0047',
#     'dancetrack0058', 'dancetrack0063', 'dancetrack0065', 'dancetrack0073',
#     'dancetrack0077', 'dancetrack0079', 'dancetrack0081', 'dancetrack0090',
#     'dancetrack0094', 'dancetrack0097',
# )

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MOTION_WEIGHTS = (
    SCRIPT_DIR.parent / 'motion-predictor' / 'checkpoints' /
    'kf_adaptive_kalman_real_low_data_lightvv_transformer_newarch_kftrackoff' /
    'best_model.pth'
)
DEFAULT_ASSOCIATION_WEIGHTS = (
    SCRIPT_DIR / 'checkpoints' / 'association_multidataset_v2' / 'best_model.pth'
)

# Established best settings.  Every experiment inherits these flags unless it
# explicitly opts out, so a run never silently falls back to ORU or the modern
# pre-assignment IoU handling.
BASELINE_TRACKER_ARGS = ('--no_use_oru', '--legacy_post_assignment_iou_gate')
BASELINE_PARAMETERS = {
    'use_byte': True,
    'use_oru': False,
    'use_learned_association': False,
    'use_mahalanobis_cost': False,
    'use_mahalanobis_gate': False,
    'legacy_post_assignment_iou_gate': True,
}


@dataclass(frozen=True)
class Experiment:
    """A named, one-factor modification to the common baseline."""

    key: str
    description: str
    tracker_args: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    use_baseline_defaults: bool = True


def effective_tracker_args(experiment: Experiment) -> tuple[str, ...]:
    """Flags actually passed to run_tracker.py, including the winning baseline."""
    prefix = BASELINE_TRACKER_ARGS if experiment.use_baseline_defaults else ()
    return (*prefix, *experiment.tracker_args)


def effective_parameters(experiment: Experiment) -> dict[str, Any]:
    """Selection-friendly parameter record for the exact experiment command."""
    parameters = dict(BASELINE_PARAMETERS) if experiment.use_baseline_defaults else {}
    parameters.update(experiment.parameters)
    return parameters


def experiments(profile: str, association_weights: Path) -> list[Experiment]:
    """Return the pre-defined, auditable ablation suite."""
    baseline = Experiment(
        'baseline',
        'Adaptive Kalman baseline with legacy post-assignment IoU gating and ORU disabled.',
    )
    core = [
        baseline,
        Experiment('no-byte', 'Disable ByteTrack low-score recovery.', ('--no_use_byte',), {
            'use_byte': False,
        }),
        Experiment('restore-oru', 'Restore ORU while retaining legacy post-assignment IoU gating.', ('--use_oru',), {
            'use_oru': True,
        }),
        Experiment(
            'learned-association', 'Enable the trained association residual.',
            ('--use_learned_association', '--association_weights_path', str(association_weights)),
            {'use_learned_association': True, 'association_weights_path': str(association_weights)},
        ),
        Experiment(
            'mahalanobis-soft', 'Add a soft normalized innovation-distance cost.',
            ('--use_mahalanobis_cost', '--mahalanobis_cost_coefficient', '0.05'),
            {'use_mahalanobis_cost': True, 'mahalanobis_cost_coefficient': 0.05},
        ),
        Experiment(
            'mahalanobis-gate', 'Reject covariance-improbable matches.',
            ('--use_mahalanobis_gate', '--mahalanobis_gate_threshold', '9.4877'),
            {'use_mahalanobis_gate': True, 'mahalanobis_gate_threshold': 9.4877},
        ),
        Experiment(
            'modern-iou-assignment',
            'Disable the legacy IoU behavior while keeping ORU disabled.',
            ('--no_use_oru',),
            {'use_oru': False, 'legacy_post_assignment_iou_gate': False},
            use_baseline_defaults=False,
        ),
        Experiment(
            'duplicate-birth-suppression', 'Block overlapping duplicate track births.',
            ('--suppress_duplicate_track_births',),
            {'suppress_duplicate_track_births': True},
        ),
    ]
    if profile == 'core':
        return core
    return core + [
        Experiment(
            'mahalanobis-soft-and-gate', 'Use both Mahalanobis soft cost and hard gate.',
            (
                '--use_mahalanobis_cost', '--mahalanobis_cost_coefficient', '0.05',
                '--use_mahalanobis_gate', '--mahalanobis_gate_threshold', '9.4877',
            ),
            {
                'use_mahalanobis_cost': True, 'mahalanobis_cost_coefficient': 0.05,
                'use_mahalanobis_gate': True, 'mahalanobis_gate_threshold': 9.4877,
            },
        ),
        Experiment(
            'learned-association-no-byte',
            'Learned association residual without Byte low-score recovery.',
            (
                '--no_use_byte', '--use_learned_association',
                '--association_weights_path', str(association_weights),
            ),
            {
                'use_byte': False, 'use_learned_association': True,
                'association_weights_path': str(association_weights),
            },
        ),
        Experiment(
            'duplicate-cleanup', 'Retire persistent duplicate tracks.',
            ('--cleanup_duplicate_tracks',), {'cleanup_duplicate_tracks': True},
        ),
        Experiment(
            'mature-track-priority', 'Associate mature active identities before young/lost tracks.',
            ('--prioritize_mature_tracks',), {'prioritize_mature_tracks': True},
        ),
        Experiment(
            'lost-track-output', 'Emit high-confidence short-gap track predictions.',
            (
                '--output_lost_tracks', '--lost_output_max_age', '3',
                '--lost_output_score_decay', '0.7', '--lost_output_min_score', '0.3',
            ),
            {
                'output_lost_tracks': True, 'lost_output_max_age': 3,
                'lost_output_score_decay': 0.7, 'lost_output_min_score': 0.3,
            },
        ),
        Experiment(
            'learned-association-mahalanobis-soft',
            'Test whether learned association benefits from a soft Mahalanobis cost.',
            (
                '--use_learned_association', '--association_weights_path', str(association_weights),
                '--use_mahalanobis_cost', '--mahalanobis_cost_coefficient', '0.05',
            ),
            {
                'use_learned_association': True, 'association_weights_path': str(association_weights),
                'use_mahalanobis_cost': True, 'mahalanobis_cost_coefficient': 0.05,
            },
        ),
    ]


def read_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        key, separator, value = line.partition(':')
        if not separator:
            continue
        try:
            metrics[key.strip().upper()] = float(value.strip())
        except ValueError:
            pass
    return metrics


def resolve_sequences(args: argparse.Namespace) -> list[str]:
    if args.seqs:
        return list(dict.fromkeys(args.seqs))
    if args.scope == 'hard':
        return list(HARD_DANCETRACK_VAL_SEQS)
    val_dir = args.datasets_dir / 'DanceTrack' / 'val'
    if not val_dir.is_dir():
        raise FileNotFoundError(f'DanceTrack validation directory not found: {val_dir}')
    return sorted(path.name for path in val_dir.iterdir() if path.is_dir())


def run_experiment(
    experiment: Experiment,
    args: argparse.Namespace,
    seqs: list[str],
    run_dir: Path,
) -> str:
    tracker_name = f'{args.tracker_prefix}-{experiment.key}'
    command = [
        sys.executable, str(SCRIPT_DIR / 'run_tracker.py'),
        '--dataset', 'DanceTrack', '--split', 'val',
        '--datasets_dir', str(args.datasets_dir),
        '--detections_dir', str(args.detections_dir),
        '--detector_name', args.detector_name,
        '--tracker_name', tracker_name,
        '--model_type', 'adaptive_kalman',
        '--weights_path', str(args.weights_path),
        '--kalman_fusion_blend', '0',
        '--sequence_workers', str(args.sequence_workers),
        '--seqs', *seqs,
        *effective_tracker_args(experiment),
    ]
    if args.device:
        command.extend(['--device', args.device])
    if args.tracker_args:
        command.extend(args.tracker_args)

    (run_dir / 'commands' / f'{experiment.key}.json').write_text(
        json.dumps({'tracker_name': tracker_name, 'command': command}, indent=2),
        encoding='utf-8',
    )
    print(f'\n=== [{experiment.key}] {experiment.description} ===', flush=True)
    with (run_dir / 'logs' / f'{experiment.key}.log').open('w', encoding='utf-8') as log:
        process = subprocess.run(
            command, cwd=SCRIPT_DIR, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return tracker_name


def evaluate_and_collect(
    experiment_list: list[Experiment], tracker_names: dict[str, str],
    args: argparse.Namespace, run_dir: Path,
) -> list[dict[str, Any]]:
    print('\n=== Evaluating all completed experiments ===', flush=True)
    evaluate(
        'DanceTrack', 'val',
        trackers_to_eval=[tracker_names[experiment.key] for experiment in experiment_list],
        datasets_dir=str(args.datasets_dir),
    )
    rows: list[dict[str, Any]] = []
    for experiment in experiment_list:
        tracker_name = tracker_names[experiment.key]
        source_metrics = SCRIPT_DIR / 'results' / f'{tracker_name}-results.txt'
        source_per_sequence = SCRIPT_DIR / 'results' / 'per_seq' / f'{tracker_name}.csv'
        if not source_metrics.is_file():
            raise FileNotFoundError(f'TrackEval did not write metrics for {tracker_name}: {source_metrics}')
        metrics = read_metrics(source_metrics)
        record = {
            'experiment': experiment.key,
            'description': experiment.description,
            'tracker_name': tracker_name,
            'parameters': effective_parameters(experiment),
            'metrics': metrics,
        }
        (run_dir / 'metrics' / f'{experiment.key}.json').write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding='utf-8',
        )
        shutil.copy2(source_metrics, run_dir / 'metrics' / f'{experiment.key}.txt')
        if source_per_sequence.is_file():
            shutil.copy2(source_per_sequence, run_dir / 'per_sequence' / f'{experiment.key}.csv')
        rows.append({
            'experiment': experiment.key,
            'description': experiment.description,
            'tracker_name': tracker_name,
            **{name: metrics.get(name, '') for name in ('HOTA', 'ASSA', 'DETA', 'IDF1', 'MOTA', 'IDSW')},
            'parameters_json': json.dumps(effective_parameters(experiment), sort_keys=True),
        })
    return rows


def write_ranking(rows: list[dict[str, Any]], args: argparse.Namespace, run_dir: Path) -> None:
    lower_is_better = args.rank_by == 'IDSW'
    rows.sort(key=lambda row: float(row[args.rank_by]), reverse=not lower_is_better)
    for rank, row in enumerate(rows, start=1):
        row['rank'] = rank
    fields = ['rank', 'experiment', 'description', 'tracker_name', 'HOTA', 'ASSA', 'DETA', 'IDF1', 'MOTA', 'IDSW', 'parameters_json']
    with (run_dir / 'ranking.csv').open('w', newline='', encoding='utf-8') as summary:
        writer = csv.DictWriter(summary, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    best = rows[0]
    best_params = {
        'selection_metric': args.rank_by,
        'scope': args.scope,
        'best_experiment': best['experiment'],
        'tracker_name': best['tracker_name'],
        'metrics': {key: best[key] for key in ('HOTA', 'ASSA', 'DETA', 'IDF1', 'MOTA', 'IDSW')},
        'effective_parameters': json.loads(best['parameters_json']),
        'reproduce_with': json.loads(
            (run_dir / 'commands' / f"{best['experiment']}.json").read_text(encoding='utf-8')
        )['command'],
    }
    (run_dir / 'best_params.json').write_text(json.dumps(best_params, indent=2), encoding='utf-8')
    print(f"\nBest {args.rank_by}: {best['experiment']} ({best[args.rank_by]})")
    print(f'Results: {run_dir}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['hard', 'val'], default='hard',
                        help='hard = curated difficult sequences; val = every DanceTrack val sequence')
    parser.add_argument('--seqs', nargs='*', default=None,
                        help='explicit sequence list; overrides --scope')
    parser.add_argument('--profile', choices=['core', 'extended'], default='core',
                        help='core runs 8 focused ablations; extended adds 6 management/interaction tests')
    parser.add_argument('--datasets_dir', type=Path, default=Path('C:/Projects/.Datasets'))
    parser.add_argument('--detections_dir', type=Path, default=Path('C:/Projects/.Detections'))
    parser.add_argument('--detector_name', default='YOLOXx')
    parser.add_argument('--weights_path', type=Path, default=DEFAULT_MOTION_WEIGHTS)
    parser.add_argument('--association_weights_path', type=Path, default=DEFAULT_ASSOCIATION_WEIGHTS)
    parser.add_argument('--device', default=None)
    parser.add_argument('--sequence_workers', type=int, default=1)
    parser.add_argument('--rank_by', choices=['HOTA', 'ASSA', 'DETA', 'IDF1', 'MOTA', 'IDSW'], default='HOTA')
    parser.add_argument('--run_name', default=None,
                        help='output folder name; default includes scope/profile and UTC timestamp')
    parser.add_argument('--tracker_prefix', default='dancetrack-ablation',
                        help='prefix for outputs/<tracker_name> folders')
    parser.add_argument('tracker_args', nargs=argparse.REMAINDER,
                        help='extra run_tracker.py arguments; put them after --')
    args = parser.parse_args()
    if args.sequence_workers < 1:
        parser.error('--sequence_workers must be at least 1')
    if args.tracker_args[:1] == ['--']:
        args.tracker_args = args.tracker_args[1:]
    return args


def main(args: argparse.Namespace) -> None:
    if not args.weights_path.is_file():
        raise FileNotFoundError(f'adaptive-Kalman checkpoint not found: {args.weights_path}')
    if not args.association_weights_path.is_file():
        raise FileNotFoundError(f'association checkpoint not found: {args.association_weights_path}')
    seqs = resolve_sequences(args)
    if not seqs:
        raise ValueError('no DanceTrack sequences selected')
    run_name = args.run_name or (
        f'dancetrack-{args.scope}-{args.profile}-'
        f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    )
    run_dir = SCRIPT_DIR / 'ablation_results' / run_name
    if run_dir.exists():
        raise FileExistsError(f'output directory already exists: {run_dir}; choose --run_name')
    for directory in ('commands', 'configs', 'logs', 'metrics', 'per_sequence'):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)

    suite = experiments(args.profile, args.association_weights_path.resolve())
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': args.scope,
        'sequences': seqs,
        'motion_weights': str(args.weights_path.resolve()),
        'association_weights': str(args.association_weights_path.resolve()),
        'baseline_tracker_args': BASELINE_TRACKER_ARGS,
        'baseline_parameters': BASELINE_PARAMETERS,
        'experiments': [
            {
                **asdict(experiment),
                'effective_tracker_args': effective_tracker_args(experiment),
                'effective_parameters': effective_parameters(experiment),
            }
            for experiment in suite
        ],
    }
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for experiment in suite:
        (run_dir / 'configs' / f'{experiment.key}.json').write_text(
            json.dumps({
                **asdict(experiment),
                'effective_tracker_args': effective_tracker_args(experiment),
                'effective_parameters': effective_parameters(experiment),
            }, indent=2), encoding='utf-8',
        )

    tracker_names = {
        experiment.key: run_experiment(experiment, args, seqs, run_dir)
        for experiment in suite
    }
    rows = evaluate_and_collect(suite, tracker_names, args, run_dir)
    write_ranking(rows, args, run_dir)


if __name__ == '__main__':
    main(parse_args())
