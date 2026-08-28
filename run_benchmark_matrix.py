"""Run the three requested tracker methods over all datasets and detectors.

Metrics, commands, logs, and per-sequence CSVs are kept under
benchmark_results/method_grid_20260829.  Each tracker output name starts with
the requested method prefix, so it remains identifiable under outputs/ too.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "benchmark_results" / "method_grid_20260829"
DATASETS = ("MOT17", "MOT20", "DanceTrack")
# The requested "YOLO6x" is not present in C:/Projects/.Detections; this is
# the available third detector and is also named in run_tracker.py's help.
DETECTORS = ("YOLOXx", "YOLO11x", "YOLO26x")

METHODS = {
    "method_1": [
        "--weights_path", "motion_model_weights/transformer-encoder-d256-ff512-6l-ft.pth",
        "--model_type", "transformer",
        "--use_oru",
        "--legacy_post_assignment_iou_gate",
        "--reupdate_type", "relative",
    ],
    "method_2": [
        "--no_motion",
        "--use_confidence_r",
        "--reupdate_type", "none",
        "--use_byte",
        "--use_oru",
        "--legacy_post_assignment_iou_gate",
    ],
    "method_3": [
        "--legacy_post_assignment_iou_gate",
        "--use_mahalanobis_cost",
        "--mahalanobis_cost_coefficient", "0.1",
        "--use_learned_association",
        "--association_weights_path", "checkpoints/association_multidataset_v2/best_model.pth",
        "--association_cost_weight", "1.0",
        "--suppress_duplicate_track_births",
        "--cleanup_duplicate_tracks",
        "--prioritize_mature_tracks",
        "--weights_path", "../motion-predictor/checkpoints/kf_adaptive_kalman_real_low_data_lightvv_transformer_newarch_kftrackoff/best_model.pth",
    ],
}

SUMMARY_FIELDS = [
    "method", "dataset", "split", "detector", "tracker_name", "exit_code",
    "aggregate_fps", "tracking_time_seconds", "total_frames",
    "HOTA", "ASSA", "DETA", "IDF1", "MOTA", "MOTP", "IDSW",
    "TP", "FP", "FN", "MT", "ML",
]

FPS_LINE = re.compile(
    r"^\[(?P<sequence>[^\]]+)\] completed in (?P<seconds>[0-9.]+)s \((?P<fps>[0-9.]+) FPS\)$"
)


def parse_metrics(path: Path) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        try:
            number = float(value)
        except ValueError:
            continue
        metrics[key] = int(number) if number.is_integer() else number
    return metrics


def parse_tracking_performance(log_path: Path, dataset: str) -> dict:
    """Read tracker-reported sequence timings and calculate an overall FPS."""
    per_sequence = []
    if not log_path.exists():
        return {"per_sequence": per_sequence}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FPS_LINE.match(line)
        if not match:
            continue
        item = {
            "sequence": match["sequence"],
            "tracking_time_seconds": float(match["seconds"]),
            "fps": float(match["fps"]),
        }
        seqinfo = Path("C:/Projects/.Datasets") / dataset / "val" / item["sequence"] / "seqinfo.ini"
        if seqinfo.exists():
            length_match = re.search(r"^seqLength=(\d+)$", seqinfo.read_text(encoding="utf-8"), re.MULTILINE)
            if length_match:
                item["frames"] = int(length_match.group(1))
        per_sequence.append(item)
    result = {"per_sequence": per_sequence}
    if per_sequence:
        result["tracking_time_seconds"] = round(sum(item["tracking_time_seconds"] for item in per_sequence), 6)
        total_frames = sum(item.get("frames", 0) for item in per_sequence)
        if total_frames:
            result["total_frames"] = total_frames
            result["aggregate_fps"] = round(total_frames / result["tracking_time_seconds"], 6)
    return result


def write_summary_files(runs: list[dict]) -> None:
    """Persist the summary in both machine-readable and thesis-table formats."""
    (RESULTS_ROOT / "benchmark_summary.json").write_text(
        json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8"
    )
    with (RESULTS_ROOT / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for run in runs:
            flattened = {**run, **run.get("performance", {}), **run.get("metrics", {})}
            writer.writerow({field: flattened.get(field, "") for field in SUMMARY_FIELDS})


def backfill_existing_records() -> list[dict]:
    """Add performance data to records written before the FPS schema existed."""
    summary_path = RESULTS_ROOT / "benchmark_summary.json"
    if not summary_path.exists():
        return []
    runs = json.loads(summary_path.read_text(encoding="utf-8")).get("runs", [])
    for run in runs:
        run["performance"] = parse_tracking_performance(
            RESULTS_ROOT / run["log"], run["dataset"]
        )
        run_path = RESULTS_ROOT / run["tracker_name"] / "run.json"
        run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    write_summary_files(runs)
    return runs


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runs = backfill_existing_records()
    completed_keys = {
        (run["method"], run["dataset"], run["detector"])
        for run in runs
        if run.get("exit_code") == 0 and run.get("metrics")
    }
    for method, method_args in METHODS.items():
        for dataset in DATASETS:
            for detector in DETECTORS:
                if (method, dataset, detector) in completed_keys:
                    print(f"skipping completed {method}_{dataset.lower()}_{detector.lower()}", flush=True)
                    continue
                tracker_name = f"{method}_{dataset.lower()}_{detector.lower()}"
                run_dir = RESULTS_ROOT / tracker_name
                run_dir.mkdir(exist_ok=True)
                command = [
                    sys.executable,
                    "run_tracker.py",
                    "--dataset", dataset,
                    "--split", "val",
                    "--detector_name", detector,
                    "--tracker_name", tracker_name,
                    *method_args,
                ]
                evaluation_command = [
                    sys.executable,
                    "-c",
                    (
                        "from evaluate import evaluate; "
                        f"evaluate({dataset!r}, 'val', trackers_to_eval=[{tracker_name!r}], "
                        "datasets_dir='C:/Projects/.Datasets')"
                    ),
                ]
                started_at = datetime.now(timezone.utc).isoformat()
                print(f"\n[{started_at}] starting {tracker_name}", flush=True)
                metric_source = ROOT / "results" / f"{tracker_name}-results.txt"
                csv_source = ROOT / "results" / "per_seq" / f"{tracker_name}.csv"
                expected_outputs = [
                    ROOT / "outputs" / tracker_name / f"{seq.name}.txt"
                    for seq in (Path("C:/Projects/.Datasets") / dataset / "val").iterdir()
                    if seq.is_dir()
                ]
                existing_outputs_complete = bool(expected_outputs) and all(path.exists() for path in expected_outputs)
                with (run_dir / "run.log").open("a", encoding="utf-8") as log:
                    if existing_outputs_complete:
                        log.write("Reusing complete tracker outputs from a prior evaluation-only failure.\n\n")
                        tracking_code = 0
                    else:
                        log.write("Tracking command:\n" + " ".join(command) + "\n\n")
                        tracking_code = subprocess.run(
                            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
                        ).returncode
                    evaluation_code = None
                    if tracking_code == 0:
                        log.write("\nEvaluation command:\n" + " ".join(evaluation_command) + "\n\n")
                        evaluation_code = subprocess.run(
                            evaluation_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
                        ).returncode
                exit_code = tracking_code if tracking_code != 0 else evaluation_code
                if metric_source.exists():
                    shutil.copy2(metric_source, run_dir / "metrics.txt")
                if csv_source.exists():
                    shutil.copy2(csv_source, run_dir / "per_sequence_metrics.csv")
                run = {
                    "method": method,
                    "dataset": dataset,
                    "split": "val",
                    "detector": detector,
                    "tracker_name": tracker_name,
                    "tracking_command": command,
                    "evaluation_command": evaluation_command,
                    "started_at_utc": started_at,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "exit_code": exit_code,
                    "metrics": parse_metrics(metric_source),
                    "performance": parse_tracking_performance(run_dir / "run.log", dataset),
                    "log": str((run_dir / "run.log").relative_to(RESULTS_ROOT)),
                }
                runs = [prior for prior in runs if (prior["method"], prior["dataset"], prior["detector"]) != (method, dataset, detector)]
                runs.append(run)
                (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
                write_summary_files(runs)
                print(f"finished {tracker_name}: exit code {exit_code}", flush=True)

    write_summary_files(runs)


if __name__ == "__main__":
    main()
