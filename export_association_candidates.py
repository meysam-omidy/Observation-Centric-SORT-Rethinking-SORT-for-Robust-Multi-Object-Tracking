"""Export causally labelled, no-ReID association candidates for training.

The tracker is run normally. Ground truth is used only after each frame to
label which already-offered candidate was the source track's correct detector
box on that frame. It never affects prediction, gating, assignment or births.
"""

from __future__ import annotations

import argparse
import configparser
import os
import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_tracking import AssociationAttribution, load_gt, oracle_gt_detection_matches
from association_model import PAIR_FEATURE_NAMES, build_pair_features
from ocsort import OCSORTTracker, OCSORTTrackerConfig


class CandidateCollector:
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        self.previous_track_to_gt: dict[int, int] = {}
        self.gt_to_detection: dict[int, int] = {}
        self.features: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.groups: list[np.ndarray] = []
        self.next_group_id = 0
        self.counts = Counter()

    def begin_frame(self, track_to_gt, gt_to_detection):
        self.previous_track_to_gt = dict(track_to_gt)
        self.gt_to_detection = dict(gt_to_detection)

    def __call__(self, *, tracks, detection_indices, valid_pairs, **event):
        if not len(tracks) or not len(detection_indices):
            return
        features = build_pair_features(
            tracks, event["detections"], event["scores"], event["iou"],
            event["direction_cost"], self.image_width, self.image_height,
        )
        detection_indices = np.asarray(detection_indices, dtype=int)
        valid_pairs = np.asarray(valid_pairs, dtype=bool)
        for row, track in enumerate(tracks):
            source_gt = self.previous_track_to_gt.get(track.id)
            expected_detection = self.gt_to_detection.get(source_gt)
            if expected_detection is None:
                self.counts["missing_gt_detection"] += 1
                continue
            cols = np.flatnonzero(valid_pairs[row])
            expected_cols = np.flatnonzero(detection_indices == expected_detection)
            if not len(expected_cols) or expected_cols[0] not in cols:
                self.counts["correct_pair_not_valid"] += 1
                continue
            # A listwise group needs a decision. Singleton groups carry no
            # association ranking signal and only inflate class imbalance.
            if len(cols) < 2:
                self.counts["singleton_groups"] += 1
                continue
            label = (cols == expected_cols[0]).astype(np.uint8)
            self.features.append(features[row, cols].astype(np.float32, copy=False))
            self.labels.append(label)
            self.groups.append(np.full(len(cols), self.next_group_id, dtype=np.int64))
            self.next_group_id += 1
            self.counts["groups"] += 1
            self.counts["candidates"] += len(cols)

    def save(self, output_path: str, metadata: dict):
        if not self.features:
            raise RuntimeError("no trainable candidate groups were exported")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=np.concatenate(self.features, axis=0),
            labels=np.concatenate(self.labels, axis=0),
            groups=np.concatenate(self.groups, axis=0),
            feature_names=np.asarray(PAIR_FEATURE_NAMES),
            metadata=np.asarray(str(metadata)),
        )
        return path


def detector_path(args, dataset: str, seq: str) -> str:
    parts = [args.detections_dir]
    if args.detector_name:
        parts.append(args.detector_name)
    parts.extend([dataset, f"{seq}.txt"])
    return os.path.join(*parts)


def tracker_config(args, width: int, height: int) -> dict:
    config = OCSORTTrackerConfig().model_dump()
    config.update({
        "image_width": width, "image_height": height,
        "max_age": args.max_age, "delta_t": args.delta_t,
        "high_score_det_threshold": args.high_score_det_threshold,
        "low_score_det_threshold": args.low_score_det_threshold,
        "init_track_score_threshold": args.init_track_score_threshold,
        "match_high_score_dets_with_confirmed_trks_threshold": args.match_high_threshold,
        "match_low_score_dets_with_confirmed_trks_threshold": args.match_low_threshold,
        "match_remained_high_score_dets_with_unconfirmed_trks_threshold": args.match_unconfirmed_threshold,
        "association_iou_coefficient": args.association_iou_coefficient,
        "association_speed_direction_coefficient": args.association_speed_direction_coefficient,
        "legacy_post_assignment_iou_gate": args.legacy_post_assignment_iou_gate,
        "use_byte": args.use_byte, "use_oru": args.use_oru,
        "use_confidence_r": args.use_confidence_r, "use_learned_q": args.use_learned_q,
        "q_scale": args.q_scale, "r_scale": args.r_scale,
        "collect_association_diagnostics": True,
        "motion": {
            "enabled": args.motion_enabled, "model_type": args.model_type,
            "weights_path": args.weights_path, "device": args.device,
            "use_kalman": True, "kalman_fusion_blend": args.kalman_fusion_blend,
            "max_gap_norm": args.max_gap_norm,
        },
    })
    return config


def split_sequences(sequences: list[str], heldout_fraction: float, seed: int, dataset: str):
    """Make a deterministic sequence-level train/held-out split.

    Frames from a sequence are correlated and share identities, therefore they
    must never be divided between the two partitions.
    """
    if len(sequences) < 2:
        raise ValueError(f"{dataset}: need at least two sequences for an 80/20 split")
    heldout_count = int(round(len(sequences) * heldout_fraction))
    heldout_count = min(max(heldout_count, 1), len(sequences) - 1)
    # crc32 makes the split stable across Python processes and independent of
    # the order in which dataset names were supplied on the command line.
    dataset_seed = (int(seed) + zlib.crc32(dataset.encode("utf-8"))) % (2**32)
    shuffled = np.asarray(sorted(sequences), dtype=object)
    np.random.default_rng(dataset_seed).shuffle(shuffled)
    heldout = sorted(shuffled[:heldout_count].tolist())
    train = sorted(shuffled[heldout_count:].tolist())
    return train, heldout


def export_sequences(args, dataset: str, sequences: list[str]):
    split_root = Path(args.datasets_dir) / dataset / args.split
    global_features, global_labels, global_groups = [], [], []
    counts = Counter()
    group_offset = 0
    for seq in sequences:
        seq_dir = split_root / seq
        ini = configparser.ConfigParser(); ini.read(seq_dir / "seqinfo.ini")
        section = ini["Sequence"]
        width, height, length = int(section["imWidth"]), int(section["imHeight"]), int(section["seqLength"])
        detection_path = detector_path(args, dataset, seq)
        if not os.path.isfile(detection_path):
            raise FileNotFoundError(f"{dataset}/{seq}: detection file not found: {detection_path}")
        dets = np.loadtxt(detection_path, delimiter=",")
        dets = np.atleast_2d(dets)
        gt = load_gt(str(seq_dir), dataset in {"MOT17", "MOT20"})
        attribution = AssociationAttribution(args.match_iou)
        collector = CandidateCollector(width, height)
        tracker = OCSORTTracker(tracker_config(args, width, height))
        print(f"[{dataset}/{seq}] exporting {length} frames")
        for frame in range(1, length + 1):
            frame_dets = dets[dets[:, 0] == frame][:, 1:]
            gt_to_det, _ = oracle_gt_detection_matches(gt.get(frame, []), frame_dets, args.match_iou)
            collector.begin_frame(attribution.track_to_gt, gt_to_det)
            tracker.update(frame_dets, association_observer=collector)
            attribution.consume_frame(frame, gt.get(frame, []), frame_dets,
                                      tracker.last_association_diagnostics, tracker)
        if collector.features:
            global_features.append(np.concatenate(collector.features))
            global_labels.append(np.concatenate(collector.labels))
            global_groups.append(np.concatenate(collector.groups) + group_offset)
            group_offset += collector.next_group_id
        counts.update(collector.counts)
    if not global_features:
        raise RuntimeError(f"{dataset}: no trainable candidate groups found")
    return {
        "features": np.concatenate(global_features).astype(np.float32, copy=False),
        "labels": np.concatenate(global_labels).astype(np.uint8, copy=False),
        "groups": np.concatenate(global_groups).astype(np.int64, copy=False),
        "group_count": group_offset,
        "counts": dict(sorted(counts.items())),
        "sequences": sequences,
    }


def save_partition(output_path: Path, partition: dict, metadata: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=partition["features"], labels=partition["labels"], groups=partition["groups"],
        feature_names=np.asarray(PAIR_FEATURE_NAMES), metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"saved {len(partition['features']):,} candidates in {partition['group_count']:,} groups to {output_path}")
    print("  collection counts:", partition["counts"])


def combine_partitions(partitions: list[dict]):
    features, labels, groups = [], [], []
    group_offset = 0
    for partition in partitions:
        features.append(partition["features"])
        labels.append(partition["labels"])
        groups.append(partition["groups"] + group_offset)
        group_offset += partition["group_count"]
    return {
        "features": np.concatenate(features), "labels": np.concatenate(labels),
        "groups": np.concatenate(groups), "group_count": group_offset,
        "counts": {}, "sequences": [],
    }


def main(args):
    datasets = args.datasets or ([args.dataset] if args.dataset else ["MOT17", "MOT20", "DanceTrack", "SportsMOT"])
    if args.dataset and args.datasets:
        raise ValueError("use either --dataset or --datasets, not both")
    if args.seqs and len(datasets) != 1:
        raise ValueError("--seqs is only supported with exactly one --dataset/--datasets value")
    if not 0 < args.heldout_fraction < 1:
        raise ValueError("--heldout_fraction must be between zero and one")
    output_dir = Path(args.output_dir)
    per_dataset = {"train": [], "heldout": []}
    split_manifest = {}
    for dataset in datasets:
        split_root = Path(args.datasets_dir) / dataset / args.split
        if not split_root.is_dir():
            if args.skip_missing_datasets:
                print(f"skipping {dataset}: missing split directory {split_root}")
                continue
            raise FileNotFoundError(f"{dataset}: split directory not found: {split_root}")
        sequences = args.seqs or sorted(path.name for path in split_root.iterdir() if path.is_dir())
        train_sequences, heldout_sequences = split_sequences(
            sequences, args.heldout_fraction, args.seed, dataset
        )
        split_manifest[dataset] = {"train": train_sequences, "heldout": heldout_sequences}
        print(f"\n{dataset}: {len(train_sequences)} train / {len(heldout_sequences)} held-out sequences")
        for name, partition_sequences in (("train", train_sequences), ("heldout", heldout_sequences)):
            partition = export_sequences(args, dataset, partition_sequences)
            per_dataset[name].append(partition)
            if args.output_path and len(datasets) == 1:
                legacy_path = Path(args.output_path)
                output_path = legacy_path.with_name(
                    f"{legacy_path.stem}_{name}{legacy_path.suffix or '.npz'}"
                )
            else:
                output_path = output_dir / f"{dataset.lower()}_{args.split}_{name}.npz"
            save_partition(output_path, partition, {
                "dataset": dataset, "split": args.split, "partition": name,
                "seed": args.seed, "heldout_fraction": args.heldout_fraction,
                "sequences": partition_sequences,
            })
    if not per_dataset["train"] or not per_dataset["heldout"]:
        raise RuntimeError("no dataset partitions were exported")
    manifest_path = output_dir / f"sequence_split_seed{args.seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "split": args.split, "seed": args.seed, "heldout_fraction": args.heldout_fraction,
        "datasets": split_manifest,
    }, indent=2), encoding="utf-8")
    print(f"wrote sequence split manifest to {manifest_path}")
    if args.combine:
        for name in ("train", "heldout"):
            combined = combine_partitions(per_dataset[name])
            save_partition(output_dir / f"{args.combined_prefix}_{name}.npz", combined, {
                "datasets": datasets, "split": args.split, "partition": name,
                "seed": args.seed, "heldout_fraction": args.heldout_fraction,
                "source_split_manifest": str(manifest_path),
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["MOT17", "MOT20", "DanceTrack", "SportsMOT"],
                        help="one dataset; use --datasets for several datasets")
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=["MOT17", "MOT20", "DanceTrack", "SportsMOT"],
                        help="datasets to export; default = all supported datasets")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--datasets_dir", default="C:/Projects/.Datasets")
    parser.add_argument("--detections_dir", default="C:/Projects/.Detections")
    parser.add_argument("--detector_name", default="YOLOXx")
    parser.add_argument("--output_dir", default="./association_data",
                        help="directory for per-dataset .npz files and split manifest")
    parser.add_argument("--output_path", default=None,
                        help="legacy output path, allowed only for a single dataset")
    parser.add_argument("--heldout_fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42,
                        help="stable seed used independently for each dataset")
    parser.add_argument("--combine", action="store_true", default=False,
                        help="also write all-dataset train and held-out .npz files")
    parser.add_argument("--combined_prefix", default="all_datasets")
    parser.add_argument("--skip_missing_datasets", action="store_true", default=False,
                        help="skip a selected dataset when its requested split is unavailable")
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--delta_t", type=int, default=3)
    parser.add_argument("--high_score_det_threshold", type=float, default=0.6)
    parser.add_argument("--low_score_det_threshold", type=float, default=0.1)
    parser.add_argument("--init_track_score_threshold", type=float, default=0.6)
    parser.add_argument("--match_high_threshold", type=float, default=0.2)
    parser.add_argument("--match_low_threshold", type=float, default=0.5)
    parser.add_argument("--match_unconfirmed_threshold", type=float, default=0.3)
    parser.add_argument("--association_iou_coefficient", type=float, default=1.0)
    parser.add_argument("--association_speed_direction_coefficient", type=float, default=0.3)
    parser.add_argument("--legacy_post_assignment_iou_gate", action="store_true", default=False,
                        help="match historical post-assignment IoU filtering when exporting candidates")
    parser.add_argument("--use_byte", action="store_true", default=True)
    parser.add_argument("--no_use_byte", action="store_false", dest="use_byte")
    parser.add_argument("--use_oru", action="store_true", default=False)
    parser.add_argument('--no_use_oru', action='store_false', dest='use_oru')
    parser.add_argument("--use_confidence_r", action="store_true", default=False)
    parser.add_argument("--no_use_confidence_r", action="store_false", dest="use_confidence_r")
    # Match run_tracker.py: learned Q is the normal baseline unless explicitly
    # disabled. Export and inference must use identical KF settings because P,
    # predicted boxes and the association features all depend on that choice.
    parser.add_argument("--use_learned_q", action="store_true", default=True)
    parser.add_argument("--no_use_learned_q", action="store_false", dest="use_learned_q")
    parser.add_argument("--q_scale", type=float, default=1.0)
    parser.add_argument("--r_scale", type=float, default=1.0)
    parser.add_argument("--motion_enabled", action="store_true", default=True)
    parser.add_argument("--no_motion", action="store_false", dest="motion_enabled")
    parser.add_argument("--model_type", default="adaptive_kalman", choices=["transformer", "transformer_learned", "lstm", "lstm_learned", "adaptive_kalman"])
    parser.add_argument("--weights_path", default="../motion-predictor/checkpoints/kf_adaptive_kalman_real_low_data_light_transformer_lr_new_kftrackoff2/best_model.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--kalman_fusion_blend", type=float, default=0.0)
    parser.add_argument("--max_gap_norm", type=float, default=30.0)
    main(parser.parse_args())
