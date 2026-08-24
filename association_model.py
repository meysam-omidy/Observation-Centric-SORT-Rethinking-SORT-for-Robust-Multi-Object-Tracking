"""Tiny no-ReID association scorer used as an optional OC-SORT cost residual.

The model is intentionally small and only sees geometry, motion and tracker
state.  It never creates candidates or relaxes an OC-SORT gate: its output is
used solely to rank pairs which are already geometrically valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


# Keep this public and versioned: candidate .npz files and checkpoints rely on
# the feature ordering below.
PAIR_FEATURE_NAMES = (
    "iou",
    "center_dx_over_diag",
    "center_dy_over_diag",
    "log_width_ratio",
    "log_height_ratio",
    "detector_score",
    "direction_cost",
    "track_age_norm",
    "track_observations_norm",
    "track_is_lost",
    "center_distance_over_diag",
    "detection_crowding_iou",
    "position_uncertainty_over_diag",
    "scale_uncertainty",
)
PAIR_FEATURE_DIM = len(PAIR_FEATURE_NAMES)
ASSOCIATION_MODEL_VERSION = 1


class AssociationMLP(nn.Module):
    """A deliberately cheap per-candidate scorer (about 3k parameters)."""

    def __init__(self, input_dim: int = PAIR_FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _pair_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise TLBR IoU, local copy to avoid an import cycle with utils."""
    left_top = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    right_bottom = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(right_bottom - left_top, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(boxes_a[:, 2] - boxes_a[:, 0], 0.0) * np.maximum(
        boxes_a[:, 3] - boxes_a[:, 1], 0.0
    )
    area_b = np.maximum(boxes_b[:, 2] - boxes_b[:, 0], 0.0) * np.maximum(
        boxes_b[:, 3] - boxes_b[:, 1], 0.0
    )
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-6)


def build_pair_features(
    tracks: Sequence,
    detections: np.ndarray,
    scores: np.ndarray,
    iou: np.ndarray,
    direction_cost: np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Construct finite, scale-normalised pair features for one association call."""
    detections = np.asarray(detections, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    num_tracks, num_detections = len(tracks), len(detections)
    out = np.zeros((num_tracks, num_detections, PAIR_FEATURE_DIM), dtype=np.float32)
    if not num_tracks or not num_detections:
        return out

    predicted = np.asarray([track.bbox.to_tlbr() for track in tracks], dtype=np.float32)
    pred_width = np.maximum(predicted[:, 2] - predicted[:, 0], 1.0)
    pred_height = np.maximum(predicted[:, 3] - predicted[:, 1], 1.0)
    det_width = np.maximum(detections[:, 2] - detections[:, 0], 1.0)
    det_height = np.maximum(detections[:, 3] - detections[:, 1], 1.0)
    pred_center = (predicted[:, :2] + predicted[:, 2:]) * 0.5
    det_center = (detections[:, :2] + detections[:, 2:]) * 0.5
    diagonal = np.sqrt(pred_width**2 + pred_height**2)[:, None]
    dx = (det_center[None, :, 0] - pred_center[:, None, 0]) / diagonal
    dy = (det_center[None, :, 1] - pred_center[:, None, 1]) / diagonal

    out[..., 0] = iou
    out[..., 1] = np.clip(dx, -8.0, 8.0)
    out[..., 2] = np.clip(dy, -8.0, 8.0)
    out[..., 3] = np.clip(np.log(det_width[None, :] / pred_width[:, None]), -4.0, 4.0)
    out[..., 4] = np.clip(np.log(det_height[None, :] / pred_height[:, None]), -4.0, 4.0)
    out[..., 5] = scores[None, :]
    out[..., 6] = np.asarray(direction_cost, dtype=np.float32)
    out[..., 7] = np.asarray(
        [min(float(track.age), 60.0) / 60.0 for track in tracks], dtype=np.float32
    )[:, None]
    out[..., 8] = np.asarray(
        [min(float(track.observation_count), 30.0) / 30.0 for track in tracks], dtype=np.float32
    )[:, None]
    out[..., 9] = np.asarray(
        [float(getattr(track.state, "name", "") == "Lost") for track in tracks], dtype=np.float32
    )[:, None]
    out[..., 10] = np.clip(np.sqrt(dx**2 + dy**2), 0.0, 8.0)

    det_iou = _pair_iou(detections, detections)
    np.fill_diagonal(det_iou, 0.0)
    out[..., 11] = det_iou.max(axis=1)[None, :]

    # SORT's first two state components are image centre coordinates.  The
    # exact covariance conventions are model-dependent, so use it only as a
    # bounded relative feature rather than a gate or raw distance.
    pos_uncertainty, scale_uncertainty = [], []
    for track, diag in zip(tracks, diagonal[:, 0]):
        covariance = getattr(getattr(track, "kf", None), "P", None)
        if covariance is None:
            pos_uncertainty.append(0.0)
            scale_uncertainty.append(0.0)
            continue
        covariance = np.asarray(covariance, dtype=float)
        pos_uncertainty.append(np.sqrt(max(float(covariance[0, 0]), 0.0) + max(float(covariance[1, 1]), 0.0)) / max(diag, 1.0))
        scale_uncertainty.append(np.log1p(np.sqrt(max(float(covariance[2, 2]), 0.0))))
    out[..., 12] = np.clip(np.asarray(pos_uncertainty, dtype=np.float32), 0.0, 8.0)[:, None]
    out[..., 13] = np.clip(np.asarray(scale_uncertainty, dtype=np.float32), 0.0, 12.0)[:, None]
    return np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0)


class AssociationScorerEngine:
    """Load a trained association checkpoint and score a batch of candidates."""

    def __init__(self, weights_path: str, device: str | None = None):
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"association checkpoint not found: {path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if int(checkpoint.get("association_model_version", -1)) != ASSOCIATION_MODEL_VERSION:
            raise ValueError("unsupported association checkpoint format")
        feature_dim = int(checkpoint.get("feature_dim", 0))
        if feature_dim != PAIR_FEATURE_DIM:
            raise ValueError(f"checkpoint feature dimension {feature_dim} != {PAIR_FEATURE_DIM}")
        self.model = AssociationMLP(feature_dim, int(checkpoint.get("hidden_dim", 64))).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.maximum(np.asarray(checkpoint["feature_std"], dtype=np.float32), 1e-6)

    @torch.inference_mode()
    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32).reshape(-1, PAIR_FEATURE_DIM)
        if not len(features):
            return np.empty((0,), dtype=np.float32)
        normalised = (features - self.feature_mean) / self.feature_std
        tensor = torch.from_numpy(normalised).to(self.device)
        return self.model(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
