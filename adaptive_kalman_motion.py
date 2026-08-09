"""
Context-aware adaptive Kalman Q/R predictor — inference-only port.

Architecture mirrors ../motion-predictor/adaptive_kalman_motion.py, where the model is
trained (train_adaptive_kalman.py) and evaluated (eval_adaptive_kalman.py). No bbox
regression head: the tracker keeps its own constant-velocity Kalman prediction and only
asks this model for per-step Q/R (log-variance) given the observation history.

Feature layout (15-D, see utils.compute_adaptive_kalman_features):
  [x, y, w, h, vx..vh, ax..ah, det_score, frames_since_obs, is_observed]
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VAR_FLOOR = 1e-8


def exp_var(log_v: torch.Tensor, floor: float = VAR_FLOOR) -> torch.Tensor:
    """Recover a positive variance from a log-variance (non-saturating)."""
    return torch.exp(log_v.clamp(min=-20.0, max=10.0)).clamp(min=floor)


def confidence_log_r_prior(
    score: torch.Tensor,
    alpha: float = 2.0,
    base_log_var: float = -9.0,
) -> torch.Tensor:
    """Log-variance prior for normalized xywh; low score -> larger R (additive in log-space)."""
    s = score.clamp(0.0, 1.0)
    return base_log_var + alpha * (1.0 - s)

def nfc(n_layers, input_dim, output_dim, dropout):
    components = []
    dims = torch.linspace(input_dim, output_dim, n_layers + 1)
    dims = [int(x) for x in dims]
    for i in range(len(dims) - 2):
        components.extend([
            nn.Linear(dims[i], dims[i+1]),
            nn.LayerNorm(dims[i+1]),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        ])
    components.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*components)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class _AdaptiveKalmanHead(nn.Module):
    """Shared Q/R output heads with confidence-R prior on R."""

    def __init__(self, hidden_dim: int, dropout: float, conf_alpha: float = 2.0, q_init_bias: float = -12.0):
        super().__init__()
        self.conf_alpha = conf_alpha
        # Q head output is interpreted directly as log(var_q) by the loss.
        # Bias = -12 → exp(-12) ≈ 6e-6 at init: a mid-scale start that sits
        # between low-motion (MOT17 ~6e-7) and high-motion (DanceTrack ~6e-5)
        # process noise, so the head does not have to travel far in either
        # direction and low-motion data is not stuck high early in training.
        
        self.q_head = nfc(
            n_layers=3,
            input_dim=hidden_dim + 1,
            output_dim=4,
            dropout=dropout * 0.5
        )
        self.r_residual_head = nfc(
            n_layers=3,
            input_dim=hidden_dim,
            output_dim=4,
            dropout=dropout * 0.5
        )
        # self.q_head = nn.Linear(hidden_dim, 4)
        # nn.init.constant_(self.q_head.bias, q_init_bias)
        # nn.init.xavier_uniform_(self.q_head.weight, gain=0.1)
        # Additive log-space delta on the confidence prior. Zero init → start at
        # prior; signed delta lets R go above or below the prior (unlike the old
        # prior_var + exp(logit) floor which froze loss_r when prior was too high).
        # self.r_residual_head = nn.Linear(hidden_dim, 4)
        # nn.init.zeros_(self.r_residual_head.weight)
        # nn.init.zeros_(self.r_residual_head.bias)

    def predict_q(self, hidden: torch.Tensor, prediction_gap: torch.Tensor) -> torch.Tensor:
        return self.q_head(torch.cat([hidden, prediction_gap], dim=-1))

    def predict_r(self, hidden: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        log_r_prior = confidence_log_r_prior(scores, alpha=self.conf_alpha)
        delta = self.r_residual_head(hidden).clamp(-6.0, 6.0)
        return (log_r_prior + delta).clamp(-16.0, 8.0)



class AdaptiveKalmanTransformer(nn.Module):
    """
    Transformer encoder over [history | future-context] → per-step log_var_q, log_var_r.
    No bbox output.
    """

    def __init__(
        self,
        input_dim: int = 15,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        conf_alpha: float = 2.0,
        max_gap_norm: float = 30.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.conf_alpha = conf_alpha
        self.max_gap_norm = float(max_gap_norm)

        self.in_fc = nfc(
            n_layers=3,
            input_dim=input_dim,
            output_dim=d_model,
            dropout=dropout * 0.5
        )
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                batch_first=True,
                dropout=dropout,
                activation="gelu",
            ),
            mask_check=False,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.head = _AdaptiveKalmanHead(d_model, dropout, conf_alpha=conf_alpha)

    @staticmethod
    def _causal_mask(src_len: int, ctx_len: int, device: torch.device) -> torch.Tensor:
        total = src_len + ctx_len
        mask = torch.triu(torch.ones(total, total, device=device), diagonal=1)
        mask[:, :src_len] = 0
        return mask.bool()

    def _encode(
        self,
        src: torch.Tensor,
        ctx: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, src_len, _ = src.shape
        x = torch.cat([src, ctx], dim=1)
        emb = self.pos_enc(self.in_fc(x) * math.sqrt(self.d_model))
        mask = self._causal_mask(src_len, ctx.size(1), x.device)
        pad = None
        if src_key_padding_mask is not None:
            pad = torch.zeros(
                b, src_len + ctx.size(1), dtype=torch.bool, device=x.device
            )
            pad[:, :src_len] = src_key_padding_mask
        if pad is None:
            out = self.transformer(emb, mask=mask)
        else:
            out = self.transformer(emb, mask=mask, src_key_padding_mask=pad)
        return out[:, -ctx.size(1):, :]

    @torch.no_grad()
    def predict_q(
        self,
        src: torch.Tensor,
        prediction_gap: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single-step Q/R from observation history. ``src``'s last row must be the
        current (most recent) step — pad any shorter sequences at the front.
        """
        ctx = src[:, -1:, :].clone()
        hidden = self._encode(src, ctx, src_key_padding_mask)
        gap = prediction_gap.reshape(src.size(0), 1, 1).to(src)
        return self.head.predict_q(hidden[:, -1:, :], gap)[:, 0, :]

    @torch.no_grad()
    def predict_r(
        self,
        src: torch.Tensor,
        measurement: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self._encode(src, measurement, src_key_padding_mask)
        return self.head.predict_r(
            hidden[:, -1:, :], measurement[..., 12:13]
        )[:, 0, :]

    def save_weight(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weight(self, path: str, map_location: Optional[str] = None) -> None:
        loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=loc, weights_only=True))


class AdaptiveKalmanLSTM(nn.Module):
    """LSTM encoder over history + current step -> Q/R only. No padding-mask support."""

    def __init__(
        self,
        input_dim: int = 15,
        d_model: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        conf_alpha: float = 2.0,
        teacher_forcing_ratio: float = 0.5,
        max_gap_norm: float = 30.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_gap_norm = float(max_gap_norm)

        self.in_fc = nn.Sequential(
            nn.Linear(input_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
        )
        self.lstm = nn.LSTM(
            d_model,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _AdaptiveKalmanHead(
            hidden_dim, dropout, conf_alpha=conf_alpha
        )

    @torch.no_grad()
    def predict_q(self, src: torch.Tensor, prediction_gap: torch.Tensor) -> torch.Tensor:
        """Single-step Q/R; ``src``'s last row must be the current step (no padding mask)."""
        src_embed = self.in_fc(src)
        _, (h, c) = self.lstm(src_embed)
        prev = src[:, -1:, :]
        out, (h, c) = self.lstm(self.in_fc(prev), (h, c))
        gap = prediction_gap.reshape(src.size(0), 1, 1).to(src)
        return self.head.predict_q(out, gap)[:, 0, :]

    @torch.no_grad()
    def predict_r(self, src: torch.Tensor, measurement: torch.Tensor) -> torch.Tensor:
        src_embed = self.in_fc(src)
        _, (h, c) = self.lstm(src_embed)
        out, (h, c) = self.lstm(self.in_fc(measurement), (h, c))
        return self.head.predict_r(out, measurement[..., 12:13])[:, 0, :]

    def save_weight(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weight(self, path: str, map_location: Optional[str] = None) -> None:
        loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=loc, weights_only=True))


def build_adaptive_kalman_model(
    model_type: str = "transformer",
    input_dim: int = 15,
    **kwargs,
) -> nn.Module:
    if model_type == "transformer":
        return AdaptiveKalmanTransformer(input_dim=input_dim, **kwargs)
    if model_type == "lstm":
        return AdaptiveKalmanLSTM(input_dim=input_dim, **kwargs)
    raise ValueError(f"Unknown model_type: {model_type}")
