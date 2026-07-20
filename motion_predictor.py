import os
from typing import Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field

from adaptive_kalman_motion import build_adaptive_kalman_model, exp_var
from motion_learned import ImprovedLSTMLearnedNoise, MotionTransformerLearnedNoise, softplus_var
from motion_lstm import ImprovedLSTMPredictor
from motion_transformer import MotionTransformer

ADAPTIVE_KALMAN_FEATURE_DIM = 15


class MotionPredictorConfig(BaseModel):
    enabled: bool = False
    model_type: Literal[
        'transformer', 'transformer_learned', 'lstm', 'lstm_learned', 'adaptive_kalman'
    ] = 'transformer_learned'
    weights_path: str = 'motion_model_weights/phase2_transformer_learned.pth'
    device: Optional[str] = Field(
        default=None,
        description='cuda | cpu | mps; None = auto (cuda if torch.cuda.is_available())',
    )
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_ff: int = 1024
    dropout: float = 0.1
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 2
    use_kalman: bool = True
    kalman_fusion_blend: float = 1.0
    max_gap_norm: Optional[float] = Field(
        default=None,
        description='adaptive_kalman only; None = read from checkpoint (falls back to 30.0)',
    )


def _resolve_device(cfg: MotionPredictorConfig) -> torch.device:
    if cfg.device is not None:
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_model(cfg: MotionPredictorConfig, device: torch.device) -> torch.nn.Module:
    t_kw = dict(
        input_dim=13,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
    )
    if cfg.model_type == 'transformer':
        return MotionTransformer(output_dim=5, **t_kw).to(device)
    if cfg.model_type == 'transformer_learned':
        return MotionTransformerLearnedNoise(**t_kw).to(device)
    if cfg.model_type == 'lstm':
        return ImprovedLSTMPredictor(
            input_dim=13,
            output_dim=5,
            d_model=cfg.d_model,
            hidden_dim=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.dropout,
        ).to(device)
    if cfg.model_type == 'lstm_learned':
        return ImprovedLSTMLearnedNoise(
            input_dim=13,
            d_model=cfg.d_model,
            hidden_dim=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.dropout,
        ).to(device)
    raise ValueError(cfg.model_type)


def _load_adaptive_kalman(
    cfg: MotionPredictorConfig, device: torch.device
) -> tuple[torch.nn.Module, str, float]:
    """
    Load a motion-predictor checkpoint (best_model.pth: model_state_dict + args) and
    rebuild the matching architecture — mirrors eval_adaptive_kalman.load_model_and_config
    so this project never has to duplicate the training hyperparameters by hand.
    """
    ckpt = torch.load(cfg.weights_path, map_location=device, weights_only=False)
    train_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    model_type = ckpt.get('model_type') or train_args.get('model_type', 'transformer')

    model_kw = dict(
        input_dim=ckpt.get('feature_dim', ADAPTIVE_KALMAN_FEATURE_DIM),
        d_model=train_args.get('d_model', 256),
        dropout=train_args.get('dropout', 0.1),
        conf_alpha=train_args.get('conf_alpha', 2.0),
    )
    if model_type == 'transformer':
        model_kw.update(
            nhead=train_args.get('nhead', 8),
            num_layers=train_args.get('num_layers', 6),
            dim_ff=train_args.get('dim_ff', 512),
        )
    else:
        model_kw.update(
            hidden_dim=train_args.get('lstm_hidden_dim', 256),
            num_layers=train_args.get('lstm_num_layers', 1),
        )

    model = build_adaptive_kalman_model(model_type, **model_kw).to(device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    max_gap_norm = (
        cfg.max_gap_norm if cfg.max_gap_norm is not None else train_args.get('max_gap_norm', 30.0)
    )
    return model, model_type, max_gap_norm


class MotionPredictorEngine:
    """Loads a phase-2 motion model and runs one-step autoregressive prediction."""

    def __init__(self, cfg: MotionPredictorConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg)
        if not os.path.isfile(cfg.weights_path):
            raise FileNotFoundError(
                f'Motion predictor weights not found: {cfg.weights_path}'
            )
        if cfg.model_type == 'adaptive_kalman':
            self.model, self.adaptive_model_type, self.max_gap_norm = _load_adaptive_kalman(
                cfg, self.device
            )
        else:
            self.model = build_model(cfg, self.device)
            self.model.load_weight(cfg.weights_path, map_location=str(self.device))
            self.model.eval()

    @torch.no_grad()
    def predict_batch(
        self,
        src: np.ndarray,
        valid_lens: list[int],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        :param src: (B, T, F) float32 motion features (F=13 legacy models, F=15 adaptive_kalman)
        :param valid_lens: valid timesteps per row (rest is padding)
        :return: pred (B, 5) xywh + conf in normalized space, or None (adaptive_kalman has
                 no bbox head — the tracker keeps its own CV/Kalman prediction); var_q, var_r
                 (B, 4) or None
        """
        t = torch.as_tensor(src, dtype=torch.float32, device=self.device)
        B, T, _ = t.shape

        mt = self.cfg.model_type
        if mt == 'adaptive_kalman':
            # ocsort.py left-pads this model type (data right-aligned) so the last row
            # of `t` is always the current step, matching predict_noise's expectation.
            pad = torch.zeros(B, T, dtype=torch.bool, device=self.device)
            for i, L in enumerate(valid_lens):
                if L < T:
                    pad[i, :T - L] = True
            if self.adaptive_model_type == 'transformer':
                log_q, log_r = self.model.predict_noise(t, src_key_padding_mask=pad)
            else:
                log_q, log_r = self.model.predict_noise(t)
            return None, exp_var(log_q), exp_var(log_r)

        pad = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        for i, L in enumerate(valid_lens):
            if L < T:
                pad[i, L:] = True
        idx = torch.tensor(valid_lens, device=self.device, dtype=torch.long) - 1
        trg0 = t[torch.arange(B, device=self.device), idx].unsqueeze(1)

        if mt == 'transformer':
            out = self.model.inference(
                t, trg0, num_steps=1, src_key_padding_mask=pad
            )
            return out[:, 0, :], None, None
        if mt == 'transformer_learned':
            pred, lq, lr = self.model.inference(
                t, trg0, num_steps=1, src_key_padding_mask=pad
            )
            vq = softplus_var(lq[:, 0, :])
            vr = softplus_var(lr[:, 0, :])
            return pred[:, 0, :], vq, vr
        if mt == 'lstm':
            out = self.model.inference(t, trg0, num_steps=1)
            return out[:, 0, :], None, None
        if mt == 'lstm_learned':
            pred, lq, lr = self.model.inference(t, trg0, num_steps=1)
            vq = softplus_var(lq[:, 0, :])
            vr = softplus_var(lr[:, 0, :])
            return pred[:, 0, :], vq, vr
        raise ValueError(mt)
