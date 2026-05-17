import os
from typing import Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field

from motion_learned import ImprovedLSTMLearnedNoise, MotionTransformerLearnedNoise, softplus_var
from motion_lstm import ImprovedLSTMPredictor
from motion_transformer import MotionTransformer


class MotionPredictorConfig(BaseModel):
    enabled: bool = False
    model_type: Literal['transformer', 'transformer_learned', 'lstm', 'lstm_learned'] = (
        'transformer_learned'
    )
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


class MotionPredictorEngine:
    """Loads a phase-2 motion model and runs one-step autoregressive prediction."""

    def __init__(self, cfg: MotionPredictorConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg)
        if not os.path.isfile(cfg.weights_path):
            raise FileNotFoundError(
                f'Motion predictor weights not found: {cfg.weights_path}'
            )
        self.model = build_model(cfg, self.device)
        self.model.load_weight(cfg.weights_path, map_location=str(self.device))
        self.model.eval()

    @torch.no_grad()
    def predict_batch(
        self,
        src: np.ndarray,
        valid_lens: list[int],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        :param src: (B, T, 13) float32, normalized xywh motion features + detection score
        :param valid_lens: valid timesteps per row (rest is padding)
        :return: pred (B, 5) xywh + conf in normalized space; var_q, var_r (B, 4) or None
        """
        t = torch.as_tensor(src, dtype=torch.float32, device=self.device)
        B, T, _ = t.shape
        pad = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        for i, L in enumerate(valid_lens):
            if L < T:
                pad[i, L:] = True
        idx = torch.tensor(valid_lens, device=self.device, dtype=torch.long) - 1
        trg0 = t[torch.arange(B, device=self.device), idx].unsqueeze(1)

        mt = self.cfg.model_type
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
