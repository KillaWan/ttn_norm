"""ResidualContentPath — structured additive content correction via dilated TCN.

Concatenates y_base and future_state_time along the channel axis, then passes
through a small dilated-conv decoder to predict delta_y.  The final conv layer
is zero-initialised so delta_y == 0 at the start of training.

Unified path interface:
    forward(y_norm, y_base, future_state_hat, future_state_time,
            mu_base_fut, std_base_fut, base_time_mean, base_time_std)
    -> y_route: (B, H, C)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualContentPath(nn.Module):
    """Additive residual correction: y_route = y_base + delta_y.

    TCN architecture (all 1-D convolutions, causal-style padding):
        1. Channel-mixing Conv1d(2C → hidden, kernel=1)  + GELU
        2. Local-context  Conv1d(hidden → hidden, kernel=3, dilation=1, pad=1) + GELU
        3. Wide-context   Conv1d(hidden → hidden, kernel=3, dilation=2, pad=2) + GELU
        4. Output         Conv1d(hidden → C, kernel=1) — zero-initialised

    Zero-init on the output layer guarantees delta_y = 0 at initialisation,
    so y_route == y_base at the start of training.
    """

    _HIDDEN: int = 64

    def __init__(self, pred_len: int, enc_in: int, sigma_min: float = 1e-3, **kwargs):
        super().__init__()
        hidden = self._HIDDEN
        self.tcn = nn.Sequential(
            nn.Conv1d(2 * enc_in, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=1, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, enc_in, kernel_size=1),  # zero-init below
        )
        nn.init.zeros_(self.tcn[-1].weight)
        nn.init.zeros_(self.tcn[-1].bias)

    def forward(
        self,
        y_norm: torch.Tensor,
        y_base: torch.Tensor,             # (B, H, C)
        future_state_hat: torch.Tensor,
        future_state_time: torch.Tensor,  # (B, H, C)
        mu_base_fut: torch.Tensor,
        std_base_fut: torch.Tensor,
        base_time_mean: torch.Tensor,
        base_time_std: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate along feature dim → (B, H, 2C), then permute for Conv1d
        feat = torch.cat([y_base, future_state_time], dim=-1)  # (B, H, 2C)
        feat = feat.permute(0, 2, 1)                           # (B, 2C, H)
        delta_y = self.tcn(feat).permute(0, 2, 1)             # (B, H, C)
        return y_base + delta_y
