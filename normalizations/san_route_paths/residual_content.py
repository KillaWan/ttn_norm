"""ResidualContentPath — bounded additive content correction via dilated TCN.

Applies a small dilated-conv decoder to predict delta_y from (y_base, future_state_time).
A global learnable scalar gate (tanh-bounded, zero-initialised) caps the initial
correction to zero and constrains the maximum delta amplitude to tanh(delta_scale).

Design intent: non-stationary residual restoration, NOT a free forecast corrector.
The capacity is deliberately kept small (_HIDDEN=32) and the gate prevents the path
from acting as an unconstrained mini-forecaster.

Unified path interface:
    forward(y_norm, y_base, future_state_hat, future_state_time,
            mu_base_fut, std_base_fut, base_time_mean, base_time_std)
    -> y_route: (B, H, C)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualContentPath(nn.Module):
    """Gated additive residual correction: y_route = y_base + tanh(delta_scale) * delta_y.

    TCN architecture (all 1-D convolutions):
        1. Channel-mixing Conv1d(2C → hidden, kernel=1)  + GELU
        2. Local-context  Conv1d(hidden → hidden, kernel=3, dilation=1, pad=1) + GELU
        3. Wide-context   Conv1d(hidden → hidden, kernel=3, dilation=2, pad=2) + GELU
        4. Output         Conv1d(hidden → C, kernel=1) — zero-initialised

    delta_scale: global learnable scalar, initialised to 0.
        tanh(0) = 0 → delta contribution is zero at init (y_route == y_base).
        tanh saturates at ±1 → amplitude of delta_y is bounded to ±1 × delta_y.
    """

    _HIDDEN: int = 32

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

        # Global amplitude gate — zero-init ensures y_route == y_base at start.
        self.delta_scale = nn.Parameter(torch.zeros(1))

        # Diagnostic cache (not part of computation graph)
        self._diag_delta_l2_over_base_l2: float = 0.0
        self._diag_delta_mean_abs: float = 0.0
        self._diag_delta_std: float = 0.0
        self._diag_delta_lowfreq_ratio: float = 0.0

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

        y_route = y_base + torch.tanh(self.delta_scale) * delta_y

        with torch.no_grad():
            base_l2 = (y_base ** 2).mean().clamp_min(1e-8)
            self._diag_delta_l2_over_base_l2 = float(((delta_y ** 2).mean() / base_l2).item())
            self._diag_delta_mean_abs = float(delta_y.abs().mean().item())
            self._diag_delta_std = float(delta_y.std().item())
            # Low-freq ratio: DC energy (time-mean) vs total energy
            delta_dc = delta_y.mean(dim=1, keepdim=True)      # (B, 1, C)
            dc_energy = (delta_dc ** 2).mean()
            total_energy = (delta_y ** 2).mean().clamp_min(1e-8)
            self._diag_delta_lowfreq_ratio = float((dc_energy / total_energy).item())

        return y_route

    def get_route_diagnostics(self) -> dict:
        return {
            "delta_l2_over_base_l2": self._diag_delta_l2_over_base_l2,
            "delta_mean_abs": self._diag_delta_mean_abs,
            "delta_std": self._diag_delta_std,
            "delta_lowfreq_ratio": self._diag_delta_lowfreq_ratio,
        }
