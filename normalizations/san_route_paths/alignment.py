"""AlignmentPath — monotonic temporal warping + per-channel linear interpolation.

A shared linear head maps future_state_time → positive per-channel increments v.
Cumulative sum of v produces a strictly monotonic time mapping tau, which is then
normalised to [0, H-1] and used to resample y_base via linear interpolation.

Identity initialisation: the head is zero-initialised, so v = softplus(0) + eps
is constant across all time steps and channels.  cumsum of a constant sequence
is linear, and after normalisation tau(t) = t, giving y_route == y_base.

Unified path interface:
    forward(y_norm, y_base, future_state_hat, future_state_time,
            mu_base_fut, std_base_fut, base_time_mean, base_time_std)
    -> y_route: (B, H, C)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentPath(nn.Module):
    """Monotonic time-warping of y_base driven by future_state_time.

    For each (batch, channel), a strictly monotonic mapping tau is learned:
        raw_v = head(future_state_time)   (B, H, C)
        v     = softplus(raw_v) + eps     (B, H, C)  — strictly positive
        tau   = cumsum(v, dim=1)          (B, H, C)  — monotonically increasing
        tau   = normalise tau to [0, H-1] per (batch, channel)

    y_base is then resampled at positions tau via linear interpolation,
    yielding a temporally warped forecast.
    """

    _EPS: float = 1e-4

    def __init__(self, pred_len: int, enc_in: int, sigma_min: float = 1e-3, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        # Head: (B, H, C) → (B, H, C) raw increments.  Zero-init → identity warp.
        self.head = nn.Linear(enc_in, enc_in)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Diagnostic cache (read via get_route_diagnostics, not part of loss)
        self._diag_mean_abs_tau_shift: float = 0.0
        self._diag_mean_abs_local_speed_minus_1: float = 0.0
        self._diag_mean_tau_curvature: float = 0.0

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
        B, H, C = y_base.shape

        raw_v = self.head(future_state_time)          # (B, H, C)
        v = F.softplus(raw_v) + self._EPS             # (B, H, C) strictly positive

        tau = torch.cumsum(v, dim=1)                  # (B, H, C) monotone

        # Normalise to [0, H-1] per (batch, channel)
        tau_min = tau[:, :1, :]                        # (B, 1, C)
        tau_max = tau[:, -1:, :]                       # (B, 1, C)
        tau_norm = (tau - tau_min) / (tau_max - tau_min + 1e-8) * (H - 1)
        tau_norm = tau_norm.clamp(0.0, float(H - 1))  # (B, H, C)

        # Per-channel linear interpolation of y_base at positions tau_norm
        tau_floor = tau_norm.long().clamp(0, H - 2)           # (B, H, C)
        tau_ceil  = (tau_floor + 1).clamp(0, H - 1)           # (B, H, C)
        frac      = (tau_norm - tau_floor.float()).clamp(0.0, 1.0)  # (B, H, C)

        # gather along time dimension (dim=1)
        y_floor = y_base.gather(1, tau_floor)                 # (B, H, C)
        y_ceil  = y_base.gather(1, tau_ceil)                  # (B, H, C)

        y_route = y_floor * (1.0 - frac) + y_ceil * frac

        with torch.no_grad():
            t_grid = torch.arange(H, device=y_base.device, dtype=y_base.dtype).view(1, H, 1)
            self._diag_mean_abs_tau_shift = float((tau_norm - t_grid).abs().mean().item())
            # Local speed = v / mean(v) per (B, C); deviation from 1 measures non-uniformity
            v_mean = v.mean(dim=1, keepdim=True).clamp_min(1e-8)
            self._diag_mean_abs_local_speed_minus_1 = float((v / v_mean - 1.0).abs().mean().item())
            # Curvature: mean absolute second difference of tau_norm
            increments = tau_norm[:, 1:, :] - tau_norm[:, :-1, :]  # (B, H-1, C)
            if increments.shape[1] >= 2:
                curvature = (increments[:, 1:, :] - increments[:, :-1, :]).abs().mean()
                self._diag_mean_tau_curvature = float(curvature.item())

        return y_route

    def get_route_diagnostics(self) -> dict:
        return {
            "mean_abs_tau_shift": self._diag_mean_abs_tau_shift,
            "mean_abs_local_speed_minus_1": self._diag_mean_abs_local_speed_minus_1,
            "mean_tau_curvature": self._diag_mean_tau_curvature,
        }
