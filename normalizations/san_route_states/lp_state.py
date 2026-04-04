"""LPState — low-pass patch state.

Role in the lp_state + lp_state_correction combination
-------------------------------------------------------
The future low-pass sequence (pred_lowpass_fut) is the structured mean-state
used directly in the denorm equation:

    y_out = pred_lowpass_fut + mu_res_time + sigma_res_time * y_norm

This is a normalization-framework mechanism, NOT a second forecasting branch.

Lowpass helpers
---------------
_centered_moving_average(x, period_len) — applied to the full sequence:
  - period_len odd:  single pass, kernel_size = period_len.
  - period_len even: two passes — kernel_size=period_len then kernel_size=2
                     — giving a centered 2×period_len moving average.

The methods in this class (LPState) define oracle patch-level states for
diagnostics only.  The main output path does NOT depend on patch-level
compression of the low-pass sequence — pred_lowpass_fut enters denorm as
a full time-domain sequence.

Framework contracts:
  - _normalize_patch_state() is NOT applied to lp_state.
  - adapt_future_state is identity.
  - Allowed combination: route_path="lp_state_correction", route_state="lp_state" only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import RouteStateBase


def _fixed_lowpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Fixed moving-average lowpass filter along the time dimension.

    Args:
        x:           (B, T, C)
        kernel_size: averaging window length (must be >= 1).
                     When kernel_size <= 1, returns x unchanged.

    Returns:
        (B, T, C)  same shape as input, same device and dtype.
    """
    if kernel_size <= 1:
        return x
    B, T, C = x.shape
    # Permute to (B, C, T) for F.avg_pool1d which expects (N, C_in, L_in)
    x_t = x.permute(0, 2, 1)                        # (B, C, T)
    pad_left  = (kernel_size - 1) // 2
    pad_right = kernel_size - 1 - pad_left
    # replicate padding avoids edge-boundary artifacts
    x_padded = F.pad(x_t, (pad_left, pad_right), mode="replicate")
    lp = F.avg_pool1d(x_padded, kernel_size=kernel_size, stride=1, padding=0)
    return lp.permute(0, 2, 1)                      # (B, T, C)


def _centered_moving_average(x: torch.Tensor, period_len: int) -> torch.Tensor:
    """Centered moving average for sequence-level lowpass filtering.

    For odd period_len:  single pass of _fixed_lowpass with kernel_size=period_len.
    For even period_len: _fixed_lowpass(x, period_len) then _fixed_lowpass(result, 2),
    giving a centered 2×period_len moving average.

    Args:
        x:          (B, T, C)
        period_len: patch/period size (>= 1).

    Returns:
        (B, T, C)  same shape as input.
    """
    if period_len <= 1:
        return x
    out = _fixed_lowpass(x, period_len)
    if period_len % 2 == 0:
        out = _fixed_lowpass(out, 2)
    return out


class LPState(RouteStateBase):
    """Low-pass patch state: patch mean and std of the within-patch lowpass sequence.

    Physical states: (B, P, 2C) = cat([lp_patch_mean, lp_patch_std], dim=-1).
    """

    def __init__(self, period_len: int = 12):
        self._period_len = period_len

    @property
    def name(self) -> str:
        return "lp_state"

    def extract_hist_state(
        self,
        x_hist: torch.Tensor,
        hist_windows: torch.Tensor,   # (B, P_hist, W, C)
        mu_hist: torch.Tensor,
        std_hist: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute (B, P_hist, 2C) low-pass patch states.

        Applies the centered moving average within each patch independently, then
        computes per-patch mean and std.  No cross-patch leakage.
        """
        B, P, W, C = hist_windows.shape
        # Reshape to treat each (batch, patch) as an independent sequence
        wins = hist_windows.reshape(B * P, W, C)              # (B*P, W, C)
        wins_lp = _centered_moving_average(wins, self._period_len)  # (B*P, W, C)
        wins_lp = wins_lp.reshape(B, P, W, C)

        lp_mean = wins_lp.mean(dim=2)                         # (B, P, C)
        lp_std  = wins_lp.std(dim=2).clamp_min(sigma_min)    # (B, P, C)
        return torch.cat([lp_mean, lp_std], dim=-1)           # (B, P, 2C)

    def build_future_oracle_state(
        self,
        y_true: torch.Tensor,
        fut_windows: torch.Tensor,    # (B, P_fut, W, C)
        oracle_mu: torch.Tensor,
        oracle_std: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute (B, P_fut, 2C) low-pass patch states from future ground truth."""
        B, P, W, C = fut_windows.shape
        wins = fut_windows.reshape(B * P, W, C)
        wins_lp = _centered_moving_average(wins, self._period_len)
        wins_lp = wins_lp.reshape(B, P, W, C)

        lp_mean = wins_lp.mean(dim=2)
        lp_std  = wins_lp.std(dim=2).clamp_min(sigma_min)
        return torch.cat([lp_mean, lp_std], dim=-1)           # (B, P, 2C)

    def adapt_future_state(
        self,
        future_state_hat_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Identity — lp_state supervises the low-pass sequence directly."""
        return future_state_hat_raw
