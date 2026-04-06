"""SlopeState — patch-wise slope (kappa) state.

Role in the slope_state + slope_state_correction combination
------------------------------------------------------------
Implements patch-wise zero-mean slope extraction.  Each patch is modelled as:
    slope_curve(t) = kappa * tau(t)
where tau is mean-zero, centred on [-1, 1].  The slope correction is zero-mean
by construction, so SAN base statistics (mu + std) are computed on the raw input
and remain unmodified.

Physical state: (B, P, C) = kappa per patch.
SANRouteNorm special-case bypasses _normalize_patch_state() for this state
(exempt, like lp_state and omega_spec).

Framework contracts:
  - _normalize_patch_state() is NOT applied to slope_state.
  - adapt_future_state is identity.
  - Allowed combination: route_path="slope_state_correction", route_state="slope_state" only.
"""
from __future__ import annotations

import torch

from .base import RouteStateBase


def _make_centered_tau(window_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Mean-zero centred time coordinates over a patch.

    Returns a (W,) tensor with values evenly spaced on [-1, 1] then shifted so
    that tau.mean() == 0 exactly (for odd W the shift is already 0).

    Args:
        window_len: W, number of time steps per patch.
        device:     target device.
        dtype:      target dtype.

    Returns:
        tau: (W,)
    """
    if window_len <= 1:
        return torch.zeros(window_len, device=device, dtype=dtype)
    tau = torch.linspace(-1.0, 1.0, window_len, device=device, dtype=dtype)
    tau = tau - tau.mean()
    return tau


def _compute_patch_kappa(
    windows: torch.Tensor,   # (B, P, W, C)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute patch-wise LS slope via projection onto mean-zero tau.

    Returns:
        kappa       (B, P, C)   patch linear slope (LS fit of window onto tau)
        slope_wins  (B, P, W, C) kappa_i * tau — zero-mean by construction
    """
    B, P, W, C = windows.shape
    tau = _make_centered_tau(W, windows.device, windows.dtype)  # (W,)
    tau2_sum = (tau * tau).sum()                                 # scalar

    mu = windows.mean(dim=2)                                     # (B, P, C)
    dev = windows - mu.unsqueeze(2)                              # (B, P, W, C)
    kappa = (dev * tau.view(1, 1, W, 1)).sum(dim=2) / tau2_sum  # (B, P, C)

    slope_wins = kappa.unsqueeze(2) * tau.view(1, 1, W, 1)      # (B, P, W, C)
    return kappa, slope_wins


class SlopeState(RouteStateBase):
    """Patch-wise kappa state: slope coefficient per patch.

    Physical state: (B, P, C) = kappa.
    No trainable parameters.
    """

    def __init__(self, period_len: int = 12):
        self._period_len = period_len

    @property
    def name(self) -> str:
        return "slope_state"

    def extract_hist_state(
        self,
        x_hist: torch.Tensor,
        hist_windows: torch.Tensor,   # (B, P_hist, W, C)
        mu_hist: torch.Tensor,
        std_hist: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute (B, P_hist, C) kappa states from historical windows."""
        kappa, _ = _compute_patch_kappa(hist_windows)
        return kappa   # (B, P_hist, C)

    def build_future_oracle_state(
        self,
        y_true: torch.Tensor,
        fut_windows: torch.Tensor,    # (B, P_fut, W, C)
        oracle_mu: torch.Tensor,
        oracle_std: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute (B, P_fut, C) kappa states from future ground truth."""
        kappa, _ = _compute_patch_kappa(fut_windows)
        return kappa   # (B, P_fut, C)

    def adapt_future_state(
        self,
        future_state_hat_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Identity — slope_state supervises kappa directly."""
        return future_state_hat_raw
