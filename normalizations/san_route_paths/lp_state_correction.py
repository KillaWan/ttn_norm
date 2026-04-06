"""LPStateCorrection — compatibility stub for the lp_state_correction route path.

Taxonomy note
-------------
This file exists only to satisfy the route_path builder interface (build_route_path).
It has no trainable parameters and performs no computation.

Role in the lp_state + lp_state_correction combination
-------------------------------------------------------
The actual denormalization for the lp combo is performed entirely inside
SANRouteNorm.denormalize():

    lp_time_stats = _window_stats_to_time_stats(lp_mu_fut_hat, base_std_fut_hat, T)
    y_hat = mu_lp_time + (sigma_base_time + eps) * y_norm

where lp_mu_fut_hat is the output of lp_state_predictor, trained to match
oracle slow state defined by:
    mu_lp = patch_mean(lowpass_time(raw_series))

This file is a retained compatibility stub.  No gate, no gain, no mu correction,
no residual-domain stats, no amplitude scale live here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LPStateCorrection(nn.Module):
    """Minimal structured lowpass-state path — no trainable parameters.

    Carries diagnostics only.  The actual denorm equation lives in SANRouteNorm.
    """

    def __init__(self, pred_len: int, enc_in: int, sigma_min: float = 1e-3, **kwargs):
        super().__init__()
        # No trainable parameters.
        self._diag_low_raw_pred_mean: float = 0.0
        self._diag_mu_pred_mean: float = 0.0
        self._diag_mu_from_low_mean: float = 0.0
        self._diag_mu_corr_mean: float = 0.0
        self._diag_low_corr_pred_mean: float = 0.0
        self._diag_low_raw_abs_error: float = 0.0
        self._diag_mu_abs_error: float = 0.0
        self._diag_low_corr_abs_error: float = 0.0
        # Self-check: patch means of lifted correction should reproduce delta_mu exactly.
        # Should be ≈ 0 when mean-preserving lift is correct.
        self._diag_mu_match_after_lift_abs_error: float = 0.0

    def update_diagnostics(
        self,
        pred_lowpass_raw_fut: torch.Tensor,   # (B, pred_len, C)
        pred_mu_slow_fut: torch.Tensor,       # (B, P_fut, C)
        pred_mu_from_low_fut: torch.Tensor,   # (B, P_fut, C)
        pred_mu_corr_fut: torch.Tensor,       # (B, pred_len, C)
        pred_lowpass_corr_fut: torch.Tensor,  # (B, pred_len, C)
        oracle_lowpass_fut: torch.Tensor | None,   # (B, pred_len, C) or None
        oracle_mu_slow_fut: torch.Tensor | None,   # (B, P_fut, C) or None
        mu_corr_patch_mean: torch.Tensor | None,   # (B, P_fut, C) or None
        delta_mu: torch.Tensor | None,             # (B, P_fut, C) or None
    ) -> None:
        with torch.no_grad():
            self._diag_low_raw_pred_mean = float(pred_lowpass_raw_fut.mean().item())
            self._diag_mu_pred_mean = float(pred_mu_slow_fut.mean().item())
            self._diag_mu_from_low_mean = float(pred_mu_from_low_fut.mean().item())
            self._diag_mu_corr_mean = float(pred_mu_corr_fut.mean().item())
            self._diag_low_corr_pred_mean = float(pred_lowpass_corr_fut.mean().item())
            if oracle_lowpass_fut is not None:
                self._diag_low_raw_abs_error = float(
                    (pred_lowpass_raw_fut - oracle_lowpass_fut).abs().mean().item()
                )
                self._diag_low_corr_abs_error = float(
                    (pred_lowpass_corr_fut - oracle_lowpass_fut).abs().mean().item()
                )
            if oracle_mu_slow_fut is not None:
                self._diag_mu_abs_error = float(
                    (pred_mu_slow_fut - oracle_mu_slow_fut).abs().mean().item()
                )
            # Self-check: M(pred_mu_corr_fut) should equal delta_mu exactly.
            if mu_corr_patch_mean is not None and delta_mu is not None:
                self._diag_mu_match_after_lift_abs_error = float(
                    (mu_corr_patch_mean - delta_mu).abs().mean().item()
                )

    def get_route_diagnostics(self) -> dict:
        return {
            "low_raw_pred_mean":           self._diag_low_raw_pred_mean,
            "mu_pred_mean":                self._diag_mu_pred_mean,
            "mu_from_low_mean":            self._diag_mu_from_low_mean,
            "mu_corr_mean":                self._diag_mu_corr_mean,
            "low_corr_pred_mean":          self._diag_low_corr_pred_mean,
            "low_raw_abs_error":           self._diag_low_raw_abs_error,
            "mu_abs_error":                self._diag_mu_abs_error,
            "low_corr_abs_error":          self._diag_low_corr_abs_error,
            "mu_match_after_lift_abs_error": self._diag_mu_match_after_lift_abs_error,
        }
