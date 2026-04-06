"""SlopeStateCorrection — additive zero-mean slope correction path.

Taxonomy note
-------------
This path adds a zero-mean slope correction on top of the standard SAN
base denormalization.  It does NOT replace SAN's main mean/std path.

Role in the slope_state + slope_state_correction combination
------------------------------------------------------------
The denormalization is:

    y_base = base_time_mean + (base_time_std + eps) * y_norm   [standard SAN]
    y_out  = y_base + pred_slope_corr_fut                      [additive correction]

where pred_slope_corr_fut = _patch_slope_to_time(pred_kappa_fut) is zero-mean
by construction (kappa_i * tau, tau mean-zero).

This file exists only to satisfy the route_path builder interface (build_route_path)
and to carry diagnostics about the predicted vs oracle slope correction.
No gate, no amplitude scale, no mu logic.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SlopeStateCorrection(nn.Module):
    """Minimal slope correction path — no trainable parameters.

    Carries diagnostics only.  The actual denorm equation lives in SANRouteNorm.
    """

    def __init__(self, pred_len: int, enc_in: int, sigma_min: float = 1e-3, **kwargs):
        super().__init__()
        # No trainable parameters.
        self._diag_kappa_pred_mean: float = 0.0
        self._diag_corr_pred_mean: float = 0.0
        self._diag_kappa_abs_error: float = 0.0
        self._diag_corr_abs_error: float = 0.0

    def update_diagnostics(
        self,
        pred_kappa: torch.Tensor,            # (B, P_fut, C)
        pred_corr_time: torch.Tensor,        # (B, pred_len, C)
        oracle_kappa: torch.Tensor | None,   # (B, P_fut, C) or None
        oracle_corr_time: torch.Tensor | None,  # (B, pred_len, C) or None
    ) -> None:
        with torch.no_grad():
            self._diag_kappa_pred_mean = float(pred_kappa.mean().item())
            self._diag_corr_pred_mean = float(pred_corr_time.mean().item())
            if oracle_kappa is not None:
                self._diag_kappa_abs_error = float(
                    (pred_kappa - oracle_kappa).abs().mean().item()
                )
            if oracle_corr_time is not None:
                self._diag_corr_abs_error = float(
                    (pred_corr_time - oracle_corr_time).abs().mean().item()
                )

    def get_route_diagnostics(self) -> dict:
        return {
            "kappa_pred_mean": self._diag_kappa_pred_mean,
            "corr_pred_mean":  self._diag_corr_pred_mean,
            "kappa_abs_error": self._diag_kappa_abs_error,
            "corr_abs_error":  self._diag_corr_abs_error,
        }
