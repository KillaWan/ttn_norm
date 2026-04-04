"""LPStateCorrection — Base-family structured mean-state denorm path.

Taxonomy note
-------------
This path belongs to the Base family of structured mean-state denorm methods.
It does NOT apply any gate, correction, or shift to the base SAN statistics.
It does NOT have trainable parameters.

Role in the lp_state + lp_state_correction combination
-------------------------------------------------------
The actual denormalization is performed entirely inside SANRouteNorm.denormalize():

    y_out = pred_lowpass_fut + mu_res_time + sigma_res_time * y_norm

where:
  - pred_lowpass_fut  : future low-pass sequence predicted by lp_state_predictor (B, T, C)
  - mu_res_time       : residual-domain future mean broadcast to time (B, T, C)
  - sigma_res_time    : residual-domain future std  broadcast to time (B, T, C)
  - y_norm            : backbone output in the residual-normalized domain

This file exists only to satisfy the route_path builder interface (build_route_path)
and to carry diagnostics about the predicted vs oracle low-pass sequence.
No gate, no amplitude scale, no correction logic.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LPStateCorrection(nn.Module):
    """Minimal structured mean-state path — no trainable parameters.

    Carries diagnostics only.  The actual denorm equation lives in SANRouteNorm.
    """

    def __init__(self, pred_len: int, enc_in: int, sigma_min: float = 1e-3, **kwargs):
        super().__init__()
        # No trainable parameters.
        self._diag_lp_pred_mean: float = 0.0
        self._diag_lp_oracle_mean: float = 0.0
        self._diag_lp_abs_error: float = 0.0

    def update_diagnostics(
        self,
        pred_lowpass_fut: torch.Tensor,           # (B, T, C)
        oracle_lowpass_fut: torch.Tensor | None,  # (B, T, C) or None
    ) -> None:
        with torch.no_grad():
            self._diag_lp_pred_mean = float(pred_lowpass_fut.mean().item())
            if oracle_lowpass_fut is not None:
                self._diag_lp_oracle_mean = float(oracle_lowpass_fut.mean().item())
                self._diag_lp_abs_error = float(
                    (pred_lowpass_fut - oracle_lowpass_fut).abs().mean().item()
                )

    def get_route_diagnostics(self) -> dict:
        return {
            "lp_pred_mean":   self._diag_lp_pred_mean,
            "lp_oracle_mean": self._diag_lp_oracle_mean,
            "lp_abs_error":   self._diag_lp_abs_error,
        }
