"""TimeAPNState — compatibility stub for the timeapn_state route state.

Taxonomy note
-------------
This file exists only to satisfy the route_state builder interface (build_route_state).
It has no trainable parameters and performs no computation.

Role in the timeapn_correction + timeapn_state combination
-----------------------------------------------------------
The actual TimeAPN state logic is implemented entirely inside SANRouteNorm:

    normalize():
        mu_hist_time  = sliding_mean(x)                  <- time-domain mean
        x_center      = x - mu_hist_time                 <- mean-centered series
        phi_hist      = angle(rfft(x_center, dim=time))  <- historical phase
        A_hist        = |rfft(x_center, dim=time)|        <- historical amplitude

These cached values feed into dedicated predictors (timeapn_mean_predictor,
timeapn_phase_predictor, timeapn_amp_predictor) which are attributes of
SANRouteNorm, not of this stub.

This file does NOT go through RouteStateBase's generic pipeline
(_normalize_patch_state, RouteStatePredictor, etc.).

This file is a retained compatibility stub.
"""
from __future__ import annotations

import torch

from .base import RouteStateBase


class TimeAPNState(RouteStateBase):
    """Minimal structured TimeAPN state stub.

    All actual state logic lives in SANRouteNorm.  This stub is registered in
    the builder to prevent KeyError; it is never called during normal operation
    because SANRouteNorm short-circuits to its own TimeAPN branch before
    reaching the generic route_state_impl path.
    """

    @property
    def name(self) -> str:
        return "timeapn_state"

    def extract_hist_state(
        self,
        x_hist: torch.Tensor,
        hist_windows: torch.Tensor,
        mu_hist: torch.Tensor,
        std_hist: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        # Stub: return mu_hist as a no-op identity so shape is valid (B, P_hist, C).
        return mu_hist

    def build_future_oracle_state(
        self,
        y_true: torch.Tensor,
        fut_windows: torch.Tensor,
        oracle_mu: torch.Tensor,
        oracle_std: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        # Stub: return oracle_mu as a no-op identity so shape is valid (B, P_fut, C).
        return oracle_mu

    def adapt_future_state(
        self,
        future_state_hat_raw: torch.Tensor,
    ) -> torch.Tensor:
        return future_state_hat_raw
