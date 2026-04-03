"""Base interface for route state implementations.

Design contract
---------------
extract_hist_state and build_future_oracle_state define *raw physical quantities*
(e.g. first-order differences of patch mean, log-std differences).  They do NOT
apply any general-purpose scale normalisation — that is the responsibility of the
SANRouteNorm framework layer, which calls _normalize_patch_state() on the outputs.

adapt_future_state is responsible only for *domain constraints* (e.g. clamping,
applying a bijection to keep values in a valid domain).  It must NOT perform
general-purpose standardisation, which is handled upstream by SANRouteNorm.

Output shape contract (patch-wise):
    extract_hist_state        -> (B, P_hist, C)   raw physical state
    build_future_oracle_state -> (B, P_fut,  C)   raw oracle target (before normalisation)
    adapt_future_state        -> (B, P_fut,  C)   domain-constrained predictor output

All methods receive the complete set of available inputs so implementations can
draw from any of them (raw signal, windows, pre-computed stats).  Unused
parameters should simply be ignored.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class RouteStateBase(ABC):
    """Interface contract for route state implementations.

    Implementations define the raw physical quantity for a route state.
    SANRouteNorm applies _normalize_patch_state() after extraction so that
    the route_state_predictor always receives and predicts scale-normalised inputs.
    State files must not apply their own general standardisation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """String identifier for this state (e.g. 'nu', 'dlogsigma')."""

    @abstractmethod
    def extract_hist_state(
        self,
        x_hist: torch.Tensor,       # (B, T_hist, C)  raw historical input
        hist_windows: torch.Tensor, # (B, P_hist, window_len, C)  pre-extracted windows
        mu_hist: torch.Tensor,      # (B, P_hist, C)  per-window mean
        std_hist: torch.Tensor,     # (B, P_hist, C)  per-window std
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute raw hist_state (B, P_hist, C) from historical context.

        Returns the raw physical quantity only.  SANRouteNorm normalises the
        output with _normalize_patch_state before feeding it to route_state_predictor.
        """

    @abstractmethod
    def build_future_oracle_state(
        self,
        y_true: torch.Tensor,        # (B, T_fut, C)  raw future ground truth
        fut_windows: torch.Tensor,   # (B, P_fut, window_len, C)  pre-extracted windows
        oracle_mu: torch.Tensor,     # (B, P_fut, C)  per-window oracle mean
        oracle_std: torch.Tensor,    # (B, P_fut, C)  per-window oracle std
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute raw oracle target (B, P_fut, C) for the state prediction loss.

        Returns the raw physical quantity only.  SANRouteNorm normalises the
        output with _normalize_patch_state before computing the MSE loss against
        the (already normalised) route_state_predictor output.
        """

    @abstractmethod
    def adapt_future_state(
        self,
        future_state_hat_raw: torch.Tensor,  # (B, P_fut, C)
    ) -> torch.Tensor:
        """Apply domain constraints to raw predictor output.

        Responsible only for domain-validity (e.g. clamping to a legal range,
        applying a bijection).  General-purpose scale normalisation is handled
        by SANRouteNorm after this call.
        """
