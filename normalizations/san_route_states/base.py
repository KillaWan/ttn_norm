"""Base interface for route state implementations.

Each state must implement:
    name                      — string identifier
    extract_hist_state        — build hist_state from the full historical context
    build_future_oracle_state — build oracle target for the state prediction loss
    adapt_future_state        — post-process raw predictor output (identity for most states)

All methods receive the complete set of available inputs so that future states can
draw from any of them (raw signal, windows, or pre-computed stats).  Implementations
should simply ignore parameters they do not need.

Output shape contract (patch-wise):
    extract_hist_state        -> (B, P_hist, C)
    build_future_oracle_state -> (B, P_fut,  C)
    adapt_future_state        -> (B, P_fut,  C)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class RouteStateBase(ABC):
    """Interface contract for route state implementations."""

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
        """Compute hist_state (B, P_hist, C) from historical context."""

    @abstractmethod
    def build_future_oracle_state(
        self,
        y_true: torch.Tensor,        # (B, T_fut, C)  raw future ground truth
        fut_windows: torch.Tensor,   # (B, P_fut, window_len, C)  pre-extracted windows
        oracle_mu: torch.Tensor,     # (B, P_fut, C)  per-window oracle mean
        oracle_std: torch.Tensor,    # (B, P_fut, C)  per-window oracle std
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute oracle target for the state prediction loss (B, P_fut, C)."""

    @abstractmethod
    def adapt_future_state(
        self,
        future_state_hat_raw: torch.Tensor,  # (B, P_fut, C)
    ) -> torch.Tensor:
        """Post-process raw predictor output into the final state representation."""
