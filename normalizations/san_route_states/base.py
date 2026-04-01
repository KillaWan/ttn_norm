"""Base interface for route state implementations.

Each state must implement:
    name                 — string identifier
    extract_hist_state   — build hist_state from historical window stats
    build_future_oracle_state — build oracle target for state loss
    adapt_future_state   — post-process raw predictor output (identity for most states)
    path_kwargs_for      — return path-specific kwargs dict for a named route path
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
        mu_hist: torch.Tensor,    # (B, P_hist, C)
        std_hist: torch.Tensor,   # (B, P_hist, C)
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute hist_state (B, P_hist, C) from historical window stats."""

    @abstractmethod
    def build_future_oracle_state(
        self,
        oracle_mu: torch.Tensor,  # (B, P_fut, C)
        oracle_std: torch.Tensor, # (B, P_fut, C)
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute oracle target for the state loss (B, P_fut, C)."""

    @abstractmethod
    def adapt_future_state(
        self,
        future_state_hat_raw: torch.Tensor,  # (B, P_fut, C)
    ) -> torch.Tensor:
        """Post-process raw predictor output into the state representation used by the path."""

    @abstractmethod
    def path_kwargs_for(self, route_path: str) -> dict:
        """Return kwargs that configure the named route path for this state.

        Example for 'affine_residual':
            {"affect_mu": True, "affect_logsigma": False}
        """
