"""NuState — first-order difference of patch mean.

hist_state:
    nu[:, 0, :] = 0
    nu[:, i, :] = mu_hist[:, i, :] - mu_hist[:, i-1, :]

future_oracle_state:
    Same first-order difference applied to oracle_mu.

adapt_future_state:
    Identity.
"""
from __future__ import annotations

import torch

from .base import RouteStateBase


class NuState(RouteStateBase):

    @property
    def name(self) -> str:
        return "nu"

    def extract_hist_state(
        self,
        x_hist: torch.Tensor,
        hist_windows: torch.Tensor,
        mu_hist: torch.Tensor,
        std_hist: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        nu = torch.zeros_like(mu_hist)
        nu[:, 1:, :] = mu_hist[:, 1:, :] - mu_hist[:, :-1, :]
        return nu

    def build_future_oracle_state(
        self,
        y_true: torch.Tensor,
        fut_windows: torch.Tensor,
        oracle_mu: torch.Tensor,
        oracle_std: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        nu = torch.zeros_like(oracle_mu)
        nu[:, 1:, :] = oracle_mu[:, 1:, :] - oracle_mu[:, :-1, :]
        return nu

    def adapt_future_state(
        self,
        future_state_hat_raw: torch.Tensor,
    ) -> torch.Tensor:
        return future_state_hat_raw
