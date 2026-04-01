"""NuState — first-order difference of patch mean.

hist_state:
    nu[:, 0, :] = 0
    nu[:, i, :] = mu_hist[:, i, :] - mu_hist[:, i-1, :]

future_oracle_state:
    Same first-order difference applied to oracle_mu.

adapt_future_state:
    Identity.

path_kwargs_for("affine_residual"):
    affect_mu=True, affect_logsigma=False
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
        mu_hist: torch.Tensor,
        std_hist: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        nu = torch.zeros_like(mu_hist)
        nu[:, 1:, :] = mu_hist[:, 1:, :] - mu_hist[:, :-1, :]
        return nu

    def build_future_oracle_state(
        self,
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

    def path_kwargs_for(self, route_path: str) -> dict:
        if route_path == "affine_residual":
            return {"affect_mu": True, "affect_logsigma": False}
        raise ValueError(f"NuState has no path_kwargs for route_path='{route_path}'")
