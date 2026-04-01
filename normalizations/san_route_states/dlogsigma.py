"""DLogSigmaState — first-order difference of patch log-std.

hist_state:
    logstd_hist = log(clamp(std_hist, min=sigma_min))
    dlogsigma[:, 0, :] = 0
    dlogsigma[:, i, :] = logstd_hist[:, i, :] - logstd_hist[:, i-1, :]

future_oracle_state:
    oracle_logstd = log(clamp(oracle_std, min=sigma_min))
    Same first-order difference applied to oracle_logstd.

adapt_future_state:
    Identity.

path_kwargs_for("affine_residual"):
    affect_mu=False, affect_logsigma=True
"""
from __future__ import annotations

import torch

from .base import RouteStateBase


class DLogSigmaState(RouteStateBase):

    @property
    def name(self) -> str:
        return "dlogsigma"

    def extract_hist_state(
        self,
        mu_hist: torch.Tensor,
        std_hist: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        logstd_hist = torch.log(std_hist.clamp(min=sigma_min))
        dlogsigma = torch.zeros_like(logstd_hist)
        dlogsigma[:, 1:, :] = logstd_hist[:, 1:, :] - logstd_hist[:, :-1, :]
        return dlogsigma

    def build_future_oracle_state(
        self,
        oracle_mu: torch.Tensor,
        oracle_std: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        oracle_logstd = torch.log(oracle_std.clamp(min=sigma_min))
        dlogsigma = torch.zeros_like(oracle_logstd)
        dlogsigma[:, 1:, :] = oracle_logstd[:, 1:, :] - oracle_logstd[:, :-1, :]
        return dlogsigma

    def adapt_future_state(
        self,
        future_state_hat_raw: torch.Tensor,
    ) -> torch.Tensor:
        return future_state_hat_raw

    def path_kwargs_for(self, route_path: str) -> dict:
        if route_path == "affine_residual":
            return {"affect_mu": False, "affect_logsigma": True}
        raise ValueError(f"DLogSigmaState has no path_kwargs for route_path='{route_path}'")
