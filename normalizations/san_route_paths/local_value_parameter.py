"""LocalValueParameterPath — shared path that corrects future raw stats via
additive residuals on both mu and log-std simultaneously.

Input:
    future_state_hat: (B, P_fut, C)
    mu_base_fut:      (B, P_fut, C)
    std_base_fut:     (B, P_fut, C)

Output:
    mu_route_fut:  (B, P_fut, C)
    std_route_fut: (B, P_fut, C)

Fixed formula (same regardless of which state is used):
    delta_mu       = delta_mu_head(future_state_hat)
    delta_logsigma = delta_logsigma_head(future_state_hat)
    mu_route       = mu_base_fut + delta_mu
    logstd_base    = log(clamp(std_base_fut, min=sigma_min))
    std_route      = exp(logstd_base + delta_logsigma).clamp(min=sigma_min)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LocalValueParameterPath(nn.Module):
    """Corrects future base stats (mu, std) via additive residuals predicted from
    a future state estimate.

    Both mu and log-std are always corrected; delta_mu and delta_logsigma heads
    are kept separate and zero-initialized so the path starts as identity.
    """

    def __init__(self, pred_stat_len: int, enc_in: int, sigma_min: float = 1e-3):
        super().__init__()
        self.pred_stat_len = pred_stat_len
        self.enc_in = enc_in
        self.sigma_min = sigma_min

        # Separate linear projections: (B, P_fut, C) -> (B, P_fut, C)
        self.delta_mu_head = nn.Linear(enc_in, enc_in)
        self.delta_logsigma_head = nn.Linear(enc_in, enc_in)

        nn.init.zeros_(self.delta_mu_head.weight)
        nn.init.zeros_(self.delta_mu_head.bias)
        nn.init.zeros_(self.delta_logsigma_head.weight)
        nn.init.zeros_(self.delta_logsigma_head.bias)

    def forward(
        self,
        future_state_hat: torch.Tensor,  # (B, P_fut, C)
        mu_base_fut: torch.Tensor,        # (B, P_fut, C)
        std_base_fut: torch.Tensor,       # (B, P_fut, C)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_mu = self.delta_mu_head(future_state_hat)
        delta_logsigma = self.delta_logsigma_head(future_state_hat)

        mu_route = mu_base_fut + delta_mu

        logstd_base = torch.log(std_base_fut.clamp(min=self.sigma_min))
        std_route = torch.exp(logstd_base + delta_logsigma).clamp(min=self.sigma_min)

        return mu_route, std_route
