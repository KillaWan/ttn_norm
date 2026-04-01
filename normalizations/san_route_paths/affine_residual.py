"""AffineResidualPath — shared path that corrects future raw stats via affine residuals.

Input:
    future_state_hat: (B, P_fut, C)
    mu_base_fut:      (B, P_fut, C)
    std_base_fut:     (B, P_fut, C)
    affect_mu:        bool
    affect_logsigma:  bool

Output:
    mu_route_fut:  (B, P_fut, C)
    std_route_fut: (B, P_fut, C)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AffineResidualPath(nn.Module):
    """Corrects future base stats (mu, std) via additive residuals predicted from
    a future state estimate.

    delta_mu and delta_logsigma heads are kept separate.
    """

    def __init__(self, pred_stat_len: int, enc_in: int, sigma_min: float = 1e-3):
        super().__init__()
        self.pred_stat_len = pred_stat_len
        self.enc_in = enc_in
        self.sigma_min = sigma_min

        # Separate linear projections: (B, P_fut, C) -> (B, P_fut, C)
        # Applied per-channel independently (weight shared across time via Linear on C).
        self.delta_mu_head = nn.Linear(enc_in, enc_in)
        self.delta_logsigma_head = nn.Linear(enc_in, enc_in)

        nn.init.zeros_(self.delta_mu_head.weight)
        nn.init.zeros_(self.delta_mu_head.bias)
        nn.init.zeros_(self.delta_logsigma_head.weight)
        nn.init.zeros_(self.delta_logsigma_head.bias)

    def forward(
        self,
        future_state_hat: torch.Tensor,   # (B, P_fut, C)
        mu_base_fut: torch.Tensor,         # (B, P_fut, C)
        std_base_fut: torch.Tensor,        # (B, P_fut, C)
        affect_mu: bool = True,
        affect_logsigma: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_mu = self.delta_mu_head(future_state_hat)           # (B, P_fut, C)
        delta_logsigma = self.delta_logsigma_head(future_state_hat)  # (B, P_fut, C)

        if affect_mu:
            mu_route = mu_base_fut + delta_mu
        else:
            mu_route = mu_base_fut

        logstd_base = torch.log(std_base_fut.clamp(min=self.sigma_min))
        if affect_logsigma:
            std_route = torch.exp(logstd_base + delta_logsigma).clamp(min=self.sigma_min)
        else:
            std_route = std_base_fut

        return mu_route, std_route
