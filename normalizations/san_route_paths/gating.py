"""GatingPath — state-conditioned 2-branch coefficient assignment over level and residual.

Taxonomy: factor allocation / coefficient assignment.
This path does NOT generate new content, does NOT perform MoE routing, does NOT
modify time coordinates, and does NOT alter the value-domain mapping.  It only
re-weights the two natural decomposition factors of y_base:
    mu_base = base_time_mean           (level component, large magnitude)
    r_base  = y_base - base_time_mean  (residual component, large magnitude)

A single gate head maps future_state_time → per-channel 2-branch softmax
coefficients for (mu_base, r_base).

Identity initialisation:  gate weights & biases are zero-initialised so
    softmax(0, 0) = (0.5, 0.5)  →  coefficients = 2 * (0.5, 0.5) = (1, 1)
    y_route = 1 * mu_base + 1 * r_base = y_base.

The constraint is w_mu + w_r = 2.0, so the gate controls the trade-off
between emphasising level vs residual per channel.

Under the non-overlap main protocol, future_state_time is the patch-wise
broadcasted state, so the resulting coefficients are patch-level constants.

Unified path interface:
    forward(y_norm, y_base, future_state_hat, future_state_time,
            mu_base_fut, std_base_fut, base_time_mean, base_time_std)
    -> y_route: (B, H, C)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingPath(nn.Module):
    """State-conditioned 2-branch gate over mu_base (level) and r_base (residual).

    Both factors have large magnitude, ensuring strong gradient signal to
    the gate head — unlike the previous 4-factor decomposition where high-
    frequency sub-components (mu_fast, r_osc) starved the gradients.
    """

    def __init__(self, pred_len: int, enc_in: int, sigma_min: float = 1e-3, **kwargs):
        super().__init__()

        # Single gate head for (mu_base, r_base): zero-init → identity at start
        self.gate_head = nn.Linear(enc_in, enc_in * 2)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

        # Diagnostic cache
        self._diag_gate_entropy_mean: float = 0.0
        self._diag_mu_weight_mean: float = 1.0
        self._diag_r_weight_mean: float = 1.0

    def forward(
        self,
        y_norm: torch.Tensor,
        y_base: torch.Tensor,             # (B, H, C)
        future_state_hat: torch.Tensor,
        future_state_time: torch.Tensor,   # (B, H, C)
        mu_base_fut: torch.Tensor,
        std_base_fut: torch.Tensor,
        base_time_mean: torch.Tensor,      # (B, H, C)
        base_time_std: torch.Tensor,
    ) -> torch.Tensor:
        B, H, C = y_base.shape
        mu_base = base_time_mean                               # (B, H, C)
        r_base = y_base - mu_base                              # (B, H, C)

        # --- State-conditioned 2-branch gate ---
        logits = self.gate_head(future_state_time).view(B, H, C, 2)
        weights = 2.0 * F.softmax(logits, dim=-1)             # (B, H, C, 2)

        y_route = weights[..., 0] * mu_base + weights[..., 1] * r_base  # (B, H, C)

        # --- Diagnostics (no gradient, logging only) ---
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            self._diag_gate_entropy_mean = float(ent.item())
            self._diag_mu_weight_mean = float(weights[..., 0].mean().item())
            self._diag_r_weight_mean = float(weights[..., 1].mean().item())

        return y_route

    def get_route_diagnostics(self) -> dict:
        return {
            "gate_entropy_mean": self._diag_gate_entropy_mean,
            "mu_weight_mean": self._diag_mu_weight_mean,
            "r_weight_mean": self._diag_r_weight_mean,
        }
