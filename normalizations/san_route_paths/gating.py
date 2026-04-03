"""GatingPath — state-conditioned coefficient assignment over existing restoration factors.

Taxonomy: factor allocation / coefficient assignment.
This path does NOT generate new content, does NOT perform MoE routing, does NOT
modify time coordinates, and does NOT alter the value-domain mapping.  It only
re-weights the fixed decomposition factors already present in y_base.

Decomposition of y_base into four fixed factors:
    mu_slow  = low-pass(base_time_mean)        slow level trend
    mu_fast  = base_time_mean - mu_slow         fast level residual
    r_smooth = low-pass(y_base - base_time_mean)  smooth residual
    r_osc    = (y_base - base_time_mean) - r_smooth  oscillatory residual

Low-pass filters are non-learnable (replicate-padded avg_pool1d).

Two small gate heads map future_state_time → per-channel 2-branch softmax
coefficients for (mu_slow, mu_fast) and (r_smooth, r_osc) respectively.

Identity initialisation:  gate weights & biases are zero-initialised so
    softmax(0,0) = (0.5, 0.5)  →  coefficients = 2 * (0.5, 0.5) = (1, 1)
    y_route = 1*mu_slow + 1*mu_fast + 1*r_smooth + 1*r_osc = y_base.

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

# Fixed smoothing kernel sizes (odd, non-learnable)
MU_SMOOTH_K = 5
R_SMOOTH_K = 5


def _fixed_smooth(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Non-learnable low-pass filter via replicate-padded avg_pool1d.

    Args:
        x: (B, H, C)
        kernel_size: odd integer.
    Returns:
        smoothed (B, H, C).
    """
    # avg_pool1d expects (B, C, H)
    pad = kernel_size // 2
    xt = x.permute(0, 2, 1)                          # (B, C, H)
    xt = F.pad(xt, (pad, pad), mode="replicate")      # (B, C, H + 2*pad)
    xt = F.avg_pool1d(xt, kernel_size=kernel_size, stride=1)  # (B, C, H)
    return xt.permute(0, 2, 1)                        # (B, H, C)


class GatingPath(nn.Module):
    """State-conditioned factor coefficient assignment over existing y_base factors.

    This path decomposes y_base into four fixed (non-learnable) factors and
    assigns state-conditioned coefficients to each via two 2-branch softmax gates.
    No new content is generated — the output is strictly a re-weighting of
    components already present in y_base.
    """

    def __init__(self, pred_len: int, enc_in: int, sigma_min: float = 1e-3, **kwargs):
        super().__init__()

        # Gate head for (mu_slow, mu_fast): zero-init → identity at start
        self.mu_gate_head = nn.Linear(enc_in, enc_in * 2)
        nn.init.zeros_(self.mu_gate_head.weight)
        nn.init.zeros_(self.mu_gate_head.bias)

        # Gate head for (r_smooth, r_osc): zero-init → identity at start
        self.r_gate_head = nn.Linear(enc_in, enc_in * 2)
        nn.init.zeros_(self.r_gate_head.weight)
        nn.init.zeros_(self.r_gate_head.bias)

        # Diagnostic cache (read via get_route_diagnostics, not part of loss)
        self._diag_mu_gate_entropy_mean: float = 0.0
        self._diag_r_gate_entropy_mean: float = 0.0
        self._diag_mu_branch_usage_mean: list = [1.0, 1.0]
        self._diag_r_branch_usage_mean: list = [1.0, 1.0]
        self._diag_mu_fast_weight_mean: float = 1.0
        self._diag_r_osc_weight_mean: float = 1.0

    def forward(
        self,
        y_norm: torch.Tensor,
        y_base: torch.Tensor,             # (B, H, C)
        future_state_hat: torch.Tensor,
        future_state_time: torch.Tensor,  # (B, H, C)
        mu_base_fut: torch.Tensor,
        std_base_fut: torch.Tensor,
        base_time_mean: torch.Tensor,     # (B, H, C)
        base_time_std: torch.Tensor,
    ) -> torch.Tensor:
        B, H, C = y_base.shape
        mu_base_time = base_time_mean                         # (B, H, C)
        r_base = y_base - mu_base_time                        # (B, H, C)

        # --- Fixed low-pass decomposition of mu and r ---
        mu_slow = _fixed_smooth(mu_base_time, MU_SMOOTH_K)    # (B, H, C)
        mu_fast = mu_base_time - mu_slow                      # (B, H, C)

        r_smooth = _fixed_smooth(r_base, R_SMOOTH_K)          # (B, H, C)
        r_osc = r_base - r_smooth                             # (B, H, C)

        # --- State-conditioned coefficient gates ---
        # Under non-overlap protocol, future_state_time is patch-wise broadcasted
        # state, so the resulting coefficients are patch-level constants.
        mu_logits = self.mu_gate_head(future_state_time).view(B, H, C, 2)
        r_logits = self.r_gate_head(future_state_time).view(B, H, C, 2)

        mu_weights = 2.0 * F.softmax(mu_logits, dim=-1)      # (B, H, C, 2)
        r_weights = 2.0 * F.softmax(r_logits, dim=-1)        # (B, H, C, 2)

        # --- Factor re-weighting (identity when all weights = 1) ---
        y_route = (
            mu_weights[..., 0] * mu_slow
            + mu_weights[..., 1] * mu_fast
            + r_weights[..., 0] * r_smooth
            + r_weights[..., 1] * r_osc
        )                                                     # (B, H, C)

        # --- Diagnostics (no gradient, logging only) ---
        with torch.no_grad():
            mu_probs = F.softmax(mu_logits, dim=-1)           # (B, H, C, 2)
            r_probs = F.softmax(r_logits, dim=-1)

            mu_ent = -(mu_probs * mu_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            r_ent = -(r_probs * r_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            self._diag_mu_gate_entropy_mean = float(mu_ent.item())
            self._diag_r_gate_entropy_mean = float(r_ent.item())

            self._diag_mu_branch_usage_mean = [
                float(mu_weights[..., k].mean().item()) for k in range(2)
            ]
            self._diag_r_branch_usage_mean = [
                float(r_weights[..., k].mean().item()) for k in range(2)
            ]
            self._diag_mu_fast_weight_mean = float(mu_weights[..., 1].mean().item())
            self._diag_r_osc_weight_mean = float(r_weights[..., 1].mean().item())

        return y_route

    def get_route_diagnostics(self) -> dict:
        return {
            "mu_gate_entropy_mean": self._diag_mu_gate_entropy_mean,
            "r_gate_entropy_mean": self._diag_r_gate_entropy_mean,
            "mu_branch_usage_mean": self._diag_mu_branch_usage_mean,
            "r_branch_usage_mean": self._diag_r_branch_usage_mean,
            "mu_fast_weight_mean": self._diag_mu_fast_weight_mean,
            "r_osc_weight_mean": self._diag_r_osc_weight_mean,
        }
