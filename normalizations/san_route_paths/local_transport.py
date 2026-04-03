"""LocalTransportPath — monotonic piecewise-linear spline transform.

Applies a per-timestep, per-channel monotonic PL spline to y_base.
Each (b, t, c) position receives its own K-bin spline, whose parameters are
generated from future_state_time by a shared linear head that maps the C-dim
feature vector to C × 2K scalar parameters.

Identity initialisation: zero-init head → softmax(0)=1/K (uniform widths),
exp(0)=1 (unit slopes) → y_route == y_base for all inputs including
out-of-domain values (linear extrapolation with boundary slope).

Unified path interface:
    forward(y_norm, y_base, future_state_hat, future_state_time,
            mu_base_fut, std_base_fut, base_time_mean, base_time_std)
    -> y_route: (B, H, C)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalTransportPath(nn.Module):
    """Monotonic K-bin piecewise-linear spline — per-timestep, per-channel.

    The head maps future_state_time (B, H, C) → (B, H, C, 2K):
      first K values per channel: raw bin widths  → softmax → sum to domain
      last  K values per channel: raw bin slopes  → exp     → strictly positive

    Values outside the nominal domain are linearly extrapolated using the
    boundary bin slope, preserving the identity mapping at initialisation
    for any input magnitude.
    """

    K: int = 8
    CLIP: float = 6.0

    def __init__(self, pred_len: int, enc_in: int, sigma_min: float = 1e-3, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.domain = 2.0 * self.CLIP  # total domain width = 12

        # Head: C features → C × 2K per-channel spline params.
        # Output (B, H, C * 2K) reshaped to (B, H, C, 2K).
        # Zero-init: widths uniform (12/K each), slopes = 1 → identity.
        self.head = nn.Linear(enc_in, enc_in * 2 * self.K)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Diagnostic cache (read via get_route_diagnostics, not part of loss)
        self._diag_mean_abs_log_slope: float = 0.0
        self._diag_mean_abs_slope_minus_1: float = 0.0
        self._diag_mean_bin_entropy: float = 0.0
        self._diag_mean_out_of_domain_ratio: float = 0.0

    def forward(
        self,
        y_norm: torch.Tensor,
        y_base: torch.Tensor,             # (B, H, C)
        future_state_hat: torch.Tensor,
        future_state_time: torch.Tensor,  # (B, H, C)
        mu_base_fut: torch.Tensor,
        std_base_fut: torch.Tensor,
        base_time_mean: torch.Tensor,
        base_time_std: torch.Tensor,
    ) -> torch.Tensor:
        B, H, C = y_base.shape
        K = self.K

        # Per-channel spline params: (B, H, C, 2K)
        raw = self.head(future_state_time).view(B, H, C, 2 * K)
        raw_w = raw[..., :K]   # (B, H, C, K) — raw bin widths
        raw_s = raw[..., K:]   # (B, H, C, K) — raw bin slopes

        # Widths: softmax over K bins → positive, sum to domain (12)
        widths = F.softmax(raw_w, dim=-1) * self.domain   # (B, H, C, K)

        # Slopes: exp → positive (initialised to 1.0)
        slopes = torch.exp(raw_s)                          # (B, H, C, K)

        # x-boundaries: (B, H, C, K+1), first entry = -CLIP
        zeros = torch.zeros(B, H, C, 1, device=y_base.device, dtype=y_base.dtype)
        x_boundaries = (
            torch.cat([zeros, torch.cumsum(widths, dim=-1)], dim=-1) - self.CLIP
        )  # [..., 0] = -CLIP, [..., K] = CLIP

        # y-boundaries: (B, H, C, K+1), first entry = -CLIP
        heights = slopes * widths                          # (B, H, C, K)
        y_boundaries = (
            torch.cat([zeros, torch.cumsum(heights, dim=-1)], dim=-1) - self.CLIP
        )

        # Bin lookup: for each (b, h, c) find which bin y_base falls into.
        # x: (B, H, C, 1)  vs  x_boundaries: (B, H, C, K+1)
        x = y_base.unsqueeze(-1)                          # (B, H, C, 1)
        mask = (x >= x_boundaries)                        # (B, H, C, K+1)
        bin_idx = mask.sum(dim=-1).clamp(1, K) - 1        # (B, H, C) in [0, K-1]

        idx = bin_idx.unsqueeze(-1)                        # (B, H, C, 1)
        x_left  = x_boundaries.gather(-1, idx).squeeze(-1)  # (B, H, C)
        y_left  = y_boundaries.gather(-1, idx).squeeze(-1)  # (B, H, C)
        slope_k = slopes.gather(-1, idx).squeeze(-1)         # (B, H, C)

        # Linear interpolation within bin (extrapolation uses boundary slope)
        y_route = y_left + slope_k * (y_base - x_left)

        with torch.no_grad():
            # |log(slope)| = |raw_s| since slopes = exp(raw_s)
            self._diag_mean_abs_log_slope = float(raw_s.abs().mean().item())
            self._diag_mean_abs_slope_minus_1 = float((slopes - 1.0).abs().mean().item())
            # Bin entropy: entropy of softmax(raw_w) = widths / domain
            probs = widths / self.domain   # (B, H, C, K), sums to 1 per bin
            bin_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            self._diag_mean_bin_entropy = float(bin_entropy.item())
            out_of_domain = ((y_base < -self.CLIP) | (y_base > self.CLIP)).float().mean()
            self._diag_mean_out_of_domain_ratio = float(out_of_domain.item())

        return y_route

    def get_route_diagnostics(self) -> dict:
        return {
            "mean_abs_log_slope": self._diag_mean_abs_log_slope,
            "mean_abs_slope_minus_1": self._diag_mean_abs_slope_minus_1,
            "mean_bin_entropy": self._diag_mean_bin_entropy,
            "mean_out_of_domain_ratio": self._diag_mean_out_of_domain_ratio,
        }
