"""LocalTransportPath — monotonic piecewise-linear spline transform.

Applies a per-timestep, channel-shared monotonic PL spline to y_base.
Spline parameters are generated from future_state_time by a shared linear head.

Identity initialisation: zero-init head produces uniform bin widths and unit
slopes, which yields y_route == y_base for all inputs (including extrapolation
beyond the clip domain).

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
    """Monotonic K-bin piecewise-linear spline over y_base.

    The spline is parametrised by:
      - K bin widths  (softmax → sum to domain, so positive and normalised)
      - K bin slopes  (exp → positive; init 1.0 gives identity)

    Values outside the nominal domain [-CLIP, CLIP] are handled by linear
    extrapolation using the boundary bin slope, so identity is preserved
    everywhere at initialisation, not just inside the clip range.
    """

    K: int = 8
    CLIP: float = 6.0

    def __init__(self, pred_len: int, enc_in: int, sigma_min: float = 1e-3, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.domain = 2.0 * self.CLIP  # total width = 12

        # Shared head: maps C state features → 2K spline params per timestep.
        # Zero-init → softmax(0)=1/K (uniform widths), exp(0)=1 (unit slopes) → identity.
        self.head = nn.Linear(enc_in, 2 * self.K)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        y_norm: torch.Tensor,
        y_base: torch.Tensor,           # (B, H, C)
        future_state_hat: torch.Tensor,
        future_state_time: torch.Tensor,  # (B, H, C)
        mu_base_fut: torch.Tensor,
        std_base_fut: torch.Tensor,
        base_time_mean: torch.Tensor,
        base_time_std: torch.Tensor,
    ) -> torch.Tensor:
        B, H, C = y_base.shape
        K = self.K

        # Generate spline params: (B, H, 2K)
        raw = self.head(future_state_time)
        raw_w, raw_s = raw[..., :K], raw[..., K:]  # (B, H, K) each

        # Bin widths: softmax → all positive, sum to domain
        widths = F.softmax(raw_w, dim=-1) * self.domain  # (B, H, K)

        # Bin slopes: exp → positive (1.0 at zero init)
        slopes = torch.exp(raw_s)  # (B, H, K)

        # x-boundaries: (B, H, K+1) starting from -CLIP
        zeros = torch.zeros(B, H, 1, device=y_base.device, dtype=y_base.dtype)
        x_boundaries = torch.cat(
            [zeros, torch.cumsum(widths, dim=-1)], dim=-1
        ) - self.CLIP  # [..., 0] = -CLIP, [..., K] = CLIP

        # y-boundaries: (B, H, K+1) starting from -CLIP
        heights = slopes * widths  # (B, H, K)
        y_boundaries = torch.cat(
            [zeros, torch.cumsum(heights, dim=-1)], dim=-1
        ) - self.CLIP

        # For each value in y_base find the containing bin and interpolate.
        # y_base: (B, H, C); boundaries are channel-shared (B, H, K+/-1).
        x = y_base  # extrapolation handles out-of-domain correctly

        # Bin index: count how many left boundaries are <= x, clamp to [0, K-1]
        # x: (B, H, C, 1)  vs  x_boundaries: (B, H, 1, K+1)
        mask = (x.unsqueeze(-1) >= x_boundaries.unsqueeze(2))  # (B, H, C, K+1)
        bin_idx = mask.sum(dim=-1).clamp(1, K) - 1             # (B, H, C) in [0, K-1]

        # Expand boundary tensors to (B, H, C, K+/-1) for gather
        idx = bin_idx.unsqueeze(-1)  # (B, H, C, 1)
        x_b = x_boundaries.unsqueeze(2).expand(B, H, C, K + 1)
        y_b = y_boundaries.unsqueeze(2).expand(B, H, C, K + 1)
        s_b = slopes.unsqueeze(2).expand(B, H, C, K)

        x_left  = x_b.gather(-1, idx).squeeze(-1)   # (B, H, C)
        y_left  = y_b.gather(-1, idx).squeeze(-1)   # (B, H, C)
        slope_k = s_b.gather(-1, idx).squeeze(-1)   # (B, H, C)

        # Linear interpolation within bin (extrapolation uses boundary slope)
        y_route = y_left + slope_k * (x - x_left)
        return y_route
