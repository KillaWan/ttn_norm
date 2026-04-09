"""FlowNorm — Adaptive Per-Channel Rational-Quadratic Spline Normalizer.

Architecture
------------
Two-stage per-channel transform:
  1. Linear standardization: z_pre = (x − μ_obs) / σ_obs  (like RevIN)
  2. Rational-Quadratic Spline (RQS): z = f_θ(z_pre) ≈ N(0, 1)

The spline knot parameters θ are predicted on-the-fly from the standardised
window's skewness and excess-kurtosis via a small shared MLP. This replaces
the rigid affine mapping of RevIN with a smooth invertible non-linear
warp that adapts to the local distribution *shape* (long tails, skew, etc.).

API (mirrors RevON / MissCorrRevIN)
------------------------------------
    nm = FlowNorm(num_features=C)
    nm.set_missing_context(mask)     # optional; enables observed-only stats
    z  = nm.normalize(x)             # (B, T, C) → (B, T, C)
    y  = nm.denormalize(pred)        # (B, H, C) → (B, H, C)
    nm.loss(x_full)                  # returns 0 (no auxiliary loss)
    nm.get_last_aux_stats()          # dict with spline diagnostics

References
----------
  Durkan et al. (2019) "Neural Spline Flows", NeurIPS.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowNorm(nn.Module):
    """Per-channel adaptive normalizing flow (Rational-Quadratic Spline).

    Args:
        num_features:   Number of input channels C.
        num_knots:      Number of spline bins K (default 8).
        hidden_dim:     Hidden units in the spline-parameter MLP (default 32).
        tail_bound:     Spline is active in [−B, B]; identity outside (default 5).
        sigma_min:      Lower clamp for σ_obs (default 1e-4).
        eps:            Numerical stability constant.
        min_deriv:      Minimum derivative at spline knots (default 1e-3).
    """

    def __init__(
        self,
        num_features: int,
        num_knots: int = 8,
        hidden_dim: int = 32,
        tail_bound: float = 5.0,
        sigma_min: float = 1e-4,
        eps: float = 1e-8,
        min_deriv: float = 1e-3,
    ) -> None:
        super().__init__()
        self.C = num_features
        self.K = num_knots
        self.bound = tail_bound
        self.sigma_min = sigma_min
        self.eps = eps
        self.min_deriv = min_deriv

        # MLP: (skew, excess_kurt) per channel → (3K+1) raw spline params
        # Shared across channels; processes each channel independently.
        out_dim = 3 * num_knots + 1   # K widths + K heights + (K+1) derivs
        self.spline_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # per-batch cache
        self._mask: Optional[torch.Tensor] = None
        self._cache: Optional[tuple] = None
        self._last_skew: float = 0.0
        self._last_ekurt: float = 0.0

    # ------------------------------------------------------------------
    # Missing-data interface (mask is optional; used for statistics only)
    # ------------------------------------------------------------------

    def set_missing_context(self, mask: torch.Tensor, **kwargs) -> None:
        """Cache the missingness mask.  normalize() uses only observed values."""
        self._mask = mask

    # ------------------------------------------------------------------
    # Spline helpers
    # ------------------------------------------------------------------

    def _parse_spline_params(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split and constrain raw MLP output → (widths, heights, derivs).

        Args:
            raw: (N, 3K+1) raw output from spline_net.

        Returns:
            widths:  (N, K)   positive, sum to 2*bound
            heights: (N, K)   positive, sum to 2*bound
            derivs:  (N, K+1) positive (>= min_deriv)
        """
        K, B = self.K, self.bound
        widths  = F.softmax(raw[:, :K],    dim=-1) * (2.0 * B)
        heights = F.softmax(raw[:, K:2*K], dim=-1) * (2.0 * B)
        derivs  = F.softplus(raw[:, 2*K:]) + self.min_deriv
        return widths, heights, derivs

    def _build_knots(
        self, widths: torch.Tensor, heights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cumulative-sum knot positions from bin widths/heights.

        Args:
            widths:  (N, K) positive bin widths summing to 2*bound
            heights: (N, K) positive bin heights summing to 2*bound

        Returns:
            knot_x: (N, K+1) x-knot positions in [-bound, bound]
            knot_y: (N, K+1) y-knot positions in [-bound, bound]
        """
        B = self.bound
        pad = torch.full(
            (widths.shape[0], 1), -B, device=widths.device, dtype=widths.dtype
        )
        knot_x = torch.cat([pad, -B + widths.cumsum(-1)],  dim=-1)   # (N, K+1)
        knot_y = torch.cat([pad, -B + heights.cumsum(-1)], dim=-1)   # (N, K+1)
        return knot_x, knot_y

    # ------------------------------------------------------------------
    # Core RQS transforms (vectorised over batch×channel)
    # ------------------------------------------------------------------

    def _rqs_forward(
        self,
        x: torch.Tensor,
        knot_x: torch.Tensor,
        knot_y: torch.Tensor,
        derivs: torch.Tensor,
    ) -> torch.Tensor:
        """Rational-Quadratic Spline forward pass.

        Args:
            x:      (B, T, C) input (in standardised space).
            knot_x: (B*C, K+1) x-knot positions.
            knot_y: (B*C, K+1) y-knot positions.
            derivs: (B*C, K+1) positive derivatives.

        Returns:
            z: (B, T, C) transformed values.
        """
        B_batch, T, C = x.shape
        K, bound = self.K, self.bound
        BC = B_batch * C

        # Reshape: (B, T, C) → (BC, T)
        x2d = x.permute(0, 2, 1).reshape(BC, T)

        w = knot_x[:, 1:] - knot_x[:, :-1]        # (BC, K) bin widths
        h = knot_y[:, 1:] - knot_y[:, :-1]        # (BC, K) bin heights
        s = h / w.clamp(min=1e-8)                  # (BC, K) avg slopes

        # Find bin: searchsorted returns insertion point; shift by -1 for left bin
        bin_idx = (
            torch.searchsorted(knot_x.contiguous(), x2d.contiguous(), right=True) - 1
        ).clamp(0, K - 1)                          # (BC, T)

        w_k   = w.gather(1, bin_idx)
        h_k   = h.gather(1, bin_idx)
        s_k   = s.gather(1, bin_idx)
        d_k   = derivs.gather(1, bin_idx)
        d_k1  = derivs.gather(1, (bin_idx + 1).clamp(max=K))
        kx_l  = knot_x.gather(1, bin_idx)
        ky_l  = knot_y.gather(1, bin_idx)

        zeta   = ((x2d - kx_l) / w_k.clamp(min=1e-8)).clamp(0.0, 1.0)
        numer  = h_k * (s_k * zeta.pow(2) + d_k * zeta * (1.0 - zeta))
        denom  = s_k + (d_k1 + d_k - 2.0 * s_k) * zeta * (1.0 - zeta)
        z2d    = ky_l + numer / denom.clamp(min=1e-8)

        # Identity outside [-bound, bound]
        tail   = (x2d < -bound) | (x2d > bound)
        z2d    = torch.where(tail, x2d, z2d)

        return z2d.reshape(B_batch, C, T).permute(0, 2, 1)   # (B, T, C)

    def _rqs_inverse(
        self,
        z: torch.Tensor,
        knot_x: torch.Tensor,
        knot_y: torch.Tensor,
        derivs: torch.Tensor,
    ) -> torch.Tensor:
        """Rational-Quadratic Spline inverse pass.

        Args:
            z:      (B, H, C) predictions in flow space.
            knot_x, knot_y, derivs: cached from normalize().

        Returns:
            x_pre: (B, H, C) in standardised (pre-flow) space.
        """
        B_batch, H, C = z.shape
        K, bound = self.K, self.bound
        BC = B_batch * C

        z2d    = z.permute(0, 2, 1).reshape(BC, H)

        w      = knot_x[:, 1:] - knot_x[:, :-1]
        h      = knot_y[:, 1:] - knot_y[:, :-1]
        s      = h / w.clamp(min=1e-8)

        # Find bin in y-space
        bin_idx = (
            torch.searchsorted(knot_y.contiguous(), z2d.contiguous(), right=True) - 1
        ).clamp(0, K - 1)

        w_k   = w.gather(1, bin_idx)
        h_k   = h.gather(1, bin_idx)
        s_k   = s.gather(1, bin_idx)
        d_k   = derivs.gather(1, bin_idx)
        d_k1  = derivs.gather(1, (bin_idx + 1).clamp(max=K))
        kx_l  = knot_x.gather(1, bin_idx)
        ky_l  = knot_y.gather(1, bin_idx)

        y_til = z2d - ky_l
        a     = h_k * (s_k - d_k) + y_til * (d_k1 + d_k - 2.0 * s_k)
        b_c   = h_k * d_k - y_til * (d_k1 + d_k - 2.0 * s_k)
        c_c   = -s_k * y_til

        disc  = (b_c.pow(2) - 4.0 * a * c_c).clamp(min=0.0)
        sqrtd = disc.sqrt()

        # Numerically stable root in [0, 1]:
        # quadratic: ζ = (-b + √Δ) / (2a)
        # linear fallback (a ≈ 0): ζ = -c / b
        two_a    = 2.0 * a
        zeta_q   = (-b_c + sqrtd) / torch.where(
            two_a.abs() > 1e-7, two_a, torch.ones_like(two_a)
        )
        zeta_lin = -c_c / torch.where(
            b_c.abs() > 1e-7, b_c, torch.ones_like(b_c)
        )
        zeta = torch.where(a.abs() > 1e-7, zeta_q, zeta_lin).clamp(0.0, 1.0)

        x2d_rec = kx_l + zeta * w_k

        # Identity outside [-bound, bound]
        tail    = (z2d < -bound) | (z2d > bound)
        x2d_rec = torch.where(tail, z2d, x2d_rec)

        return x2d_rec.reshape(B_batch, C, H).permute(0, 2, 1)   # (B, H, C)

    # ------------------------------------------------------------------
    # Public forward / inverse
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Two-stage forward: affine standardisation → RQS warp → z ≈ N(0,1).

        Args:
            x: (B, T, C) input (observed values; missing already zeroed).

        Returns:
            z: (B, T, C) normalised signal (missing positions remain 0).
        """
        B, T, C = x.shape
        eps = self.eps

        # Observed-only statistics
        if self._mask is not None:
            mask = self._mask
            cnt  = mask.sum(dim=1, keepdim=True).clamp(min=eps)   # (B,1,C)
            mu   = (mask * x).sum(dim=1, keepdim=True) / cnt
            sig  = (
                (mask * (x - mu).pow(2)).sum(dim=1, keepdim=True) / cnt + eps
            ).sqrt().clamp(min=self.sigma_min)
        else:
            mu  = x.mean(dim=1, keepdim=True)
            sig = x.std(dim=1, keepdim=True).clamp(min=self.sigma_min)

        z_pre = (x - mu) / sig   # (B, T, C) affine-standardised

        # Shape stats from z_pre (dimensionless → stable MLP input)
        z_pre_d = z_pre.detach()
        skew  = z_pre_d.pow(3).mean(dim=1)                  # (B, C)
        ekurt = z_pre_d.pow(4).mean(dim=1) - 3.0            # (B, C) excess kurtosis
        stats = torch.stack([skew, ekurt], dim=-1)           # (B, C, 2)

        # Predict spline params; processed per channel independently
        BC = B * C
        raw = self.spline_net(stats.reshape(BC, 2))          # (BC, 3K+1)
        widths, heights, derivs = self._parse_spline_params(raw)
        knot_x, knot_y = self._build_knots(widths, heights)  # (BC, K+1) each

        # Reshape back to (B, C, K+1) for cache
        kx_bc = knot_x.reshape(B, C, self.K + 1)
        ky_bc = knot_y.reshape(B, C, self.K + 1)
        d_bc  = derivs.reshape(B, C, self.K + 1)

        # RQS forward
        z = self._rqs_forward(z_pre, knot_x, knot_y, derivs)

        # Cache: need (B, C, K+1) shapes + affine params for denorm
        self._cache = (mu, sig, kx_bc, ky_bc, d_bc)
        self._last_skew  = float(skew.abs().mean().item())
        self._last_ekurt = float(ekurt.abs().mean().item())

        # Zero missing positions if mask provided
        if self._mask is not None:
            z = self._mask * z
        return z

    def denormalize(self, pred: torch.Tensor) -> torch.Tensor:
        """Invert the flow: pred → inverse-RQS → inverse-affine → output.

        Args:
            pred: (B, H, C) backbone output in flow space.

        Returns:
            y: (B, H, C) predictions in original space.
        """
        if self._cache is None:
            raise RuntimeError("FlowNorm.denormalize() called before normalize().")
        mu, sig, kx_bc, ky_bc, d_bc = self._cache
        B, H, C = pred.shape
        BC = B * C

        # Flatten to (BC, K+1) for _rqs_inverse
        kx = kx_bc.reshape(BC, self.K + 1)
        ky = ky_bc.reshape(BC, self.K + 1)
        d  = d_bc.reshape(BC, self.K + 1)

        z_pre = self._rqs_inverse(pred, kx, ky, d)   # (B, H, C)
        return z_pre * sig + mu                       # (B, H, C)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # convenience alias
        return self.normalize(x)

    # ------------------------------------------------------------------
    # Auxiliary interface
    # ------------------------------------------------------------------

    def loss(self, x_full: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._cache is None:
            return torch.zeros(())
        mu = self._cache[0]
        return mu.new_zeros(())

    def get_last_aux_stats(self) -> dict:
        return {
            "flow_mean_abs_skew":  self._last_skew,
            "flow_mean_abs_ekurt": self._last_ekurt,
            "delta_mu_l2":         0.0,
            "delta_log_sigma_l2":  0.0,
        }
