"""OTNorm — 1-D Optimal-Transport Normalizer (Quantile Coupling).

Theory
------
For two 1-D distributions P and Q, the unique Wasserstein-2-optimal transport
plan is the *quantile coupling*:

    T*(x) = F_Q^{-1}(F_P(x))

Setting Q = N(0, 1), this is exactly the *quantile normalisation* transform:

    z = Φ^{-1}( F_X(x) )

where Φ is the standard-normal CDF and F_X is the empirical CDF of the window.

Implementation
--------------
  Forward (normalize):
    1. Compute Q empirical quantile values from the window (strictly:
       linearly-interpolated order statistics at levels u_i = (i+0.5)/Q).
    2. Estimate the CDF at each x_t by linear interpolation into the quantile
       table (differentiable w.r.t. x_t; the table itself is detached).
    3. Apply the probit function: z_t = Φ^{-1}(p_t).

  Inverse (denormalize):
    1. Convert prediction z to probability: p = Φ(z).
    2. Linear interpolation into the stored quantile table: y = F_X^{-1}(p).

API (mirrors RevON / MissCorrRevIN)
------------------------------------
    nm = OTNorm(num_features=C)
    nm.set_missing_context(mask)     # optional; uses only observed values
    z  = nm.normalize(x)             # (B, T, C) → (B, T, C)
    y  = nm.denormalize(pred)        # (B, H, C) → (B, H, C)
    nm.loss()                        # returns 0
    nm.get_last_aux_stats()          # dict with OT diagnostics
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level numerical helpers
# ---------------------------------------------------------------------------

def _probit(p: torch.Tensor) -> torch.Tensor:
    """Probit  Φ^{-1}(p):  uses torch.erfinv for full GPU support."""
    p_safe = p.clamp(1e-6, 1.0 - 1e-6)
    return (2.0 * p_safe - 1.0).erfinv() * math.sqrt(2.0)


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """Standard-normal CDF Φ(z)."""
    return 0.5 * (1.0 + (z / math.sqrt(2.0)).erf())


def _interp1d(
    x: torch.Tensor,
    xp: torch.Tensor,
    yp: torch.Tensor,
) -> torch.Tensor:
    """Batched piecewise-linear interpolation (and linear extrapolation).

    Args:
        x:  (N, Qx) query points — same batch axis as xp/yp.
        xp: (N, Qp) known x-positions (monotone increasing).
        yp: (N, Qp) known y-values.

    Returns:
        y: (N, Qx) interpolated/extrapolated values.
    """
    # right=True: last knot strictly less than x falls in the previous bin
    idx = torch.searchsorted(xp.contiguous(), x.contiguous(), right=True)
    idx = idx.clamp(1, xp.shape[-1] - 1)          # (N, Qx) lie in [1, Qp-1]

    x0 = xp.gather(-1, idx - 1)
    x1 = xp.gather(-1, idx)
    y0 = yp.gather(-1, idx - 1)
    y1 = yp.gather(-1, idx)

    # Linear interpolation / extrapolation (slope from the enclosing bin)
    slope = (y1 - y0) / (x1 - x0).clamp(min=1e-12)
    return y0 + slope * (x - x0)


# ---------------------------------------------------------------------------
# OTNorm
# ---------------------------------------------------------------------------

class OTNorm(nn.Module):
    """1-D Wasserstein-optimal (quantile-coupling) normalizer.

    Args:
        num_features:     Number of input channels C.
        num_quantiles:    Resolution of the stored empirical quantile table Q
                          (default 64).  Higher Q ≈ smoother inverse but more
                          memory per batch.
        sigma_min:        Stability clamp (default 1e-4).  Not used directly
                          here, kept for API parity.
        eps:              Numerical epsilon.
    """

    def __init__(
        self,
        num_features: int,
        num_quantiles: int = 64,
        sigma_min: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.C = num_features
        self.Q = num_quantiles
        self.sigma_min = sigma_min
        self.eps = eps

        # per-batch cache: quantile table + quantile levels
        self._quantiles: Optional[torch.Tensor] = None   # (B, C, Q)
        self._levels:    Optional[torch.Tensor] = None   # (Q,)
        self._mask:      Optional[torch.Tensor] = None

        # Diagnostics
        self._last_range: float = 0.0
        self._last_gap_ratio: float = 0.0

    # ------------------------------------------------------------------
    # Missing-data interface
    # ------------------------------------------------------------------

    def set_missing_context(self, mask: torch.Tensor, **kwargs) -> None:
        """Cache missingness mask.  normalize() builds the CDF from observed
        values only when a mask is present."""
        self._mask = mask

    # ------------------------------------------------------------------
    # Quantile table construction
    # ------------------------------------------------------------------

    def _build_quantile_table(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q linearly-interpolated order statistics per channel.

        When a missing mask is stored, only observed values are used.

        Args:
            x: (B, T, C) windows.

        Returns:
            quantiles: (B, C, Q)  Q quantile values per channel per batch.
            levels:    (Q,)      corresponding probability levels.
        """
        B, T, C = x.shape
        Q = self.Q
        device, dtype = x.device, x.dtype

        # Probability levels u_i = (i + 0.5) / Q  for i = 0 … Q-1
        levels = torch.linspace(0.5 / Q, 1.0 - 0.5 / Q, Q, device=device, dtype=dtype)

        if self._mask is not None:
            # Build per-sample, per-channel sorted observed values, then
            # subsample to Q levels via linear interpolation.
            mask = self._mask   # (B, T, C)
            quantiles_list = []
            for c in range(C):
                col_q = []
                for b in range(B):
                    obs = x[b, :, c][mask[b, :, c].bool()]   # observed values
                    if obs.numel() < 2:
                        # Fallback: repeat single value
                        q_vals = obs.mean().expand(Q) if obs.numel() > 0 else x[b, :, c].mean().expand(Q)
                    else:
                        obs_sorted, _ = obs.sort()
                        n_obs = obs_sorted.shape[0]
                        # Interpolate at levels
                        positions = levels * n_obs - 0.5
                        positions = positions.clamp(0.0, float(n_obs - 1))
                        pos_int   = positions.long().clamp(0, n_obs - 2)
                        pos_frac  = (positions - pos_int.float()).clamp(0.0, 1.0)
                        q_vals    = (
                            obs_sorted[pos_int] * (1.0 - pos_frac)
                            + obs_sorted[(pos_int + 1).clamp(max=n_obs - 1)] * pos_frac
                        )   # (Q,)
                    col_q.append(q_vals)
                quantiles_list.append(torch.stack(col_q, dim=0))   # (B, Q)
            quantiles = torch.stack(quantiles_list, dim=1)          # (B, C, Q)
        else:
            # Fast path: sort the whole window
            x_sorted = x.sort(dim=1).values                         # (B, T, C)
            # Interpolate at Q levels from T samples
            positions = levels * T - 0.5                            # (Q,)
            positions = positions.clamp(0.0, float(T - 1))
            pos_int   = positions.long().clamp(0, T - 2)            # (Q,)
            pos_frac  = (positions - pos_int.float()).clamp(0.0, 1.0)  # (Q,)

            # Gather: expand to (B, T, C) → (B, C, T) for gather along last dim
            x_bct    = x_sorted.permute(0, 2, 1)                    # (B, C, T)
            idx0     = pos_int.unsqueeze(0).unsqueeze(0).expand(B, C, Q)
            idx1     = (pos_int + 1).clamp(max=T - 1).unsqueeze(0).unsqueeze(0).expand(B, C, Q)
            frac     = pos_frac.unsqueeze(0).unsqueeze(0).expand(B, C, Q)

            quantiles = x_bct.gather(2, idx0) * (1.0 - frac) + x_bct.gather(2, idx1) * frac

        return quantiles, levels   # (B, C, Q), (Q,)

    # ------------------------------------------------------------------
    # Empirical CDF via interpolation (differentiable in x)
    # ------------------------------------------------------------------

    def _empirical_cdf(
        self,
        x: torch.Tensor,
        quantiles: torch.Tensor,
        levels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute F_X(x_t) by linear interpolation into the quantile table.

        The quantile table is treated as **constants** (no gradient through
        it).  Gradients flow through x only, via the interpolation slope.

        Args:
            x:         (BC, T) input values.
            quantiles: (BC, Q) detached quantile table.
            levels:    (BC, Q) probability levels.

        Returns:
            p: (BC, T) CDF values clipped to (ε, 1-ε).
        """
        p = _interp1d(x, quantiles, levels)
        return p.clamp(self.eps, 1.0 - self.eps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map each channel window to N(0,1) via quantile coupling.

        Args:
            x: (B, T, C) input.

        Returns:
            z: (B, T, C) approximately standard-normal.
        """
        B, T, C = x.shape
        Q = self.Q

        # Build quantile table (detached constants)
        quantiles, levels = self._build_quantile_table(x)
        self._quantiles = quantiles.detach()
        self._levels    = levels.detach()

        # Diagnostics
        with torch.no_grad():
            q_range = (quantiles[..., -1] - quantiles[..., 0]).mean().item()
            q_gap   = (quantiles[..., 1:] - quantiles[..., :-1]).max().item()
            q_step  = max(q_range / max(Q - 1, 1), 1e-8)
            self._last_range     = q_range
            self._last_gap_ratio = q_gap / q_step if q_step > 0 else 0.0

        # (B, T, C) → (BC, T) for vectorised interp
        BC = B * C
        x_bc   = x.permute(0, 2, 1).reshape(BC, T)
        q_bc   = quantiles.reshape(BC, Q).detach()
        lv_bc  = levels.unsqueeze(0).expand(BC, Q)

        p_bc   = self._empirical_cdf(x_bc, q_bc, lv_bc)   # (BC, T)
        z_bc   = _probit(p_bc)                              # (BC, T)
        z      = z_bc.reshape(B, C, T).permute(0, 2, 1)   # (B, T, C)

        if self._mask is not None:
            z = self._mask * z
        return z

    def denormalize(self, pred: torch.Tensor) -> torch.Tensor:
        """Invert the quantile coupling.

        Args:
            pred: (B, H, C) predictions in flow (z) space.

        Returns:
            y: (B, H, C) in original data space.
        """
        if self._quantiles is None or self._levels is None:
            raise RuntimeError("OTNorm.denormalize() called before normalize().")

        B, H, C = pred.shape
        BC = B * C
        Q  = self.Q

        z_bc   = pred.permute(0, 2, 1).reshape(BC, H)     # (BC, H)
        p_bc   = _normal_cdf(z_bc)                         # (BC, H) ∈ (0,1)
        p_bc   = p_bc.clamp(self.eps, 1.0 - self.eps)

        q_bc   = self._quantiles.reshape(BC, Q)            # (BC, Q)
        lv_bc  = self._levels.unsqueeze(0).expand(BC, Q)   # (BC, Q)

        y_bc   = _interp1d(p_bc, lv_bc, q_bc)             # (BC, H)
        return y_bc.reshape(B, C, H).permute(0, 2, 1)     # (B, H, C)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)

    def loss(self, x_full: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._quantiles is None:
            return torch.zeros(())
        return self._quantiles.new_zeros(())

    def get_last_aux_stats(self) -> dict:
        return {
            "ot_quantile_range":    self._last_range,
            "ot_max_gap_ratio":     self._last_gap_ratio,
            "delta_mu_l2":          0.0,
            "delta_log_sigma_l2":   0.0,
        }
