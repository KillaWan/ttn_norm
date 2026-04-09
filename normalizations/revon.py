"""RevON: Observed-Only RevIN normalization for missing-data baseline.

Minimal baseline that normalizes using masked (observed-only) statistics,
with no correction net, no extra state, and no auxiliary losses.
The only question it answers: how well does observed-only mu/sigma work
under fixed contiguous-block missing?

Usage (per batch):
    nm = RevON(num_features=C)
    nm.set_missing_context(mask)          # before forward
    z_out = nm(x_obs)                     # (B, T, C): mask*z
    pred  = backbone(z_out)               # (B, H, C)
    y_hat = nm.denormalize(pred)          # (B, H, C)
    loss  = nm.loss(x_full)               # always 0
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RevON(nn.Module):
    """Observed-only RevIN baseline for missing-data normalization.

    No correction net.  No learnable parameters beyond what a plain
    nn.Module provides.  Statistics are computed directly from the
    observed (non-missing) positions.

    Args:
        num_features: Number of input channels C.
        sigma_min:    Lower clamp for sigma (guards against pathological cases
                      when very few timesteps are observed).
        eps:          Numerical stability constant.
    """

    def __init__(
        self,
        num_features: int,
        sigma_min: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.C         = num_features
        self.sigma_min = sigma_min
        self.eps       = eps

        # per-batch cache
        self._mask:  Optional[torch.Tensor] = None
        self._mu:    Optional[torch.Tensor] = None
        self._sigma: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_missing_context(
        self,
        mask: torch.Tensor,
        x_full: Optional[torch.Tensor] = None,   # accepted but not used
    ) -> None:
        """Cache the missing mask for this batch.

        Args:
            mask:   Binary mask (B, T, C).  1 = observed, 0 = missing.
            x_full: Ignored.  Accepted only to unify the interface with
                    other missing-norm classes.
        """
        self._mask = mask

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Compute observed-only mu/sigma and return mask*z as (B, T, C).

        Args:
            x: Observed input (B, T, C).  Missing positions should already
               be zeroed (x_obs = mask * x_full).

        Returns:
            Tensor of shape (B, T, C): normalized signal with missing positions zeroed.
        """
        if self._mask is None:
            raise RuntimeError(
                "RevON.normalize() called without a missing mask. "
                "Call set_missing_context(mask) before normalize()."
            )
        mask = self._mask
        eps  = self.eps

        # observed-only statistics (Req 2.3)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=eps)       # (B,1,C)
        mu_obs   = (mask * x).sum(dim=1, keepdim=True) / mask_sum     # (B,1,C)
        sigma_obs = (
            (mask * (x - mu_obs) ** 2).sum(dim=1, keepdim=True)
            / mask_sum + eps
        ).sqrt()                                                        # (B,1,C)

        # final state — no correction (Req 2.4)
        mu    = mu_obs
        sigma = sigma_obs.clamp(min=self.sigma_min)

        # cache for denormalize()
        self._mu    = mu
        self._sigma = sigma

        # normalize and zero missing positions
        z   = (x - mu) / sigma   # (B, T, C)
        return mask * z          # (B, T, C)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """Invert normalization using cached mu/sigma.

        Args:
            y: Backbone output (B, H, C).

        Returns:
            Forecast in original space, shape (B, H, C).
        """
        if self._mu is None or self._sigma is None:
            raise RuntimeError(
                "RevON.denormalize() called before normalize()."
            )
        return y * self._sigma + self._mu  # (B, H, C)

    def loss(self, x_full: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Always returns 0.  RevON has no auxiliary loss."""
        if self._mu is not None:
            return self._mu.new_zeros(())
        return torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for normalize(); called by TTNModel's generic nm(batch_x) path."""
        return self.normalize(x)

    def get_last_aux_stats(self) -> dict:
        """Return zero-valued stats to keep training logs consistent."""
        return {
            "state_loss":         0.0,
            "mu_loss":            0.0,
            "sigma_loss":         0.0,
            "delta_mu_l2":        0.0,
            "delta_log_sigma_l2": 0.0,
        }
