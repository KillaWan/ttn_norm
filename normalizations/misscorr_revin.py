"""MissCorrRevIN: observed-only RevIN with a missing-aware mean correction.

Normalizes using masked (observed-only) statistics, then applies a tiny
two-layer MLP to correct the per-sample mean estimate.  Sigma is taken
directly from observed-only statistics (no sigma correction).  There is
no oracle state supervision — only a weak L2 penalty on the correction.

Usage (per batch):
    nm = MissCorrRevIN(num_features=C, ...)
    nm.set_missing_context(mask)           # before forward
    z_out = nm(x_obs)                      # (B, T, C): mask*z
    pred   = backbone(z_out)               # (B, H, C)
    y_hat  = nm.denormalize(pred)          # (B, H, C)
    reg    = nm.loss()                     # correction regularization
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MissCorrRevIN(nn.Module):
    """Observed-only RevIN with a missing-aware mean correction.

    A tiny two-layer MLP corrects the per-sample mean estimate based on
    masked statistics.  Sigma is taken directly from observed-only data
    (no sigma correction).  No oracle state supervision; only a weak L2
    penalty on the delta_mu correction is applied.

    Args:
        num_features:     Number of input channels C.
        corr_hidden_dim:  Hidden size of the correction MLP.
                          Defaults to min(128, 2*C).
        sigma_min:        Lower clamp for sigma (observed-only).
        delta_mu_scale:   Tanh bound for delta_mu correction.
        delta_reg_weight: Weight of the L2 correction penalty in loss().
        eps:              Numerical stability constant.
    """

    def __init__(
        self,
        num_features: int,
        corr_hidden_dim: Optional[int] = None,
        sigma_min: float = 1e-4,
        delta_mu_scale: float = 1.0,
        delta_log_sigma_scale: float = 1.0,   # accepted but unused (kept for CLI compat)
        delta_reg_weight: float = 1e-2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        C = num_features
        self.C = C
        self.eps = eps
        self.sigma_min = sigma_min
        self.delta_mu_scale = delta_mu_scale
        self.delta_reg_weight = delta_reg_weight

        hidden = corr_hidden_dim if corr_hidden_dim is not None else min(128, 2 * C)

        # Correction MLP: (4C) -> hidden -> (C)  — only corrects mean
        self.corr_mlp = nn.Sequential(
            nn.Linear(4 * C, hidden),
            nn.GELU(),
            nn.Linear(hidden, C),
        )

        # -- per-batch cache (populated by set_missing_context + normalize) --
        self._mask:      Optional[torch.Tensor] = None
        # computed & stored by normalize()
        self._mu:        Optional[torch.Tensor] = None
        self._sigma:     Optional[torch.Tensor] = None
        self._delta_mu:  Optional[torch.Tensor] = None
        # filled by loss()
        self._last_aux: dict = {}

    # ------------------------------------------------------------------
    # Public API expected by the training loop
    # ------------------------------------------------------------------

    def set_missing_context(
        self,
        mask: torch.Tensor,
        x_full: Optional[torch.Tensor] = None,  # accepted but not used
    ) -> None:
        """Cache the missing mask for this batch.

        Args:
            mask:   Binary mask, shape (B, T, C).  1 = observed, 0 = missing.
            x_full: Accepted but ignored (kept for interface compatibility).
        """
        self._mask = mask

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Compute corrected mu/sigma and return mask*z as (B, T, C).

        Args:
            x: Observed input (B, T, C).  Missing positions should already be
               zeroed (x_obs = mask * x_full).

        Returns:
            Tensor of shape (B, T, C): normalized signal with missing positions zeroed.
        """
        if self._mask is None:
            raise RuntimeError(
                "MissCorrRevIN.normalize() called without a missing mask. "
                "Call set_missing_context(mask, ...) before normalize()."
            )
        mask = self._mask          # (B, T, C)
        eps  = self.eps

        # ---- masked statistics (Req 2.3) --------------------------------
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=eps)     # (B,1,C)
        mu_obs   = (mask * x).sum(dim=1, keepdim=True) / mask_sum   # (B,1,C)
        sigma_obs = (
            (mask * (x - mu_obs) ** 2).sum(dim=1, keepdim=True)
            / mask_sum + eps
        ).sqrt()                                                      # (B,1,C)

        # ---- correction MLP inputs -----------------------------------
        mask_sum_bc = mask.sum(dim=1).clamp(min=eps)                  # (B, C)
        log_sigma_obs   = (sigma_obs.squeeze(1) + eps).log()          # (B, C)
        coverage        = mask.mean(dim=1)                            # (B, C)
        masked_mean_abs = (mask * x.abs()).sum(dim=1) / mask_sum_bc   # (B, C)

        mlp_in = torch.cat(
            [
                mu_obs.squeeze(1),   # (B, C)
                log_sigma_obs,       # (B, C)
                coverage,            # (B, C)
                masked_mean_abs,     # (B, C)
            ],
            dim=-1,
        )  # (B, 4C)

        # MLP outputs only delta_mu  (B, C)
        raw_delta_mu = self.corr_mlp(mlp_in)                          # (B, C)
        delta_mu     = self.delta_mu_scale * torch.tanh(raw_delta_mu) # (B, C)
        delta_mu     = delta_mu.unsqueeze(1)                          # (B,1,C)

        # ---- final state: corrected mean, observed-only sigma ----------
        mu    = mu_obs + delta_mu                            # (B,1,C)
        sigma = sigma_obs.clamp(min=self.sigma_min)          # (B,1,C)

        # cache for denormalize() and loss()
        self._mu       = mu
        self._sigma    = sigma
        self._delta_mu = delta_mu

        # ---- RevIN-style normalization, zero missing positions (Req 2.9) ---
        z   = (x - mu) / sigma   # (B, T, C)
        return mask * z          # (B, T, C)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """Invert normalization using cached mu / sigma.

        Args:
            y: Backbone output, shape (B, H, C).

        Returns:
            Forecast in original space, shape (B, H, C).
        """
        if self._mu is None or self._sigma is None:
            raise RuntimeError(
                "MissCorrRevIN.denormalize() called before normalize()."
            )
        return y * self._sigma + self._mu  # (B, H, C)

    def loss(self, x_full: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Correction regularization penalty.

        Returns delta_reg_weight * mean(delta_mu^2).  No oracle statistics
        are used; x_full is accepted but ignored.

        Returns:
            Scalar correction penalty.
        """
        if self._delta_mu is None:
            return self._mu.new_zeros(()) if self._mu is not None else torch.tensor(0.0)

        delta_mu_l2 = self._delta_mu.pow(2).mean()
        corr_reg    = self.delta_reg_weight * delta_mu_l2

        # store for get_last_aux_stats()
        self._last_aux = {
            "state_loss":         corr_reg.detach().item(),
            "mu_loss":            0.0,
            "sigma_loss":         0.0,
            "delta_mu_l2":        delta_mu_l2.detach().item(),
            "delta_log_sigma_l2": 0.0,
        }
        return corr_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for normalize(); called by TTNModel's generic nm(batch_x) path."""
        return self.normalize(x)

    def get_last_aux_stats(self) -> dict:
        """Return stats from the most recent loss() call."""
        return dict(self._last_aux)
