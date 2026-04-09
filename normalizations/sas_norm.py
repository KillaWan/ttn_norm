"""SASNorm: Shrinkage-to-Anchor-Statistic normalization for missing data.

Observed-only statistics are unstable under contiguous missing blocks.
SASNorm stabilizes them by shrinking towards a training-set reference
("anchor") indexed by the time-phase of the lookback window's end point.

The shrinkage weight is driven by two signals:
  - coverage:  fraction of observed timesteps  (high → trust observed stats)
  - gap_ratio: longest contiguous missing block as a fraction of window
               (high → distrust observed stats, fall back to anchor)

Learnable scalars tau0 and tau1 (non-negative via softplus) control the
shrinkage trade-off.  Missing positions are filled with a learnable
per-channel missing_token before being passed to the backbone.

This is NOT a predictor, NOT a hidden-state estimator, NOT oracle supervised.
It is a missing-robust coordinate system: observed-only normalization with an
anchor prior to reduce variance when coverage is low or the missing gap is
large.

Usage (per batch):
    nm = SASNorm(num_features=C, anchor_period=P, anchor_mu=.., ...)
    nm.set_missing_context(mask, phase_id)   # before forward
    z_out = nm(x_obs)                        # (B, T, C)
    pred  = backbone(z_out)                  # (B, H, C)
    y_hat = nm.denormalize(pred)             # (B, H, C)
    loss  = nm.loss()                        # always 0
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SASNorm(nn.Module):
    """Observed-only RevIN with phase-indexed anchor shrinkage.

    Args:
        num_features:        Number of input channels C.
        anchor_period:       Phase period P (e.g. 24 for hourly diurnal,
                             168 for weekly).
        anchor_mu:           Pre-computed anchor mean table,  shape (P, C).
        anchor_log_sigma:    Pre-computed anchor log-std table, shape (P, C).
        sigma_min:           Lower clamp for sigma.
        eps:                 Numerical stability constant.
        init_missing_token:  Initial fill value for the learnable missing token.
    """

    def __init__(
        self,
        num_features: int,
        anchor_period: int,
        anchor_mu: torch.Tensor,           # (P, C)
        anchor_log_sigma: torch.Tensor,    # (P, C)
        sigma_min: float = 1e-4,
        eps: float = 1e-8,
        init_missing_token: float = 0.0,
    ) -> None:
        super().__init__()
        C = num_features
        P = anchor_period
        self.C = C
        self.P = P
        self.sigma_min = sigma_min
        self.eps = eps

        # Learnable shrinkage scalars (non-negative via softplus)
        self.tau0_raw = nn.Parameter(torch.zeros(1))
        self.tau1_raw = nn.Parameter(torch.zeros(1))

        # Learnable per-channel missing token in normalized space (1, 1, C)
        self.missing_token = nn.Parameter(
            torch.full((1, 1, C), fill_value=init_missing_token)
        )

        # Anchor stats registered as buffers (move with model.to(device))
        self.register_buffer("anchor_mu",        anchor_mu.float().clone())
        self.register_buffer("anchor_log_sigma", anchor_log_sigma.float().clone())

        # Per-batch cache
        self._mask:     Optional[torch.Tensor] = None
        self._phase_id: Optional[torch.Tensor] = None
        self._mu:       Optional[torch.Tensor] = None
        self._sigma:    Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_missing_context(
        self,
        mask: torch.Tensor,
        phase_id: torch.Tensor,
        x_full: Optional[torch.Tensor] = None,   # accepted but not used
    ) -> None:
        """Cache the missing mask and phase index for this batch.

        Args:
            mask:     Binary mask, shape (B, T, C).  1=observed, 0=missing.
            phase_id: Integer phase index per sample, shape (B,).
                      Range [0, anchor_period - 1].
            x_full:   Accepted but ignored (interface compatibility).
        """
        self._mask     = mask
        self._phase_id = phase_id.long()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Shrinkage-stabilized normalization.

        Args:
            x: Observed input (B, T, C), missing positions already zeroed
               (x_obs = mask * x_full).

        Returns:
            Tensor of shape (B, T, C):
                observed positions → normalized value  (mask * z)
                missing positions  → learnable missing_token  ((1-mask)*token)
        """
        if self._mask is None or self._phase_id is None:
            raise RuntimeError(
                "SASNorm.normalize() called without context. "
                "Call set_missing_context(mask, phase_id) first."
            )
        mask     = self._mask       # (B, T, C)
        phase_id = self._phase_id   # (B,)
        eps      = self.eps

        # ---- observed-only statistics -----------------------------------
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=eps)     # (B,1,C)
        mu_obs   = (mask * x).sum(dim=1, keepdim=True) / mask_sum   # (B,1,C)
        var_obs  = (mask * (x - mu_obs) ** 2).sum(dim=1, keepdim=True) / mask_sum
        sigma_obs = (var_obs + eps).sqrt()                           # (B,1,C)

        # ---- anchor stats for this batch --------------------------------
        # anchor_mu, anchor_log_sigma are (P, C) buffers
        mu_ref        = self.anchor_mu[phase_id].unsqueeze(1)         # (B,1,C)
        log_sigma_ref = self.anchor_log_sigma[phase_id].unsqueeze(1)  # (B,1,C)

        # ---- shrinkage weights -----------------------------------------
        # coverage: fraction of observed timesteps per (sample, channel)
        coverage  = mask.mean(dim=1, keepdim=True)                   # (B,1,C)
        # gap_ratio: longest contiguous missing block as fraction of T
        gap_ratio = self._max_gap_ratio(mask)                        # (B,1,1)

        tau0 = F.softplus(self.tau0_raw)   # scalar
        tau1 = F.softplus(self.tau1_raw)   # scalar
        w    = coverage / (coverage + tau0 + tau1 * gap_ratio + eps) # (B,1,C)

        # ---- final state ------------------------------------------------
        mu        = w * mu_obs + (1.0 - w) * mu_ref                 # (B,1,C)
        log_sigma = (
            w * (sigma_obs + eps).log() + (1.0 - w) * log_sigma_ref
        )                                                            # (B,1,C)
        sigma = log_sigma.exp().clamp(min=self.sigma_min)            # (B,1,C)

        # Cache for denormalize()
        self._mu    = mu
        self._sigma = sigma

        # ---- normalize + fill missing with token -----------------------
        z = (x - mu) / sigma                        # (B, T, C)
        return mask * z + (1.0 - mask) * self.missing_token  # (B, T, C)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """Invert normalization using cached mu/sigma.

        Args:
            y: Backbone output, shape (B, H, C).

        Returns:
            Forecast in original space, shape (B, H, C).
        """
        if self._mu is None or self._sigma is None:
            raise RuntimeError("SASNorm.denormalize() called before normalize().")
        return y * self._sigma + self._mu   # (B, H, C)

    def loss(self, x_full: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Always returns 0.  SASNorm has no auxiliary loss."""
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_gap_ratio(mask: torch.Tensor) -> torch.Tensor:
        """Compute the longest contiguous missing block as a fraction of T.

        Args:
            mask: (B, T, C) binary mask.  1=observed, 0=missing.

        Returns:
            Tensor of shape (B, 1, 1).
        """
        B, T, _ = mask.shape
        # All channels have the same missing pattern in our contiguous-block
        # setup.  Use the first channel for efficiency.
        is_missing = (mask[:, :, 0] < 0.5).cpu()   # (B, T) bool
        result = torch.zeros(B)
        for b in range(B):
            max_run = cur = 0
            for v in is_missing[b]:
                if v:
                    cur += 1
                    if cur > max_run:
                        max_run = cur
                else:
                    cur = 0
            result[b] = max_run / T
        return result.to(mask.device).view(B, 1, 1)
