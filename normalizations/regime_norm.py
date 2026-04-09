"""RegimeNorm — Prototype / Regime-Aware Normalization.

Motivation
----------
All standard normalizers (RevIN, Instance Norm, …) collapse every window
to *Mean = 0* regardless of whether the input comes from a "working-day
mode", a "holiday mode", or an "extreme-weather mode".  This discards
*which regime* the window belongs to.

Architecture
------------
A memory bank of K learnable prototype coordinate-systems (μ_k, σ_k) is
maintained.  Each input window is *softly routed* to them:

    1. Compute observed window statistics: (μ_obs, σ_obs).
    2. Encode via a small MLP: logits = enc([μ_obs, σ_obs]) ∈ ℝ^K.
    3. Routing weights:  w = softmax(logits / τ) ∈ Δ^K.
    4. Effective centroid: μ_eff = Σ_k w_k · proto_μ_k,
                            σ_eff = Σ_k w_k · exp(proto_log_σ_k).
    5. Normalize relative to the soft prototype:
           z = (x − μ_eff) / σ_eff.

Denormalization simply inverts step 5 using cached (μ_eff, σ_eff).

The prototypes are learned end-to-end.  Their diversity emerges naturally
because different windows specialise different prototypes for reconstruction
efficiency.  An optional diversity regulariser can further encourage spread.

API (mirrors RevON / FlowNorm)
--------------------------------
    nm = RegimeNorm(num_features=C)
    nm.set_missing_context(mask)
    z  = nm.normalize(x)           # (B, T, C) → (B, T, C)
    y  = nm.denormalize(pred)      # (B, H, C) → (B, H, C)
    nm.loss(x_full)                # optional diversity reg; default 0
    nm.get_last_aux_stats()        # routing entropy, top weight, …
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeNorm(nn.Module):
    """Prototype/Regime-Aware Normalizer with soft attention routing.

    Args:
        num_features:      Number of input channels C.
        num_prototypes:    Memory-bank size K (default 8).
        hidden_dim:        Encoder hidden units (default 32).
        sigma_min:         Lower clamp for σ_eff (default 1e-4).
        eps:               Numerical epsilon.
        temperature:       Softmax temperature τ (default 1.0).
        diversity_weight:  Coefficient for prototype-separation regulariser.
                           0.0 = disabled (default).
    """

    def __init__(
        self,
        num_features: int,
        num_prototypes: int = 8,
        hidden_dim: int = 32,
        sigma_min: float = 1e-4,
        eps: float = 1e-8,
        temperature: float = 1.0,
        diversity_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.C = num_features
        self.K = num_prototypes
        self.sigma_min = sigma_min
        self.eps = eps
        self.temperature = temperature
        self.diversity_weight = diversity_weight

        # Learnable prototype coordinate systems
        self.proto_mu        = nn.Parameter(torch.zeros(num_prototypes, num_features))
        self.proto_log_sigma = nn.Parameter(torch.zeros(num_prototypes, num_features))

        # Encoder: (μ_obs, σ_obs) → routing logits
        self.encoder = nn.Sequential(
            nn.Linear(2 * num_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_prototypes),
        )

        # per-batch cache
        self._mask:           Optional[torch.Tensor] = None
        self._mu_eff:         Optional[torch.Tensor] = None
        self._sigma_eff:      Optional[torch.Tensor] = None
        self._last_weights:   Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Missing-data interface
    # ------------------------------------------------------------------

    def set_missing_context(self, mask: torch.Tensor, **kwargs) -> None:
        """Cache missingness mask; statistics use only observed values."""
        self._mask = mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Route the window to its prototype blend and normalize.

        Args:
            x: (B, T, C) input.

        Returns:
            z: (B, T, C) normalized signal.
        """
        B, T, C = x.shape
        eps = self.eps

        # Observed-only statistics
        if self._mask is not None:
            mask = self._mask
            cnt  = mask.sum(dim=1).clamp(min=eps)                  # (B, C)
            mu_obs   = (mask * x).sum(dim=1) / cnt                 # (B, C)
            sigma_obs = (
                (mask * (x - mu_obs.unsqueeze(1)).pow(2)).sum(dim=1)
                / cnt.clamp(min=eps) + eps
            ).sqrt().clamp(min=self.sigma_min)
        else:
            mu_obs    = x.mean(dim=1)                              # (B, C)
            sigma_obs = x.std(dim=1).clamp(min=self.sigma_min)    # (B, C)

        # Routing: encode (μ_obs, σ_obs) → softmax weights
        stats  = torch.cat([mu_obs, sigma_obs], dim=-1)            # (B, 2C)
        logits = self.encoder(stats)                               # (B, K)
        weights = torch.softmax(logits / self.temperature, dim=-1) # (B, K)

        # Soft prototype blend
        # proto_mu:        (K, C)  → unsqueeze(0) → (1, K, C)
        # weights:         (B, K)  → unsqueeze(-1) → (B, K, 1)
        proto_sigma = self.proto_log_sigma.exp()                   # (K, C)
        mu_eff = (
            weights.unsqueeze(-1) * self.proto_mu.unsqueeze(0)
        ).sum(dim=1)                                               # (B, C)
        sigma_eff = (
            weights.unsqueeze(-1) * proto_sigma.unsqueeze(0)
        ).sum(dim=1).clamp(min=self.sigma_min)                    # (B, C)

        # Normalize
        z = (x - mu_eff.unsqueeze(1)) / sigma_eff.unsqueeze(1)   # (B, T, C)

        # Cache
        self._mu_eff      = mu_eff
        self._sigma_eff   = sigma_eff
        self._last_weights = weights.detach()

        if self._mask is not None:
            z = self._mask * z
        return z

    # ------------------------------------------------------------------
    # Inverse
    # ------------------------------------------------------------------

    def denormalize(self, pred: torch.Tensor) -> torch.Tensor:
        """Invert normalization using cached prototype blend.

        Args:
            pred: (B, H, C) backbone output.

        Returns:
            y: (B, H, C) in original space.
        """
        if self._mu_eff is None or self._sigma_eff is None:
            raise RuntimeError("RegimeNorm.denormalize() called before normalize().")
        return pred * self._sigma_eff.unsqueeze(1) + self._mu_eff.unsqueeze(1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, x_full: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Optional prototype-diversity regulariser.

        When diversity_weight > 0, penalises small pairwise distances between
        prototype centres, encouraging each prototype to specialise in a
        different region of the data space.

        L_div = -diversity_weight * mean_{k≠k'} ||μ_k - μ_k'||_2
        """
        if self.diversity_weight <= 0.0 or self._mu_eff is None:
            return self.proto_mu.new_zeros(())

        # Pairwise distances between prototype centres
        # proto_mu: (K, C) → pairwise squared dist (K, K)
        diff  = self.proto_mu.unsqueeze(0) - self.proto_mu.unsqueeze(1)  # (K,K,C)
        dists = diff.pow(2).sum(-1).clamp(min=1e-8).sqrt()              # (K, K)

        # Keep only upper triangle (exclude diagonal)
        K = self.K
        mask_upper = torch.triu(torch.ones(K, K, device=dists.device, dtype=torch.bool), diagonal=1)
        mean_dist  = dists[mask_upper].mean()

        return -self.diversity_weight * mean_dist

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_last_aux_stats(self) -> dict:
        if self._last_weights is None:
            return {
                "regime_entropy":     0.0,
                "regime_top_weight":  0.0,
                "mean_regime_id":     0.0,
                "delta_mu_l2":        0.0,
                "delta_log_sigma_l2": 0.0,
            }
        w   = self._last_weights                          # (B, K)
        ent = -(w * (w + 1e-8).log()).sum(-1).mean().item()
        top = w.max(dim=-1).values.mean().item()
        rid = w.argmax(dim=-1).float().mean().item()

        # Prototype spread: mean pairwise distance between learnable centres
        with torch.no_grad():
            diff  = self.proto_mu.unsqueeze(0) - self.proto_mu.unsqueeze(1)
            dists = diff.pow(2).sum(-1).clamp(min=1e-8).sqrt()
            K     = self.K
            mask_u = torch.triu(
                torch.ones(K, K, device=dists.device, dtype=torch.bool), diagonal=1
            )
            proto_spread = dists[mask_u].mean().item()

        return {
            "regime_entropy":       ent,
            "regime_top_weight":    top,
            "mean_regime_id":       rid,
            "regime_proto_spread":  proto_spread,
            "delta_mu_l2":          0.0,
            "delta_log_sigma_l2":   0.0,
        }
