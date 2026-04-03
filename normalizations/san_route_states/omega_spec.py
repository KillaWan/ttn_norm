"""OmegaSpecState — patch-wise spectral predictability / spectral concentration.

Physical definition
-------------------
For each patch and each channel, compute the normalized spectral entropy of the
patch's power spectrum (excluding the DC bin), then define:

    omega_spec = 1 - H_norm

where H_norm is the normalized spectral entropy:

    q     = power_non_dc / (sum(power_non_dc) + eps)     # probability distribution
    H     = - sum(q * log(q + eps))                       # Shannon entropy
    H_norm = H / log(K)                                   # K = number of non-DC bins

Interpretation:
  - omega_spec near 1: spectrum is concentrated in a few frequency bins →
    strong periodic / oscillatory local structure → more amenable to structured
    correction by route paths.
  - omega_spec near 0: spectrum is diffuse across all bins → locally noise-like
    → oscillatory restoration is less appropriate.

Output range: (0, 1) after adapt_future_state (sigmoid + clamp).
Raw output from extract/build is also in (0, 1) by construction, but may include
boundary values in degenerate cases.

Framework contract:
  - This file defines the raw physical quantity only.
  - General-purpose patch-state standardisation is applied by SANRouteNorm via
    _normalize_patch_state() after extraction — do not apply it here.
  - adapt_future_state only enforces the valid domain (0, 1) via sigmoid + clamp.
"""
from __future__ import annotations

import torch

from .base import RouteStateBase


def _compute_omega_spec_from_windows(
    windows: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute patch-wise spectral concentration from pre-extracted windows.

    Args:
        windows: (B, P, W, C)  pre-extracted patch windows.
        eps:     numerical stability floor.

    Returns:
        (B, P, C) tensor of omega_spec values in [0, 1].
        Returns zeros if the patch is too short to have any non-DC frequency bins.
    """
    B, P, W, C = windows.shape

    # Number of non-DC rfft bins: rfft gives W//2+1 bins; bin 0 is DC.
    n_rfft = W // 2 + 1
    n_non_dc = n_rfft - 1   # exclude DC bin 0

    if n_non_dc <= 1:
        return torch.zeros(B, P, C, device=windows.device, dtype=windows.dtype)

    # --- Remove per-patch mean (de-mean along time axis) ---
    x = windows - windows.mean(dim=2, keepdim=True)   # (B, P, W, C)

    # --- Hann window (same dtype/device as input) ---
    hann = torch.hann_window(W, device=windows.device, dtype=windows.dtype)
    # Reshape for broadcasting: (1, 1, W, 1)
    hann = hann.view(1, 1, W, 1)
    x = x * hann   # (B, P, W, C)

    # --- rfft along time dimension (dim=2) ---
    # torch.fft.rfft returns complex tensor: (B, P, W//2+1, C)
    x_perm = x.permute(0, 1, 3, 2)   # (B, P, C, W) — fft expects last dim
    spec = torch.fft.rfft(x_perm, dim=-1)   # (B, P, C, W//2+1)

    # --- Power spectrum (excluding DC bin 0) ---
    power = spec.real ** 2 + spec.imag ** 2   # (B, P, C, W//2+1)
    power_non_dc = power[..., 1:]              # (B, P, C, n_non_dc)

    # --- Normalize to probability distribution ---
    power_sum = power_non_dc.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = power_non_dc / power_sum               # (B, P, C, n_non_dc)

    # --- Normalized spectral entropy ---
    H = -(q * (q + eps).log()).sum(dim=-1)     # (B, P, C)
    log_K = float(n_non_dc) ** 0  # placeholder; compute properly below
    log_K = torch.tensor(n_non_dc, dtype=windows.dtype, device=windows.device).log()
    H_norm = H / log_K.clamp_min(eps)         # (B, P, C) in [0, 1]

    # --- Spectral concentration ---
    omega_spec = 1.0 - H_norm                  # (B, P, C) in [0, 1]
    return omega_spec.clamp(0.0, 1.0)


class OmegaSpecState(RouteStateBase):
    """Patch-wise spectral concentration state.

    omega_spec = 1 - normalized_spectral_entropy

    High values → spectrally concentrated patches (periodic structure).
    Low  values → spectrally diffuse patches (noise-like).
    """

    @property
    def name(self) -> str:
        return "omega_spec"

    def extract_hist_state(
        self,
        x_hist: torch.Tensor,
        hist_windows: torch.Tensor,   # (B, P_hist, W, C)
        mu_hist: torch.Tensor,
        std_hist: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute raw omega_spec for historical patches.

        Returns (B, P_hist, C) in [0, 1].
        No general-purpose standardisation applied here; SANRouteNorm handles that.
        """
        return _compute_omega_spec_from_windows(hist_windows)

    def build_future_oracle_state(
        self,
        y_true: torch.Tensor,
        fut_windows: torch.Tensor,    # (B, P_fut, W, C)
        oracle_mu: torch.Tensor,
        oracle_std: torch.Tensor,
        sigma_min: float,
    ) -> torch.Tensor:
        """Compute raw omega_spec oracle for future patches.

        Returns (B, P_fut, C) in [0, 1].
        No general-purpose standardisation applied here; SANRouteNorm handles that.
        """
        return _compute_omega_spec_from_windows(fut_windows)

    def adapt_future_state(
        self,
        future_state_hat_raw: torch.Tensor,   # (B, P_fut, C)
    ) -> torch.Tensor:
        """Enforce the (0, 1) domain via sigmoid + clamp.

        Does NOT apply general-purpose standardisation; that is done by
        SANRouteNorm._normalize_patch_state() after this call.
        """
        return torch.sigmoid(future_state_hat_raw).clamp(1e-4, 1 - 1e-4)
