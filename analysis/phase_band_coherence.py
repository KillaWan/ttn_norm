"""phase_band_coherence.py — Offline diagnostic for patch-wise phase band coherence.

Analyses whether low / mid / high frequency phase shifts share a common
temporal displacement structure across future patches.

Pure torch implementation. No scipy. Not connected to training.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# A. Patch mean centering
# ---------------------------------------------------------------------------

def patch_mean_center(x_patch: torch.Tensor) -> torch.Tensor:
    """Subtract per-patch mean along the last (time) dimension.

    Args:
        x_patch: Tensor, shape (..., W)

    Returns:
        z_patch: Tensor, same shape as x_patch, mean-centered along dim=-1
    """
    return x_patch - x_patch.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# B. Phase rotation
# ---------------------------------------------------------------------------

def compute_phase_rotation(
    z_ref: torch.Tensor,
    z_true: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-bin complex phase rotation and amplitude weight.

    For each frequency bin k (including DC=0 in the returned tensors, but
    callers should skip bin 0 for band analysis):

        cross_k = X_true_k * conj(X_ref_k)
        R_k     = cross_k / (|cross_k| + eps)          # unit complex rotation
        w_k     = |X_true_k| * |X_ref_k|               # amplitude weight

    Args:
        z_ref:  (N, W) reference patch (history)
        z_true: (N, W) target patch (future)
        eps:    Numerical stability constant

    Returns:
        R:      (N, F) complex unit rotation per bin, F = W // 2 + 1
        w:      (N, F) real amplitude weight per bin
        X_ref:  (N, F) complex reference spectrum (includes DC at index 0)
        X_true: (N, F) complex true spectrum    (includes DC at index 0)
    """
    X_ref  = torch.fft.rfft(z_ref,  dim=-1)   # (N, F)
    X_true = torch.fft.rfft(z_true, dim=-1)   # (N, F)

    cross     = X_true * X_ref.conj()          # (N, F) complex
    cross_abs = cross.abs()                     # (N, F) real
    R = cross / (cross_abs + eps)               # (N, F) unit complex

    w = X_true.abs() * X_ref.abs()             # (N, F) real

    return R, w, X_ref, X_true


# ---------------------------------------------------------------------------
# C. Band coherence score
# ---------------------------------------------------------------------------

def band_coherence_score(
    R: torch.Tensor,
    w: torch.Tensor,
    bins: list[int],
    W: int,
    tau_grid: torch.Tensor,
    eps: float = 1e-8,
    amp_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-sample band coherence score C_band = max_tau C(tau).

    For each candidate shift tau:

        C(tau) = |sum_{k in bins} w_k * R_k * exp(j * 2π * k * tau / W)|
                 / (sum_{k in bins} w_k + eps)

    C_band   = max_tau  C(tau)
    tau_best = argmax_tau C(tau)   (returned as the float value, not the index)

    Requires at least 2 bins; a single bin trivially achieves C=1 for some tau.

    Args:
        R:        (N, F) complex unit phase rotation (from compute_phase_rotation)
        w:        (N, F) real amplitude weights
        bins:     list[int] of frequency bin indices (must exclude DC=0)
        W:        patch length (original time-domain length)
        tau_grid: (T,) float tensor of candidate temporal shifts to evaluate
        eps:      Numerical stability for denominator
        amp_eps:  Minimum band energy threshold for a sample to be "valid"

    Returns:
        C_band:     (N,) max coherence; NaN for invalid samples
        tau_best:   (N,) best tau value; NaN for invalid samples
        valid_mask: (N,) bool, True when band energy > amp_eps and len(bins) >= 2
    """
    N = R.shape[0]
    device = R.device

    # Require at least 2 bins
    if len(bins) < 2:
        nan_val = torch.full((N,), float("nan"), device=device)
        return nan_val, nan_val, torch.zeros(N, dtype=torch.bool, device=device)

    bins_t = torch.tensor(bins, dtype=torch.long, device=device)  # (B_len,)

    # Slice to band
    R_band = R[:, bins_t]   # (N, B_len) complex
    w_band = w[:, bins_t]   # (N, B_len) real

    # Per-sample total band energy (for validity check)
    sum_w = w_band.sum(dim=-1)                          # (N,) real
    valid_mask = sum_w > amp_eps                        # (N,) bool

    # Weighted complex rotation for each sample and bin
    wR = w_band * R_band                                # (N, B_len) complex

    # Build phase factors: exp(j * 2π * k * tau / W)
    # angle[t, b] = 2π * bins[b] * tau_grid[t] / W
    tau_grid = tau_grid.to(device=device, dtype=torch.float32)
    T = tau_grid.shape[0]
    angle = (
        2.0 * math.pi
        * bins_t.float().unsqueeze(0)    # (1,    B_len)
        * tau_grid.unsqueeze(1)           # (T,    1)
        / float(W)
    )  # (T, B_len)

    # exp_factor[t, b] = exp(j * angle[t, b])
    exp_factor = torch.polar(
        torch.ones(T, len(bins), device=device),
        angle,
    )  # (T, B_len) complex

    # Numerator: (N, B_len) @ (B_len, T) = (N, T) complex
    numer = wR @ exp_factor.T          # (N, T) complex

    # C[n, t] = |numer[n, t]| / (sum_w[n] + eps)
    C = numer.abs() / (sum_w.unsqueeze(1) + eps)       # (N, T) real

    # Max over tau
    C_band, tau_idx = C.max(dim=1)                     # (N,) each
    tau_best = tau_grid[tau_idx]                        # (N,) float

    # Mask invalid samples with NaN
    nan_val = torch.full((N,), float("nan"), device=device)
    C_band   = torch.where(valid_mask, C_band,   nan_val)
    tau_best = torch.where(valid_mask, tau_best, nan_val)

    return C_band, tau_best, valid_mask


# ---------------------------------------------------------------------------
# D. Phase-only oracle gain
# ---------------------------------------------------------------------------

def phase_only_oracle_gain(
    z_ref: torch.Tensor,
    z_true: torch.Tensor,
    bins: list[int],
    W: int,
    tau_best: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle diagnostic: does the best tau shift improve band-limited reconstruction?

    Constructs two band-limited estimates using the true future amplitude:
        Y_base_k = |X_true_k| * exp(j * angle(X_ref_k))
        Y_tau_k  = |X_true_k| * exp(j * (angle(X_ref_k) - 2π * k * tau_best / W))

    Target is the band-limited true signal:
        z_true_band = irfft(X_true restricted to bins)

    Then:
        mse_base = mean((irfft(Y_base) - z_true_band)^2)
        mse_tau  = mean((irfft(Y_tau)  - z_true_band)^2)
        gain     = (mse_base - mse_tau) / (mse_base + eps)

    This function is a pure oracle diagnostic and is never used in training.

    Args:
        z_ref:    (N, W) reference patch (history)
        z_true:   (N, W) true future patch
        bins:     list[int] of frequency bin indices in the band (excluding DC)
        W:        patch length
        tau_best: (N,) per-sample best tau from band_coherence_score
        eps:      Numerical stability

    Returns:
        mse_base: (N,) per-sample MSE without tau shift
        mse_tau:  (N,) per-sample MSE with tau shift
        gain:     (N,) relative improvement (mse_base - mse_tau) / (mse_base + eps)
    """
    N = z_ref.shape[0]
    device = z_ref.device

    X_ref  = torch.fft.rfft(z_ref,  dim=-1)   # (N, F)
    X_true = torch.fft.rfft(z_true, dim=-1)   # (N, F)

    bins_t = torch.tensor(bins, dtype=torch.long, device=device)  # (B_len,)

    amp_band   = X_true.abs()[:, bins_t]      # (N, B_len) real
    phase_ref  = X_ref.angle()[:, bins_t]     # (N, B_len) real

    # Phase shift per bin and per sample: 2π * k * tau_best / W
    # angle_shift[n, b] = 2π * bins[b] * tau_best[n] / W
    angle_shift = (
        2.0 * math.pi
        * bins_t.float().unsqueeze(0)    # (1,    B_len)
        * tau_best.unsqueeze(1)           # (N,    1)
        / float(W)
    )  # (N, B_len)

    # Build spectra (zero everywhere except the band bins)
    F_ = W // 2 + 1
    Y_base      = torch.zeros(N, F_, dtype=X_true.dtype, device=device)
    Y_tau       = torch.zeros(N, F_, dtype=X_true.dtype, device=device)
    X_true_band = torch.zeros(N, F_, dtype=X_true.dtype, device=device)

    Y_base[:, bins_t]      = torch.polar(amp_band, phase_ref)
    Y_tau[:, bins_t]       = torch.polar(amp_band, phase_ref - angle_shift)
    X_true_band[:, bins_t] = X_true[:, bins_t]

    # Reconstruct time-domain signals
    y_base      = torch.fft.irfft(Y_base,      n=W, dim=-1)   # (N, W)
    y_tau       = torch.fft.irfft(Y_tau,       n=W, dim=-1)   # (N, W)
    z_true_band = torch.fft.irfft(X_true_band, n=W, dim=-1)   # (N, W)

    # Per-sample MSE along time axis
    mse_base = ((y_base - z_true_band) ** 2).mean(dim=-1)     # (N,)
    mse_tau  = ((y_tau  - z_true_band) ** 2).mean(dim=-1)     # (N,)

    gain = (mse_base - mse_tau) / (mse_base + eps)            # (N,)

    return mse_base, mse_tau, gain


# ---------------------------------------------------------------------------
# E. Interpretation
# ---------------------------------------------------------------------------

def interpret_band_result(stats: dict) -> str:
    """Return a short interpretive label for a band's diagnostic statistics.

    Priority order:
      1. valid_ratio < 0.2         → "phase_unreliable_low_energy"
      2. C_med >= 0.7 & gain > 0.05 → "shared_tau_likely_useful"
      3. C_med >= 0.6 & gain <= 0.05→ "shared_tau_exists_but_low_forecast_value"
      4. C_med < 0.5               → "phase_shift_likely_scattered"
      5. fallback                  → "phase_shift_moderate"

    Args:
        stats: dict with at least:
            "coherence_median"    – float
            "oracle_gain_median"  – float
            "valid_ratio"         – float

    Returns:
        label string
    """
    coh_med  = stats.get("coherence_median",   float("nan"))
    gain_med = stats.get("oracle_gain_median", float("nan"))
    valid_r  = stats.get("valid_ratio",        0.0)

    import math as _math
    def _nan(v):
        return _math.isnan(v) if isinstance(v, float) else False

    if valid_r < 0.2:
        return "phase_unreliable_low_energy"
    if not _nan(coh_med) and not _nan(gain_med):
        if coh_med >= 0.7 and gain_med > 0.05:
            return "shared_tau_likely_useful"
        if coh_med >= 0.6 and gain_med <= 0.05:
            return "shared_tau_exists_but_low_forecast_value"
        if coh_med < 0.5:
            return "phase_shift_likely_scattered"
    return "phase_shift_moderate"
