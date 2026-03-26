"""LFAN – Low-Frequency Adaptive Normalization.

Structure
---------
Input side
  1. Fixed rFFT → keep first k_low bins → low_x
  2. high_x = x - low_x
  3. Energy-ratio score per segment: score_seg = rms(high_x_seg) / (rms(low_x_seg) + 1e-8)
     score < 1  → low-freq dominant (smooth) → quiet segment
     score > 1  → high-freq comparable to low-freq → burst segment
  4. Gate: remove_seg = clamp(remove_floor + (remove_quiet - remove_floor)
                              * sigmoid(gate_gamma * (gate_tau - score_seg)),
                              remove_burst, remove_quiet)
     score << gate_tau → remove ≈ remove_quiet;  score >> gate_tau → remove ≈ remove_burst
  6. removed_x = remove_t * low_x;  x_res = x - removed_x
  7. Background-weighted instance-norm on x_res → z_x
     weights = remove_t  (quiet segments dominate stats; burst leakage not inflating σ)

Output side
  8. low_head  : history low-freq coeff → future *removed* component coeff
  9. stat_head : same input → future residual (mu, log_sigma)
  10. y_hat = pred_removed_y + pred_sigma_y * y_norm + pred_mu_y

Interface (compatible with TTNModel generic fallback)
-----------------------------------------------------
  normalize(x)        -> z_x
  denormalize(y_norm) -> y_hat
  loss(y_true)        -> scalar aux loss
  forward(x)          -> normalize(x)
  get_debug_scalars() -> dict[str, float]
  get_debug_text()    -> str
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _rfft_low(signal: Tensor, k_low: int):
    """rFFT low-pass decomposition.

    Returns:
        full_fft  : (B, F, C) complex
        low_fft   : (B, F, C) complex  – first k_low bins kept, rest 0
        low_time  : (B, T, C) float
    """
    B, T, C = signal.shape
    full_fft = torch.fft.rfft(signal, dim=1)
    low_fft  = full_fft.clone()
    if k_low < low_fft.shape[1]:
        low_fft[:, k_low:, :] = 0.0
    low_time = torch.fft.irfft(low_fft, n=T, dim=1)
    return full_fft, low_fft, low_time


def _smooth_time(x: Tensor, kernel_size: int) -> Tensor:
    """Depthwise moving-average along time (reflect padding)."""
    B, T, C = x.shape
    x_t   = x.permute(0, 2, 1)
    pad   = kernel_size // 2
    kern  = x_t.new_ones(C, 1, kernel_size) / kernel_size
    x_pad = F.pad(x_t, (pad, pad), mode="reflect")
    out   = F.conv1d(x_pad, kern, groups=C)
    return out.permute(0, 2, 1)


def _burst_and_gate(high_signal: Tensor, low_signal: Tensor,
                    gate_segments: int, remove_quiet: float,
                    remove_burst: float, gate_gamma: float,
                    gate_tau: float, remove_floor: float):
    """Energy-ratio burst detection + segment gate.

    Score = rms(high_signal_seg) / (rms(low_signal_seg) + eps)
    score << gate_tau → quiet segment → remove ≈ remove_quiet
    score >> gate_tau → burst segment → remove ≈ remove_burst

    Args:
        high_signal : (B, T, C) – high-frequency component (high_x or high_y)
        low_signal  : (B, T, C) – low-frequency component  (low_x or low_y)

    Returns:
        score_t    : (B, T, C) piecewise-constant energy-ratio score
        remove_t   : (B, T, C) piecewise-constant removal fraction
        keep_t     : (B, T, C)
        remove_seg : (B, S, C) per-segment removal fraction
    """
    B, T, C = high_signal.shape

    S       = gate_segments
    seg_len = max(1, T // S)
    T_use   = seg_len * S

    def _pad_to_T_use(t: Tensor) -> Tensor:
        if t.shape[1] >= T_use:
            return t[:, :T_use, :]
        extra = t[:, -1:, :].expand(B, T_use - t.shape[1], C)
        return torch.cat([t, extra], dim=1)

    high_use = _pad_to_T_use(high_signal)   # (B, T_use, C)
    low_use  = _pad_to_T_use(low_signal)    # (B, T_use, C)

    # ── Per-segment energy-ratio score ───────────────────────────────────────
    # rms(high_seg) / (rms(low_seg) + eps)  ∈ [0, ∞)
    # Interpretable threshold: gate_tau=1 means "high ≈ low energy"
    high_pow_seg = high_use.reshape(B, S, seg_len, C).pow(2).mean(dim=2)  # (B, S, C)
    low_pow_seg  = low_use.reshape( B, S, seg_len, C).pow(2).mean(dim=2)  # (B, S, C)
    rms_high_seg = high_pow_seg.sqrt()
    rms_low_seg  = low_pow_seg.sqrt()
    score_seg    = rms_high_seg / (rms_low_seg + 1e-8)                    # (B, S, C)

    # ── Gate: threshold at gate_tau, clamped to [remove_burst, remove_quiet] ──
    remove_seg = (
        remove_floor
        + (remove_quiet - remove_floor)
        * torch.sigmoid(gate_gamma * (gate_tau - score_seg))
    ).clamp(remove_burst, remove_quiet)  # (B, S, C)

    # ── Tile back to T ────────────────────────────────────────────────────────
    score_t  = score_seg.repeat_interleave(seg_len, dim=1)   # (B, T_use, C)
    remove_t = remove_seg.repeat_interleave(seg_len, dim=1)  # (B, T_use, C)

    def _fit_to_T(t: Tensor) -> Tensor:
        if t.shape[1] > T:
            return t[:, :T, :]
        elif t.shape[1] < T:
            extra = t[:, -1:, :].expand(B, T - t.shape[1], C)
            return torch.cat([t, extra], dim=1)
        return t

    score_t  = _fit_to_T(score_t)
    remove_t = _fit_to_T(remove_t)
    keep_t   = 1.0 - remove_t
    return score_t, remove_t, keep_t, remove_seg


# ---------------------------------------------------------------------------
# LFAN
# ---------------------------------------------------------------------------

class LFAN(nn.Module):
    def __init__(
        self,
        seq_len:         int,
        pred_len:        int,
        enc_in:          int,
        k_low:           int   = 8,
        burst_smooth:    int   = 5,
        burst_mix:       float = 0.7,
        gate_segments:   int   = 16,
        remove_quiet:    float = 0.98,
        remove_burst:    float = 0.05,
        gate_gamma:      float = 3.0,
        gate_tau:        float = 1.0,
        remove_floor:    float = 0.0,
        sigma_min:       float = 1e-5,
        hidden_dim:      int   = 64,
        loss_low_coeff:  float = 1.0,
        loss_low_shape:  float = 0.5,
        loss_mu:         float = 0.2,
        loss_sigma:      float = 0.2,
        loss_res:        float = 0.2,
        fan_equiv:       bool  = False,
        **kwargs,
    ):
        super().__init__()

        # ── Validation ───────────────────────────────────────────────────────
        if k_low < 1:
            raise ValueError(f"k_low must be >= 1, got {k_low}")
        if burst_smooth < 1 or burst_smooth % 2 == 0:
            raise ValueError(f"burst_smooth must be a positive odd integer, got {burst_smooth}")
        if gate_segments < 1:
            raise ValueError(f"gate_segments must be >= 1, got {gate_segments}")
        if gate_gamma <= 0:
            raise ValueError(f"gate_gamma must be > 0, got {gate_gamma}")
        if gate_tau <= 0:
            raise ValueError(f"gate_tau must be > 0, got {gate_tau}")
        if not (0 <= remove_floor < remove_burst < remove_quiet <= 1):
            raise ValueError(
                f"require 0 <= remove_floor < remove_burst < remove_quiet <= 1, "
                f"got remove_floor={remove_floor} remove_burst={remove_burst} remove_quiet={remove_quiet}"
            )

        self.seq_len        = seq_len
        self.pred_len       = pred_len
        self.enc_in         = enc_in
        self.k_low          = k_low
        self.burst_smooth   = burst_smooth
        self.burst_mix      = burst_mix
        self.gate_segments  = gate_segments
        self.remove_quiet   = remove_quiet
        self.remove_burst   = remove_burst
        self.gate_gamma     = gate_gamma
        self.gate_tau       = gate_tau
        self.remove_floor   = remove_floor
        self.sigma_min      = sigma_min
        self.loss_low_coeff = loss_low_coeff
        self.loss_low_shape = loss_low_shape
        self.loss_mu        = loss_mu
        self.loss_sigma     = loss_sigma
        self.loss_res       = loss_res
        self.fan_equiv      = fan_equiv

        # ── Heads ────────────────────────────────────────────────────────────
        if fan_equiv:
            # FAN-equivalent: low_head_fan takes (low_x, x) per channel → pred_len
            # Input: (B, C, 2*seq_len),  Output: (B, C, pred_len)
            self.low_head_fan = nn.Sequential(
                nn.Linear(2 * seq_len, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, pred_len),
            )
            # stat_head still defined but unused in fan_equiv (avoids param-count skew)
            _head_in_std = 2 * k_low + 6
            self.low_head  = nn.Sequential(
                nn.Linear(_head_in_std, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 2 * k_low),
            )
            self.stat_head = nn.Sequential(
                nn.Linear(_head_in_std, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            # Original LFAN heads
            _head_in = 2 * k_low + 6
            # low_head → predicts future *removed component* coefficients
            self.low_head = nn.Sequential(
                nn.Linear(_head_in, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * k_low),
            )
            # stat_head → predicts future residual (mu, log_sigma)
            self.stat_head = nn.Sequential(
                nn.Linear(_head_in, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )

        # ── Cache fields ─────────────────────────────────────────────────────
        # normalize() caches
        self._last_lfan_low_x:                   Optional[Tensor] = None
        self._last_lfan_high_x:                  Optional[Tensor] = None
        self._last_lfan_score_t:                 Optional[Tensor] = None
        self._last_lfan_remove_t:                Optional[Tensor] = None

        self._last_lfan_keep_t:                  Optional[Tensor] = None
        self._last_lfan_removed_x:               Optional[Tensor] = None
        self._last_lfan_mu_x:                    Optional[Tensor] = None
        self._last_lfan_sigma_x:                 Optional[Tensor] = None
        self._last_lfan_low_bins:                list             = list(range(k_low))
        self._last_lfan_low_energy_ratio_x:      float = float("nan")
        self._last_lfan_high_energy_ratio_x:     float = float("nan")
        self._last_lfan_removed_energy_ratio_x:  float = float("nan")
        self._last_lfan_gate_segment_values:     Optional[Tensor] = None

        self._last_lfan_pred_removed_coeff:      Optional[Tensor] = None
        self._last_lfan_pred_mu_y:               Optional[Tensor] = None
        self._last_lfan_pred_log_sigma_y:        Optional[Tensor] = None
        self._last_lfan_pred_res_norm:           Optional[Tensor] = None

        # denormalize() caches
        self._last_lfan_pred_removed_y:          Optional[Tensor] = None
        self._last_lfan_pred_sigma_y:            Optional[Tensor] = None
        self._last_lfan_y_hat:                   Optional[Tensor] = None

        # loss() caches
        self._last_lfan_low_y:                   Optional[Tensor] = None
        self._last_lfan_removed_y:               Optional[Tensor] = None
        self._last_lfan_res_y:                   Optional[Tensor] = None
        self._last_lfan_true_removed_coeff:      Optional[Tensor] = None
        self._last_lfan_true_mu_y:               Optional[Tensor] = None
        self._last_lfan_true_sigma_y:            Optional[Tensor] = None
        self._last_lfan_true_res_norm:           Optional[Tensor] = None
        self._last_lfan_removed_energy_ratio_y:  float = float("nan")
        self._last_lfan_pred_removed_energy_ratio: float = float("nan")
        self._last_lfan_true_removed_energy_ratio: float = float("nan")
        self._last_lfan_removed_coeff_mae:       float = float("nan")
        self._last_lfan_removed_shape_mae:       float = float("nan")
        self._last_lfan_bin_mae_list:            list  = [float("nan")] * k_low

        # fan_equiv mode caches
        self._cache_x:            Optional[Tensor] = None
        self._cache_low_x:        Optional[Tensor] = None
        self._cache_removed_x:    Optional[Tensor] = None
        self._cache_res_x:        Optional[Tensor] = None
        self._cache_pred_low:     Optional[Tensor] = None
        self._fan_pred_residual:  Optional[Tensor] = None
        self._fan_y_hat:          Optional[Tensor] = None

    # ------------------------------------------------------------------
    def normalize(self, x: Tensor) -> Tensor:
        """(B, T, C) → backbone input tensor."""
        B, T, C = x.shape

        # ── 1. Low-pass (shared) ─────────────────────────────────────────────
        full_fft_x, _, low_x = _rfft_low(x, self.k_low)
        high_x = x - low_x

        # ── FAN-equivalent branch ─────────────────────────────────────────────
        if self.fan_equiv:
            removed_x = low_x           # remove all low freq
            x_res     = x - low_x       # backbone sees high-freq residual (no inst-norm)

            # low_head_fan: input = (low_x, x) per channel → pred_len
            lx_t    = low_x.permute(0, 2, 1)                          # (B, C, T)
            x_t     = x.permute(0, 2, 1)                              # (B, C, T)
            head_in = torch.cat([lx_t, x_t], dim=-1)                  # (B, C, 2T)
            pred_low = self.low_head_fan(head_in).permute(0, 2, 1)    # (B, pred_len, C)

            # minimal caches required
            self._cache_x         = x
            self._cache_low_x     = low_x
            self._cache_removed_x = removed_x
            self._cache_res_x     = x_res
            self._cache_pred_low  = pred_low
            self._fan_pred_residual = None   # reset; set in denormalize()

            # also keep shared LFAN fields as nan so get_debug_scalars is safe
            self._last_lfan_low_x   = low_x
            self._last_lfan_high_x  = high_x
            self._last_lfan_score_t  = None
            self._last_lfan_remove_t = None
            self._last_lfan_keep_t   = None
            sum_x2 = x.pow(2).sum(dim=1).clamp(min=1e-8)
            self._last_lfan_low_energy_ratio_x     = float((low_x.pow(2).sum(dim=1) / sum_x2).mean().item())
            self._last_lfan_high_energy_ratio_x    = float((high_x.pow(2).sum(dim=1) / sum_x2).mean().item())
            self._last_lfan_removed_energy_ratio_x = float((removed_x.pow(2).sum(dim=1) / sum_x2).mean().item())
            self._last_lfan_gate_segment_values    = None
            self._last_lfan_mu_x     = None
            self._last_lfan_sigma_x  = None

            return x_res

        # ── Original LFAN branch ──────────────────────────────────────────────

        # ── 2. Score + segment gate ──────────────────────────────────────────
        score_t, remove_t, keep_t, remove_seg = _burst_and_gate(
            high_x, low_x,
            self.gate_segments, self.remove_quiet, self.remove_burst,
            self.gate_gamma, self.gate_tau, self.remove_floor,
        )

        # ── 3. Remove + background-weighted instance norm ────────────────────
        removed_x = remove_t * low_x
        x_res     = x - removed_x

        # Use remove_t as weights: quiet segments (remove_t ≈ remove_quiet) dominate
        # the statistics; burst/event segments (remove_t ≈ remove_burst) contribute
        # little → their leakage into x_res doesn't inflate sigma_x.
        w_x     = remove_t                                              # (B, T, C)
        w_x_sum = w_x.sum(dim=1, keepdim=True).clamp(min=1e-8)
        mu_x    = (w_x * x_res).sum(dim=1, keepdim=True) / w_x_sum    # (B, 1, C)
        var_x   = (w_x * (x_res - mu_x).pow(2)).sum(dim=1, keepdim=True) / w_x_sum
        sigma_x = var_x.sqrt().clamp(min=self.sigma_min)               # (B, 1, C)
        z_x     = (x_res - mu_x) / sigma_x

        # ── 4. Per-(B,C) features for heads ─────────────────────────────────
        sum_x2 = x.pow(2).sum(dim=1).clamp(min=1e-8)
        f1_bc  = low_x.pow(2).sum(dim=1)    / sum_x2   # low energy ratio
        f2_bc  = high_x.pow(2).sum(dim=1)   / sum_x2   # high energy ratio
        f3_bc  = removed_x.pow(2).sum(dim=1) / sum_x2  # removed energy ratio
        f4_bc  = remove_t.mean(dim=1)                   # mean remove per (B,C)
        f5_bc  = score_t.mean(dim=1)                    # mean score per (B,C)
        f6_bc  = sigma_x.squeeze(1)                     # sigma_x per (B,C)

        # ── 5. Head forward ──────────────────────────────────────────────────
        low_coeff_x  = full_fft_x[:, :self.k_low, :]           # (B, k, C) complex
        real_c       = low_coeff_x.real.permute(0, 2, 1)        # (B, C, k)
        imag_c       = low_coeff_x.imag.permute(0, 2, 1)        # (B, C, k)
        scalar_feats = torch.stack([f1_bc, f2_bc, f3_bc, f4_bc, f5_bc, f6_bc], dim=-1)
        head_in      = torch.cat([real_c, imag_c, scalar_feats], dim=-1)  # (B, C, 2k+6)

        pred_removed_coeff = self.low_head(head_in)              # (B, C, 2*k_low)
        stat_out           = self.stat_head(head_in)             # (B, C, 2)
        pred_mu_y          = stat_out[:, :, 0].unsqueeze(1)      # (B, 1, C)
        pred_log_sig_y     = stat_out[:, :, 1].unsqueeze(1)      # (B, 1, C)

        # ── 6. Cache ─────────────────────────────────────────────────────────
        self._last_lfan_low_x                  = low_x
        self._last_lfan_high_x                 = high_x
        self._last_lfan_score_t                = score_t
        self._last_lfan_remove_t               = remove_t
        self._last_lfan_keep_t                 = keep_t
        self._last_lfan_removed_x              = removed_x
        self._last_lfan_mu_x                   = mu_x
        self._last_lfan_sigma_x                = sigma_x
        self._last_lfan_low_bins               = list(range(self.k_low))
        self._last_lfan_low_energy_ratio_x     = float(f1_bc.mean().item())
        self._last_lfan_high_energy_ratio_x    = float(f2_bc.mean().item())
        self._last_lfan_removed_energy_ratio_x = float(f3_bc.mean().item())
        self._last_lfan_gate_segment_values    = remove_seg.mean(dim=(0, 2)).detach()

        self._last_lfan_pred_removed_coeff  = pred_removed_coeff
        self._last_lfan_pred_mu_y           = pred_mu_y
        self._last_lfan_pred_log_sigma_y    = pred_log_sig_y
        self._last_lfan_pred_res_norm       = None  # reset; set in denormalize()

        return z_x

    # ------------------------------------------------------------------
    def denormalize(self, y_norm: Tensor) -> Tensor:
        """(B, O, C) → (B, O, C) reconstructed forecast."""
        # ── FAN-equivalent branch ─────────────────────────────────────────────
        if self.fan_equiv:
            # y_norm is the backbone output = pred_residual (high-freq prediction)
            pred_low      = self._cache_pred_low       # (B, pred_len, C)
            y_hat         = pred_low + y_norm
            self._fan_pred_residual = y_norm
            self._fan_y_hat         = y_hat
            return y_hat

        # ── Original LFAN branch ──────────────────────────────────────────────
        self._last_lfan_pred_res_norm = y_norm

        B, O, C      = y_norm.shape
        pred_sigma_y = self._last_lfan_pred_log_sigma_y.exp().clamp(min=self.sigma_min)

        # ── Reconstruct future removed component ─────────────────────────────
        pred_rc   = self._last_lfan_pred_removed_coeff         # (B, C, 2k)
        real_part = pred_rc[:, :, :self.k_low]                 # (B, C, k)
        imag_part = pred_rc[:, :, self.k_low:]                 # (B, C, k)
        complex_c = torch.complex(real_part, imag_part).permute(0, 2, 1)  # (B, k, C)

        F_pred    = self.pred_len // 2 + 1
        full_pred = torch.zeros(B, F_pred, C,
                                dtype=complex_c.dtype, device=y_norm.device)
        k_fill    = min(self.k_low, F_pred)
        full_pred[:, :k_fill, :] = complex_c[:, :k_fill, :]
        pred_removed_y = torch.fft.irfft(full_pred, n=self.pred_len, dim=1)  # (B, O, C)

        pred_res = pred_sigma_y * y_norm + self._last_lfan_pred_mu_y          # (B, O, C)
        y_hat    = pred_removed_y + pred_res

        self._last_lfan_pred_removed_y = pred_removed_y
        self._last_lfan_pred_sigma_y   = pred_sigma_y
        self._last_lfan_y_hat          = y_hat
        return y_hat

    # ------------------------------------------------------------------
    def loss(self, y_true: Tensor) -> Tensor:
        """Auxiliary supervision loss.  Must be called after denormalize()."""
        # ── FAN-equivalent branch ─────────────────────────────────────────────
        if self.fan_equiv:
            if self._fan_pred_residual is None:
                raise RuntimeError(
                    "LFAN(fan_equiv=True).loss() called before denormalize()."
                )
            _, _, low_y = _rfft_low(y_true, self.k_low)
            res_y       = y_true - low_y
            pred_low    = self._cache_pred_low
            pred_res    = self._fan_pred_residual

            low_loss = F.mse_loss(pred_low, low_y)
            res_loss = F.mse_loss(pred_res, res_y)

            self._last_lfan_low_y   = low_y
            self._last_lfan_res_y   = res_y
            return low_loss + res_loss

        # ── Original LFAN branch ──────────────────────────────────────────────
        if self._last_lfan_pred_res_norm is None:
            raise RuntimeError(
                "LFAN.loss() called before denormalize(). "
                "Call denormalize() first in the same forward pass."
            )

        B, O, C = y_true.shape

        # ── True decomposition: same pipeline as normalize() ─────────────────
        full_fft_y, _, low_y = _rfft_low(y_true, self.k_low)
        high_y = y_true - low_y

        _, remove_y, _, _ = _burst_and_gate(
            high_y, low_y,
            self.gate_segments, self.remove_quiet, self.remove_burst,
            self.gate_gamma, self.gate_tau, self.remove_floor,
        )

        removed_y = remove_y * low_y
        res_y     = y_true - removed_y

        # Background-weighted stats (mirror normalize()):
        w_y     = remove_y
        w_y_sum = w_y.sum(dim=1, keepdim=True).clamp(min=1e-8)
        true_mu_y    = (w_y * res_y).sum(dim=1, keepdim=True) / w_y_sum
        true_var_y   = (w_y * (res_y - true_mu_y).pow(2)).sum(dim=1, keepdim=True) / w_y_sum
        true_sigma_y = true_var_y.sqrt().clamp(min=self.sigma_min)
        true_log_sig_y = true_sigma_y.log()
        true_res_norm  = (res_y - true_mu_y) / true_sigma_y

        # ── True removed coefficients ─────────────────────────────────────────
        full_fft_rem   = torch.fft.rfft(removed_y, dim=1)
        true_rem_bins  = full_fft_rem[:, :self.k_low, :]        # (B, k, C)
        true_real_r    = true_rem_bins.real.permute(0, 2, 1)    # (B, C, k)
        true_imag_r    = true_rem_bins.imag.permute(0, 2, 1)    # (B, C, k)
        true_removed_coeff = torch.cat([true_real_r, true_imag_r], dim=-1)  # (B, C, 2k)

        # ── Loss terms ───────────────────────────────────────────────────────
        mse = F.mse_loss
        pred_rc        = self._last_lfan_pred_removed_coeff
        pred_removed_y = self._last_lfan_pred_removed_y
        pred_mu_y      = self._last_lfan_pred_mu_y
        pred_log_sig_y = self._last_lfan_pred_log_sigma_y
        pred_res_norm  = self._last_lfan_pred_res_norm
        y_hat          = self._last_lfan_y_hat

        removed_coeff_loss = mse(pred_rc,         true_removed_coeff)
        removed_shape_loss = mse(pred_removed_y,  removed_y)
        mu_loss            = mse(pred_mu_y,        true_mu_y)
        sigma_loss         = mse(pred_log_sig_y,   true_log_sig_y)
        res_loss           = mse(pred_res_norm,    true_res_norm)

        total = (
            self.loss_low_coeff * removed_coeff_loss
            + self.loss_low_shape * removed_shape_loss
            + self.loss_mu         * mu_loss
            + self.loss_sigma      * sigma_loss
            + self.loss_res        * res_loss
        )

        # ── Cache ─────────────────────────────────────────────────────────────
        self._last_lfan_low_y              = low_y
        self._last_lfan_removed_y          = removed_y
        self._last_lfan_res_y              = res_y
        self._last_lfan_true_removed_coeff = true_removed_coeff
        self._last_lfan_true_mu_y          = true_mu_y
        self._last_lfan_true_sigma_y       = true_sigma_y
        self._last_lfan_true_res_norm      = true_res_norm

        with torch.no_grad():
            sum_y2  = y_true.pow(2).sum(dim=1).clamp(min=1e-8)
            true_er = float((removed_y.pow(2).sum(dim=1) / sum_y2).mean().item())
            sum_yh2 = y_hat.pow(2).sum(dim=1).clamp(min=1e-8)
            pred_er = float((pred_removed_y.pow(2).sum(dim=1) / sum_yh2).mean().item())

            self._last_lfan_removed_energy_ratio_y    = true_er
            self._last_lfan_pred_removed_energy_ratio  = pred_er
            self._last_lfan_true_removed_energy_ratio  = true_er
            self._last_lfan_removed_coeff_mae = float(
                (pred_rc - true_removed_coeff).abs().mean().item()
            )
            self._last_lfan_removed_shape_mae = float(
                (pred_removed_y - removed_y).abs().mean().item()
            )
            bin_mae = []
            for i in range(self.k_low):
                r_mae = float((pred_rc[:, :, i]              - true_removed_coeff[:, :, i]).abs().mean().item())
                i_mae = float((pred_rc[:, :, i + self.k_low] - true_removed_coeff[:, :, i + self.k_low]).abs().mean().item())
                bin_mae.append((r_mae + i_mae) * 0.5)
            self._last_lfan_bin_mae_list = bin_mae

        return total

    # ------------------------------------------------------------------
    def get_debug_scalars(self) -> dict:
        """Return epoch-averageable scalar diagnostics."""
        nan = float("nan")

        score_t  = self._last_lfan_score_t
        remove_t = self._last_lfan_remove_t
        keep_t   = self._last_lfan_keep_t
        seg_vals = self._last_lfan_gate_segment_values
        mu_x     = self._last_lfan_mu_x
        sigma_x  = self._last_lfan_sigma_x

        if score_t is not None and remove_t is not None and keep_t is not None:
            score_threshold   = self.gate_tau
            high_score_mask   = score_t > score_threshold
            high_score_frac   = float(high_score_mask.float().mean().item())
            low_score_mask    = ~high_score_mask

            keep_burst_mean    = float(keep_t[high_score_mask].mean().item())   if high_score_mask.any()  else nan
            keep_nonburst_mean = float(keep_t[low_score_mask].mean().item())    if low_score_mask.any()   else nan
            remove_burst_mean  = float(remove_t[high_score_mask].mean().item()) if high_score_mask.any()  else nan
            remove_nonburst    = float(remove_t[low_score_mask].mean().item())  if low_score_mask.any()   else nan

            sflat  = score_t.detach().float().flatten()
            rflat  = remove_t.detach().float().flatten()
            sm, rm = sflat.mean(), rflat.mean()
            num    = ((sflat - sm) * (rflat - rm)).mean()
            denom  = sflat.std().clamp(min=1e-12) * rflat.std().clamp(min=1e-12)
            corr_sr = float((num / denom).item())

            T           = remove_t.shape[1]
            h10         = max(1, T // 10)
            m_start     = T // 2 - h10 // 2
            remove_head = float(remove_t[:, :h10, :].mean().item())
            remove_mid  = float(remove_t[:, m_start:m_start + h10, :].mean().item())
            remove_tail = float(remove_t[:, T - h10:, :].mean().item())

            remove_mean = float(remove_t.mean().item())
            remove_min  = float(remove_t.min().item())
            remove_max  = float(remove_t.max().item())
            keep_mean   = float(keep_t.mean().item())
            score_mean  = float(score_t.mean().item())
            score_std   = float(score_t.std().item())
        else:
            score_threshold = self.gate_tau
            high_score_frac = nan
            keep_burst_mean = keep_nonburst_mean = nan
            remove_burst_mean = remove_nonburst = corr_sr = nan
            remove_head = remove_mid = remove_tail = nan
            remove_mean = remove_min = remove_max = nan
            keep_mean = score_mean = score_std = nan

        mu_x_abs_mean = float(mu_x.abs().mean().item())  if mu_x    is not None else nan
        sigma_x_mean  = float(sigma_x.mean().item())     if sigma_x is not None else nan

        d = {
            "lf_low_energy_ratio_x":       self._last_lfan_low_energy_ratio_x,
            "lf_high_energy_ratio_x":      self._last_lfan_high_energy_ratio_x,
            "lf_removed_energy_ratio_x":   self._last_lfan_removed_energy_ratio_x,
            "lf_removed_energy_ratio_y":   self._last_lfan_removed_energy_ratio_y,
            "lf_pred_removed_energy_ratio": self._last_lfan_pred_removed_energy_ratio,
            "lf_true_removed_energy_ratio": self._last_lfan_true_removed_energy_ratio,
            "lf_remove_mean":              remove_mean,
            "lf_remove_min":               remove_min,
            "lf_remove_max":               remove_max,
            "lf_keep_mean":                keep_mean,
            "lf_score_mean":               score_mean,
            "lf_score_std":                score_std,
            "lf_score_threshold":          float(score_threshold),
            "lf_high_score_fraction":      high_score_frac,
            "lf_keep_burst_mean":          keep_burst_mean,
            "lf_keep_nonburst_mean":       keep_nonburst_mean,
            "lf_remove_burst_mean":        remove_burst_mean,
            "lf_remove_nonburst_mean":     remove_nonburst,
            "lf_corr_score_remove":        corr_sr,
            "lf_mu_x_abs_mean":            mu_x_abs_mean,
            "lf_sigma_x_mean":             sigma_x_mean,
            "lf_removed_coeff_mae":        self._last_lfan_removed_coeff_mae,
            "lf_removed_shape_mae":        self._last_lfan_removed_shape_mae,
            "lf_freq_start":               float(0),
            "lf_freq_end":                 float(self.k_low - 1),
            "lf_remove_head_mean":         remove_head,
            "lf_remove_mid_mean":          remove_mid,
            "lf_remove_tail_mean":         remove_tail,
        }

        for i, v in enumerate(self._last_lfan_bin_mae_list):
            d[f"lf_bin_mae_{i}"] = v

        if seg_vals is not None:
            for i in range(min(self.gate_segments, len(seg_vals))):
                d[f"lf_remove_seg_{i}"] = float(seg_vals[i].item())
        else:
            for i in range(self.gate_segments):
                d[f"lf_remove_seg_{i}"] = nan

        return d

    # ------------------------------------------------------------------
    def get_debug_text(self) -> str:
        """Single-line diagnostic: highlights burst vs quiet removal."""
        d   = self.get_debug_scalars()
        seg = self._last_lfan_gate_segment_values

        def _f(v, fmt=".4f"):
            return format(v, fmt) if not math.isnan(v) else "nan"

        segs_str = (
            "[" + ",".join(_f(float(seg[i].item())) for i in range(len(seg))) + "]"
            if seg is not None else "[]"
        )

        return (
            f"low_bins=[0..{self.k_low - 1}]"
            f" low_ratio_x={_f(d['lf_low_energy_ratio_x'])}"
            f" high_ratio_x={_f(d['lf_high_energy_ratio_x'])}"
            f" removed_ratio_x={_f(d['lf_removed_energy_ratio_x'])}"
            f" removed_ratio_y={_f(d['lf_removed_energy_ratio_y'])}"
            f" score_mean={_f(d['lf_score_mean'])}"
            f" score_tau={_f(float(self.gate_tau))}"
            f" high_score_frac={_f(d['lf_high_score_fraction'])}"
            f" keep_high_score={_f(d['lf_keep_burst_mean'])}"
            f" keep_low_score={_f(d['lf_keep_nonburst_mean'])}"
            f" remove_high_score={_f(d['lf_remove_burst_mean'])}"
            f" remove_low_score={_f(d['lf_remove_nonburst_mean'])}"
            f" corr_score_remove={_f(d['lf_corr_score_remove'])}"
            f" pred_removed_ratio={_f(d['lf_pred_removed_energy_ratio'])}"
            f" true_removed_ratio={_f(d['lf_true_removed_energy_ratio'])}"
            f" remove_head={_f(d['lf_remove_head_mean'])}"
            f" remove_mid={_f(d['lf_remove_mid_mean'])}"
            f" remove_tail={_f(d['lf_remove_tail_mean'])}"
            f" segments={segs_str}"
        )

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, mode: str = "n") -> Tensor:
        if mode == "d":
            return self.denormalize(x)
        return self.normalize(x)
