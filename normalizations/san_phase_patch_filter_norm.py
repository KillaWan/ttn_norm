"""SANPhasePatchFilterNorm — patch-local phase filter extension of SANRouteNorm.

Inherits from SANRouteNorm with route_path="none", route_state="none",
use_phase=True forced.

Adds a learnable per-patch-per-channel scalar alpha that blends the
backbone output with a patch-local FFT-filtered version.  The filter
kernel is derived from the (gated) phase predictor rotation via irfft,
ensuring exact identity when the phase rotation is trivial.

When alpha=0 (phase_patch_filter_init≈0), output is identical to the
parent's gauge+phase path.  Alpha is bounded to [0, phase_patch_filter_max]
via sigmoid parameterization.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .san_route_norm import SANRouteNorm


class SANPhasePatchFilterNorm(SANRouteNorm):
    """Patch-local phase filter correction on top of the pure SAN + phase baseline.

    The parent (SANRouteNorm with use_phase=True) applies:
      normalize : z = x - patch_mean  (mean-only normalization per patch)
      denormalize: y_phase = phase_rotate(y_norm) + mu_fut

    This subclass replaces the denormalize output with:
      y_corr = y_phase + alpha * (fft_filter(y_phase) - y_phase)
      y_out  = y_corr + mu_fut

    The FFT filter uses circular convolution so that a trivial (all-1)
    frequency response gives y_filt == y_phase exactly (identity property).

    Args:
        phase_patch_filter_init:  Initial value for alpha (≈ sigmoid(logit)*max).
        phase_patch_filter_max:   Upper bound for alpha.
        phase_patch_filter_eps:   Small epsilon for numerical stability.
        **kwargs:                 Forwarded to SANRouteNorm.  route_path,
                                  route_state, and use_phase are overridden.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        enc_in: int,
        phase_patch_filter_init: float = 0.02,
        phase_patch_filter_max: float = 1.0,
        phase_patch_filter_eps: float = 1e-6,
        **kwargs,
    ):
        # Force pure-SAN + phase path; callers must not override these.
        kwargs["route_path"]  = "none"
        kwargs["route_state"] = "none"
        kwargs["use_phase"]   = True
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            period_len=period_len,
            enc_in=enc_in,
            **kwargs,
        )

        self.phase_patch_filter_max = float(phase_patch_filter_max)
        self.phase_patch_filter_eps = float(phase_patch_filter_eps)

        # alpha = phase_patch_filter_max * sigmoid(logit)
        # Initialize so that alpha ≈ phase_patch_filter_init.
        r = float(phase_patch_filter_init) / float(phase_patch_filter_max)
        r = max(min(r, 1.0 - 1e-6), 1e-6)
        init_logit = math.log(r / (1.0 - r))

        # shape: (pred_stat_len, enc_in) = (P_fut, C)
        self.phase_patch_filter_logit = nn.Parameter(
            torch.full((self.pred_stat_len, enc_in), init_logit)
        )

        # Diagnostic cache (separate from parent fields to avoid collision)
        self._last_filter_alpha_mean:           float = 0.0
        self._last_filter_alpha_max:            float = 0.0
        self._last_filter_effect_rms:           float = 0.0
        self._last_filter_effect_ratio:         float = 0.0
        self._last_filter_kernel_l1:            float = 0.0
        self._last_filter_kernel_l2:            float = 0.0
        self._last_filter_kernel_max_ratio:     float = 0.0
        self._last_filter_kernel_center_ratio:  float = 0.0
        self._last_filter_pred_residual_rms:    float = 0.0
        self._last_filter_residual_rms_ratio:   float = 0.0

    # ------------------------------------------------------------------
    # Parameter accessors for optimizer
    # ------------------------------------------------------------------

    def parameters_phase_filter(self) -> list:
        """Stage 2 optimizer target — phase filter logit only."""
        return [self.phase_patch_filter_logit]

    # ------------------------------------------------------------------
    # denormalize override
    # ------------------------------------------------------------------

    def denormalize(
        self,
        y_norm: torch.Tensor,
        y_true_oracle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Override pure-SAN + use_phase + T==pred_len path to add filter.

        All other paths (timeapn, generic routes, T != pred_len fallback,
        pure SAN without phase) fall through to the parent unchanged.
        """
        T = y_norm.shape[1]

        if (
            self.use_phase
            and self._phase_rot_fut_hat is not None
            and T == self.pred_len
            and self._pred_time_stats is not None
        ):
            return self._denormalize_phase_filter(y_norm)

        return super().denormalize(y_norm, y_true_oracle)

    def _denormalize_phase_filter(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Pure-SAN + phase + filter denormalization path."""
        B, _, C = y_norm.shape
        P_fut   = self.pred_stat_len
        W       = self.window_len
        K       = self._phase_k
        mu_fut  = self._base_mu_fut_hat          # (B, P_fut, C)

        # ---- optional energy mean gate (inherited from parent) ----
        if self.phase_energy_mean_gate and self._phase_energy_risk is not None:
            risk      = self._phase_energy_risk
            w         = (self.phase_energy_mean_strength * risk[:, None, :]).clamp(0.0, 1.0)
            mu_anchor = mu_fut.mean(dim=1, keepdim=True)
            mu_fut    = (1.0 - w) * mu_fut + w * mu_anchor

        y_patches = y_norm.reshape(B, P_fut, W, C)

        # ---- diagnostic scalars (mirror parent's field names) ----
        patch_mean = y_patches.mean(dim=2, keepdim=True)
        self._last_phase_output_mean_abs = float(
            patch_mean.detach().abs().mean().item()
        )
        self._last_output_patch_mean_abs_before_gauge = float(
            patch_mean.detach().abs().mean().item()
        )
        rms_before = float(
            torch.sqrt(y_patches.detach().pow(2).mean() + self.epsilon).item()
        )
        self._last_output_rms_before_gauge = rms_before
        self._last_dc_gauge_ratio = (
            self._last_output_patch_mean_abs_before_gauge
            / (rms_before + self.epsilon)
        )

        if self.phase_output_zero_mean:
            y_patches = y_patches - patch_mean
            self._last_phase_output_zero_mean_applied = True
        else:
            self._last_phase_output_zero_mean_applied = False

        self._last_output_rms_after_gauge = float(
            torch.sqrt(y_patches.detach().pow(2).mean() + self.epsilon).item()
        )
        y_before_phase = y_patches.detach()

        # ---- existing low-K unit phase rotation (identical to parent) ----
        BP     = B * P_fut
        y_flat = y_patches.reshape(BP, W, C)
        Y_flat = torch.fft.rfft(y_flat, dim=1)          # (BP, W//2+1, C)

        K_eff    = min(K, Y_flat.shape[1] - 1)
        phase_bp = self._phase_rot_fut_hat.reshape(BP, K, C, 2)
        a = phase_bp[:, :K_eff, :, 0]                   # (BP, K_eff, C)
        b = phase_bp[:, :K_eff, :, 1]
        r_re   = 1.0 + a
        r_im   = b
        norm_r = torch.sqrt(r_re.pow(2) + r_im.pow(2) + self.epsilon)
        r_unit = torch.complex(r_re / norm_r, r_im / norm_r)  # (BP, K_eff, C)

        Y_tilde = Y_flat.clone()
        if self.phase_energy_gate and self._phase_energy_gate is not None:
            gate      = self._phase_energy_gate            # (B, C)
            angle_hat = torch.atan2(r_unit.imag, r_unit.real).reshape(
                B, P_fut, K_eff, C
            )
            angle_hat = angle_hat * gate[:, None, None, :]
            self._last_phase_angle_abs_mean = (
                float(angle_hat.detach().abs().mean().item()) if K_eff > 0 else 0.0
            )
            r_gated = torch.polar(
                torch.ones_like(angle_hat), angle_hat
            ).reshape(BP, K_eff, C)
            Y_tilde[:, 1:K_eff + 1, :] = Y_flat[:, 1:K_eff + 1, :] * r_gated
            r_used = r_gated
        else:
            angle = torch.atan2(r_unit.imag, r_unit.real)
            self._last_phase_angle_abs_mean = (
                float(angle.detach().abs().mean().item()) if K_eff > 0 else 0.0
            )
            Y_tilde[:, 1:K_eff + 1, :] = Y_flat[:, 1:K_eff + 1, :] * r_unit
            r_used = r_unit

        y_phase_flat = torch.fft.irfft(Y_tilde, n=W, dim=1)  # (BP, W, C)
        y_phase      = y_phase_flat.reshape(B, P_fut, W, C)

        self._last_phase_effect_rms = float(
            torch.sqrt(
                (y_phase.detach() - y_before_phase).pow(2).mean() + self.epsilon
            ).item()
        )
        self._last_phase_effect_ratio = (
            self._last_phase_effect_rms / (rms_before + self.epsilon)
        )
        self._last_pred_residual_rms_after_phase = float(
            torch.sqrt(y_phase.detach().pow(2).mean() + self.epsilon).item()
        )

        # ---- patch-local phase filter correction ----
        # Pass r_used (gated if gate active, else raw unit rotation) so the
        # filter kernel is consistent with the applied phase rotation.
        y_corr = self._apply_patch_local_filter(y_phase, r_used, K_eff)

        y_out = (y_corr + mu_fut.unsqueeze(2)).reshape(B, self.pred_len, C)
        return y_out.contiguous()

    # ------------------------------------------------------------------
    # Patch-local phase filter (FFT circular convolution)
    # ------------------------------------------------------------------

    def _apply_patch_local_filter(
        self,
        y_phase: torch.Tensor,   # (B, P, W, C)
        r_used:  torch.Tensor,   # (BP, K_eff, C)  complex, unit-magnitude
        K_eff:   int,
    ) -> torch.Tensor:
        """Build a per-patch frequency-response kernel from r_used and apply
        it via circular FFT convolution.

        Identity property: when H = all-ones (K_eff=0 or all rotations are
        trivial), IRFFT(H) is a delta at t=0, its abs() is unchanged,
        RFFT(delta)=all-ones, Y * ones = Y, IRFFT(Y) = y_phase exactly.

        Args:
            y_phase: (B, P, W, C) — backbone output after phase rotation.
            r_used:  (BP, K_eff, C) — complex unit rotation actually applied
                     (gated or raw, whichever was used for Y_tilde).
            K_eff:   number of active frequency bins (≤ K).

        Returns:
            y_corr: (B, P, W, C) — filter-corrected output.
        """
        B, P, W, C = y_phase.shape
        BP         = B * P
        eps        = self.phase_patch_filter_eps

        # ---- build rFFT frequency response H ----
        # DC bin = 1+0j (preserved), bins 1..K_eff = r_used, rest = 1+0j.
        n_freq = W // 2 + 1
        H = torch.ones(BP, n_freq, C, dtype=r_used.dtype, device=y_phase.device)
        if K_eff > 0:
            H[:, 1:K_eff + 1, :] = r_used           # (BP, K_eff, C)

        # ---- time-domain kernel via irfft + abs ----
        kernel_raw = torch.fft.irfft(H, n=W, dim=1)   # (BP, W, C) real
        kernel     = kernel_raw.real.abs()             # (BP, W, C) non-negative

        # ---- kernel diagnostics ----
        kernel_sum  = kernel.sum(dim=1)                # (BP, C)
        kernel_l2sq = kernel.pow(2).sum(dim=1)         # (BP, C)
        kernel_max  = kernel.amax(dim=1)               # (BP, C)
        center_idx  = W // 2
        kernel_ctr  = kernel[:, center_idx, :]         # (BP, C)

        self._last_filter_kernel_l1 = float(kernel_sum.detach().mean().item())
        self._last_filter_kernel_l2 = float(
            torch.sqrt(kernel_l2sq).detach().mean().item()
        )
        self._last_filter_kernel_max_ratio = float(
            (kernel_max / (kernel_sum + eps)).detach().mean().item()
        )
        self._last_filter_kernel_center_ratio = float(
            (kernel_ctr / (kernel_sum + eps)).detach().mean().item()
        )

        # ---- FFT circular convolution: Y * K_freq → y_filt ----
        y_bp = y_phase.reshape(BP, W, C)
        Y      = torch.fft.rfft(y_bp, dim=1)           # (BP, n_freq, C) complex
        K_freq = torch.fft.rfft(kernel, n=W, dim=1)    # (BP, n_freq, C) complex

        y_filt_bp = torch.fft.irfft(Y * K_freq, n=W, dim=1)  # (BP, W, C) real
        y_filt = y_filt_bp.reshape(B, P, W, C)

        # ---- learnable alpha ----
        alpha = (
            self.phase_patch_filter_max
            * torch.sigmoid(self.phase_patch_filter_logit)
        )                                                # (P, C)
        alpha = alpha.view(1, P, 1, C)

        # ---- residual correction ----
        y_corr = y_phase + alpha * (y_filt - y_phase)

        # ---- effect diagnostics ----
        effect    = (y_corr - y_phase).detach()
        rms_effect = float(torch.sqrt(effect.pow(2).mean() + eps).item())
        rms_phase  = float(torch.sqrt(y_phase.detach().pow(2).mean() + eps).item())
        self._last_filter_effect_rms   = rms_effect
        self._last_filter_effect_ratio = rms_effect / (rms_phase + eps)
        self._last_filter_alpha_mean   = float(alpha.detach().mean().item())
        self._last_filter_alpha_max    = float(alpha.detach().max().item())

        # ---- post-filter residual rms (ratio deferred to compute_base_aux_loss) ----
        rms_after = float(torch.sqrt(y_corr.detach().pow(2).mean() + eps).item())
        self._last_filter_pred_residual_rms = rms_after
        self._last_filter_residual_rms_ratio = 0.0

        return y_corr

    # ------------------------------------------------------------------
    # Loss override (updates ratio after oracle is known for current batch)
    # ------------------------------------------------------------------

    def compute_base_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Delegate to parent, then compute filter_residual_rms_ratio with
        the current-batch oracle_rms that parent just refreshed."""
        loss = super().compute_base_aux_loss(y_true)
        if self._last_filter_pred_residual_rms > 0 and self._last_oracle_residual_rms > 0:
            self._last_filter_residual_rms_ratio = (
                self._last_filter_pred_residual_rms
                / (self._last_oracle_residual_rms + self.epsilon)
            )
        else:
            self._last_filter_residual_rms_ratio = 0.0
        return loss

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_last_aux_stats(self) -> dict:
        stats = super().get_last_aux_stats()
        stats.update({
            "phase_filter_alpha_mean":         self._last_filter_alpha_mean,
            "phase_filter_alpha_max":          self._last_filter_alpha_max,
            "phase_filter_effect_rms":         self._last_filter_effect_rms,
            "phase_filter_effect_ratio":       self._last_filter_effect_ratio,
            "phase_filter_kernel_l1":          self._last_filter_kernel_l1,
            "phase_filter_kernel_l2":          self._last_filter_kernel_l2,
            "phase_filter_kernel_max_ratio":   self._last_filter_kernel_max_ratio,
            "phase_filter_kernel_center_ratio": self._last_filter_kernel_center_ratio,
            "pred_residual_rms_after_filter":  self._last_filter_pred_residual_rms,
            "filter_residual_rms_ratio":       self._last_filter_residual_rms_ratio,
        })
        return stats

    def get_route_diagnostics(self) -> dict:
        diag = super().get_route_diagnostics()
        diag["phase_filter_alpha_mean"]          = self._last_filter_alpha_mean
        diag["phase_filter_alpha_max"]           = self._last_filter_alpha_max
        diag["phase_filter_effect_rms"]          = self._last_filter_effect_rms
        diag["phase_filter_effect_ratio"]        = self._last_filter_effect_ratio
        diag["phase_filter_kernel_l1"]           = self._last_filter_kernel_l1
        diag["phase_filter_kernel_l2"]           = self._last_filter_kernel_l2
        diag["phase_filter_kernel_max_ratio"]    = self._last_filter_kernel_max_ratio
        diag["phase_filter_kernel_center_ratio"] = self._last_filter_kernel_center_ratio
        diag["pred_residual_rms_after_filter"]   = self._last_filter_pred_residual_rms
        diag["filter_residual_rms_ratio"]        = self._last_filter_residual_rms_ratio
        return diag
