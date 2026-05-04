"""SANPhaseResidualGainNorm — residual amplitude gain correction on top of SANRouteNorm.

Inherits from SANRouteNorm with route_path="none", route_state="none",
use_phase=True forced.

Adds a per-patch, per-channel amplitude gain correction to the backbone
output residual.  The predictor learns to estimate the future residual RMS
from the historical residual RMS sequence.  At denormalize time, a scalar
gain = exp(clamp(log_amp_pred - log_amp_cur, log_min, log_max)) is applied
to each future patch's mean-normalized signal.

At startup the amp branch is zero-initialized so log_amp_fut ≈ log_amp_anchor
(mean of historical log amplitudes), giving gain ≈ 1 everywhere.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .san_route_norm import SANRouteNorm, _PureSANSeqPredictor


class SANPhaseResidualGainNorm(SANRouteNorm):
    """Residual amplitude gain correction on top of the pure SAN + phase baseline.

    The parent (SANRouteNorm with use_phase=True) applies:
      normalize : z = x - patch_mean  (mean-only normalization per patch)
      denormalize: y_phase = phase_rotate(y_norm) + mu_fut

    This subclass replaces the denormalize output with:
      gain  = exp(clamp(log(amp_pred) - log(amp_cur), log_min, log_max))
      y_out = (y_phase * gain) + mu_fut

    where amp_cur  = per-patch-per-channel RMS of y_phase (backbone residual)
          amp_pred = exp(log_amp_fut_hat) predicted from historical log-RMS.

    Args:
        res_gain_log_min:      Lower clamp on log gain (default log(0.5) ≈ -0.693).
        res_gain_log_max:      Upper clamp on log gain (default log(8)  ≈  2.079).
        res_gain_eps:          Epsilon for amplitude computation.
        res_gain_loss_weight:  Weight for amp prediction loss in stage1 base aux.
        **kwargs:              Forwarded to SANRouteNorm.  route_path, route_state,
                               and use_phase are overridden.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        enc_in: int,
        sigma_min: float = 1e-3,
        phase_k: int = 8,
        phase_zero_init: bool = True,
        res_gain_log_min: float = -0.6931471805599453,   # log(0.5)
        res_gain_log_max: float = 2.0794415416798357,    # log(8)
        res_gain_eps: float = 1e-6,
        res_gain_loss_weight: float = 1.0,
        **kwargs,
    ):
        kwargs["route_path"]  = "none"
        kwargs["route_state"] = "none"
        kwargs["use_phase"]   = True
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            period_len=period_len,
            enc_in=enc_in,
            sigma_min=sigma_min,
            phase_k=phase_k,
            phase_zero_init=phase_zero_init,
            **kwargs,
        )

        self.res_gain_log_min     = float(res_gain_log_min)
        self.res_gain_log_max     = float(res_gain_log_max)
        self.res_gain_eps         = float(res_gain_eps)
        self.res_gain_loss_weight = float(res_gain_loss_weight)

        # Replace parent's predictor with an amp-enabled version.
        # The parent created _PureSANSeqPredictor(use_amp=False); we rebuild it
        # with use_amp=True so predict_mean_phase_amp() is available.
        self.predictor = _PureSANSeqPredictor(
            hist_stat_len=self.hist_stat_len,
            pred_stat_len=self.pred_stat_len,
            window_len=self.window_len,
            enc_in=enc_in,
            sigma_min=sigma_min,
            use_phase=True,
            phase_k=phase_k,
            phase_zero_init=phase_zero_init,
            state_residual_scale_init=0.1,
            use_amp=True,
        )

        # Additional cache field (beyond parent's fields)
        self._log_amp_fut_hat: Optional[torch.Tensor] = None

        # Gain diagnostic cache
        self._last_amp_loss:                float = 0.0
        self._last_gain_amp_cur_mean:       float = 0.0
        self._last_gain_amp_pred_mean:      float = 0.0
        self._last_res_gain_mean:           float = 0.0
        self._last_res_gain_max:            float = 0.0
        self._last_res_gain_min:            float = 1.0
        self._last_gain_pred_residual_rms:  float = 0.0
        self._last_gain_residual_rms_ratio: float = 0.0

    # ------------------------------------------------------------------
    # normalize override
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Run parent's full normalize, then additionally predict future log
        amplitude via the amp branch of _PureSANSeqPredictor."""
        # Reset gain-specific cache fields first (parent._reset_cache doesn't touch them)
        self._log_amp_fut_hat           = None
        self._last_amp_loss             = 0.0
        self._last_gain_amp_cur_mean    = 0.0
        self._last_gain_amp_pred_mean   = 0.0
        self._last_res_gain_mean        = 0.0
        self._last_res_gain_max         = 0.0
        self._last_res_gain_min         = 1.0
        self._last_gain_pred_residual_rms  = 0.0
        self._last_gain_residual_rms_ratio = 0.0

        z_out = super().normalize(x)

        # Only add amp prediction for the pure-SAN + phase path we force.
        if not (self.use_phase and self.route_path == "none"):
            return z_out

        B, T, C = x.shape
        if T != self.seq_len:
            return z_out

        # Re-derive patch statistics (same computation as parent; inputs are
        # detached so no gradient conflicts with the parent's forward pass).
        P_hist    = self.hist_stat_len
        x_patches = x.reshape(B, P_hist, self.window_len, C)
        mu_hist   = x_patches.mean(dim=2)
        anchor    = mu_hist.mean(dim=1, keepdim=True)
        z_patches = x_patches - mu_hist.unsqueeze(2)

        # Per-patch-per-channel residual RMS (detached predictor input)
        amp_hist = torch.sqrt(
            z_patches.detach().pow(2).mean(dim=2) + self.res_gain_eps
        )   # (B, P_hist, C)

        _, _, log_amp_fut_hat = self.predictor.predict_mean_phase_amp(
            mu_hist=mu_hist,
            amp_hist=amp_hist,
            anchor=anchor,
            raw_patches=x_patches.detach(),
            z_patches=z_patches.detach(),
            eps=self.epsilon,
        )
        self._log_amp_fut_hat = log_amp_fut_hat   # (B, P_fut, C)

        return z_out

    # ------------------------------------------------------------------
    # denormalize override
    # ------------------------------------------------------------------

    def denormalize(
        self,
        y_norm: torch.Tensor,
        y_true_oracle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Intercept the pure-SAN + phase path to apply residual gain."""
        T = y_norm.shape[1]
        if (
            self.use_phase
            and self._phase_rot_fut_hat is not None
            and T == self.pred_len
            and self._pred_time_stats is not None
            and self._log_amp_fut_hat is not None
        ):
            return self._denormalize_residual_gain(y_norm)
        return super().denormalize(y_norm, y_true_oracle)

    def _denormalize_residual_gain(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Pure-SAN + phase + residual gain denormalization path."""
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

        # ---- gauge diagnostics (mirror parent field names) ----
        patch_mean = y_patches.mean(dim=2, keepdim=True)
        self._last_phase_output_mean_abs = float(patch_mean.detach().abs().mean().item())
        self._last_output_patch_mean_abs_before_gauge = float(
            patch_mean.detach().abs().mean().item()
        )
        rms_before = float(torch.sqrt(y_patches.detach().pow(2).mean() + self.epsilon).item())
        self._last_output_rms_before_gauge = rms_before
        self._last_dc_gauge_ratio = (
            self._last_output_patch_mean_abs_before_gauge / (rms_before + self.epsilon)
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

        # ---- low-K unit phase rotation (identical to parent's denormalize) ----
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
        else:
            angle = torch.atan2(r_unit.imag, r_unit.real)
            self._last_phase_angle_abs_mean = (
                float(angle.detach().abs().mean().item()) if K_eff > 0 else 0.0
            )
            Y_tilde[:, 1:K_eff + 1, :] = Y_flat[:, 1:K_eff + 1, :] * r_unit

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

        # ---- residual gain correction ----
        # amp_cur: per-patch per-channel RMS of the backbone residual
        amp_cur = torch.sqrt(
            y_phase.detach().pow(2).mean(dim=2) + self.res_gain_eps
        )   # (B, P_fut, C)
        log_amp_cur  = torch.log(amp_cur + self.res_gain_eps)   # (B, P_fut, C)
        log_amp_pred = self._log_amp_fut_hat                     # (B, P_fut, C)
        amp_pred     = torch.exp(log_amp_pred)                   # (B, P_fut, C)

        log_gain = torch.clamp(
            log_amp_pred - log_amp_cur,
            self.res_gain_log_min,
            self.res_gain_log_max,
        )
        gain   = torch.exp(log_gain)           # (B, P_fut, C)
        y_corr = y_phase * gain.unsqueeze(2)   # (B, P_fut, W, C)

        # ---- diagnostics ----
        self._last_gain_amp_cur_mean  = float(amp_cur.detach().mean().item())
        self._last_gain_amp_pred_mean = float(amp_pred.detach().mean().item())
        self._last_res_gain_mean      = float(gain.detach().mean().item())
        self._last_res_gain_max       = float(gain.detach().max().item())
        self._last_res_gain_min       = float(gain.detach().min().item())
        self._last_gain_pred_residual_rms = float(
            torch.sqrt(y_corr.detach().pow(2).mean() + self.epsilon).item()
        )
        self._last_gain_residual_rms_ratio = 0.0  # deferred to compute_base_aux_loss

        # ---- restore mean ----
        y_out = (y_corr + mu_fut.unsqueeze(2)).reshape(B, self.pred_len, C)
        return y_out.contiguous()

    # ------------------------------------------------------------------
    # Loss override
    # ------------------------------------------------------------------

    def compute_base_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Parent base aux (mu + phase) + amplitude prediction loss.

        Computes gain_residual_rms_ratio using the current-batch oracle_rms
        that the parent updates before returning, then appends amp_loss.
        """
        loss = super().compute_base_aux_loss(y_true)

        # Compute gain_residual_rms_ratio using current-batch oracle_rms
        if self._last_gain_pred_residual_rms > 0 and self._last_oracle_residual_rms > 0:
            self._last_gain_residual_rms_ratio = (
                self._last_gain_pred_residual_rms
                / (self._last_oracle_residual_rms + self.epsilon)
            )
        else:
            self._last_gain_residual_rms_ratio = 0.0

        if self._log_amp_fut_hat is None:
            return loss

        # Oracle future patch amplitudes
        B = y_true.shape[0]
        P_fut = self.pred_stat_len
        W     = self.window_len
        C     = y_true.shape[-1]

        y_patches      = y_true.reshape(B, P_fut, W, C)
        oracle_mu      = y_patches.mean(dim=2)                      # (B, P_fut, C)
        oracle_z       = y_patches - oracle_mu.unsqueeze(2)          # (B, P_fut, W, C)
        oracle_amp     = torch.sqrt(
            oracle_z.pow(2).mean(dim=2) + self.res_gain_eps
        )   # (B, P_fut, C)
        log_oracle_amp = torch.log(oracle_amp + self.res_gain_eps)   # (B, P_fut, C)

        amp_loss = F.smooth_l1_loss(
            self._log_amp_fut_hat, log_oracle_amp.detach()
        )
        self._last_amp_loss = float(amp_loss.detach().item())

        total = loss + self.res_gain_loss_weight * amp_loss
        self._last_aux_total = float(total.detach().item())
        return total

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_last_aux_stats(self) -> dict:
        stats = super().get_last_aux_stats()
        stats.update({
            "amp_loss":                     self._last_amp_loss,
            "amp_cur_mean":                 self._last_gain_amp_cur_mean,
            "amp_pred_mean":                self._last_gain_amp_pred_mean,
            "res_gain_mean":                self._last_res_gain_mean,
            "res_gain_max":                 self._last_res_gain_max,
            "res_gain_min":                 self._last_res_gain_min,
            "pred_residual_rms_after_gain": self._last_gain_pred_residual_rms,
            "gain_residual_rms_ratio":      self._last_gain_residual_rms_ratio,
        })
        return stats

    def get_route_diagnostics(self) -> dict:
        diag = super().get_route_diagnostics()
        diag["amp_loss"]                     = self._last_amp_loss
        diag["amp_cur_mean"]                 = self._last_gain_amp_cur_mean
        diag["amp_pred_mean"]                = self._last_gain_amp_pred_mean
        diag["res_gain_mean"]                = self._last_res_gain_mean
        diag["res_gain_max"]                 = self._last_res_gain_max
        diag["res_gain_min"]                 = self._last_res_gain_min
        diag["pred_residual_rms_after_gain"] = self._last_gain_pred_residual_rms
        diag["gain_residual_rms_ratio"]      = self._last_gain_residual_rms_ratio
        return diag
