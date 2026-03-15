# Adapted from FAN/torch_timeseries/normalizations/SAN.py
# Changes vs FAN original:
#   - normalize() stores pred_stats internally (self._pred_stats) and returns
#     only the normalized tensor (so TTNModel.normalize() sees a single tensor).
#   - denormalize() reads from self._pred_stats when station_pred is None.
#   - loss(true) added: computes mean/std supervision loss using stored pred_stats.
# The FAN experiment used a tuple return + external storage in Model; here the
# stats are stored inside SAN itself so train.py/TTNModel stay generic.
#
# spike_stats extension (vs original adaptation):
#   - When spike_stats=True, detected spike positions are replaced by the
#     per-channel median BEFORE normalization (x_used).  Both the statistics
#     (mean/std sent to MLP) and the actual backbone input come from x_used,
#     so the two are always consistent.
#   - sigma_min clamp prevents variance collapse after inpainting flat windows.
#   - z_clip guards against remaining large z-scores after normalization.
#   - r_max fail-safe: if spike_rate > r_max the inpainting is skipped and
#     the module falls back to vanilla SAN behaviour for that sample.
#   - All stat computation uses x_used.detach() so gradients flow only through
#     the normalization of x_used itself, not through the statistics.
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAN(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        period_len,
        enc_in,
        station_type: str = 'adaptive',
        # ---- spike-robust options (all off by default → original SAN) ----
        spike_stats: bool = False,
        spike_q: float = 0.99,
        spike_dilate: int = 1,
        spike_mode: str = "mad",
        spike_eps: float = 1e-6,
        # ---- numerical guards ----
        sigma_min: float = 1e-3,   # clamp std from below after inpainting
        z_clip: float = 8.0,       # clamp z-scores; <=0 disables
        r_max: float = 0.08,       # fail-safe: revert when spike_rate > r_max
        # ---- TTN (test-time normalization) options ----
        ttn_enabled: bool = False,
        ttn_calib_batches: int = 200,
        ttn_momentum: float = 0.95,      # EMA momentum: fraction retained from old
        ttn_alpha_max: float = 0.5,      # max blend weight toward target stats
        ttn_tau_low: float = 0.05,       # drift below this → alpha = 0 (no update)
        ttn_tau_high: float = 0.20,      # drift above this → alpha = ttn_alpha_max
        ttn_use_mem: bool = True,        # use EMA memory (False → direct target stats)
        ttn_eps: float = 1e-6,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.channels = enc_in
        self.enc_in = enc_in
        self.station_type = station_type

        # Spike-robust options
        self.spike_stats = spike_stats
        self.spike_q = spike_q
        self.spike_dilate = spike_dilate
        self.spike_mode = spike_mode
        self.spike_eps = spike_eps

        # Numerical guards
        self.sigma_min = sigma_min
        self.z_clip = z_clip
        self.r_max = r_max

        # TTN options
        self.ttn_enabled = bool(ttn_enabled)
        self.ttn_calib_batches = int(ttn_calib_batches)
        self.ttn_momentum = float(ttn_momentum)
        self.ttn_alpha_max = float(ttn_alpha_max)
        self.ttn_tau_low = float(ttn_tau_low)
        self.ttn_tau_high = float(ttn_tau_high)
        self.ttn_use_mem = bool(ttn_use_mem)
        self.ttn_eps = float(ttn_eps)

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

        # Adapted: internal storage of predicted statistics (set during normalize)
        self._pred_stats: Optional[torch.Tensor] = None

        # Diagnostic caches (updated each normalize() call)
        self._last_spike_rate: float = 0.0
        self._last_spike_thr_mean: float = 0.0
        self._last_clip_frac: float = 0.0
        self._last_sigma_min_frac: float = 0.0

        # TTN buffers: source stats (calibrated from train), EMA memory, init flag
        # Shapes: (P, 1, C) where P = seq_len_new = seq_len // period_len
        P = self.seq_len_new
        C = enc_in
        self.register_buffer("_ttn_source_mu",    torch.zeros(P, 1, C))
        self.register_buffer("_ttn_source_sigma", torch.ones(P, 1, C))
        self.register_buffer("_ttn_mem_mu",       torch.zeros(P, 1, C))
        self.register_buffer("_ttn_mem_sigma",    torch.zeros(P, 1, C))
        self.register_buffer("_ttn_initialized",  torch.zeros(1, dtype=torch.bool))
        # Source-readiness flag: persists in state_dict so checkpoint restore
        # automatically recovers calibration status alongside source stats.
        # reset_ttn_state() must NOT clear this flag.
        self.register_buffer("_ttn_source_ready", torch.zeros(1, dtype=torch.bool))

        # Per-eval-pass batch counters (reset in reset_ttn_state)
        self._ttn_n_batches: int = 0
        self._ttn_n_updated: int = 0

        # Cached blended stats from the most recent normalize() (for denorm consistency)
        self._ttn_use_mu:    Optional[torch.Tensor] = None
        self._ttn_use_sigma: Optional[torch.Tensor] = None

        # TTN diagnostic scalars (updated each normalize() eval call)
        self._last_ttn_drift:  float = 0.0
        self._last_ttn_alpha:  float = 0.0
        self._last_ttn_d_mu:   float = 0.0
        self._last_ttn_d_sigma: float = 0.0

    # ------------------------------------------------------------------
    # TTN helpers
    # ------------------------------------------------------------------

    def extract_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-period mean and std from x (B, T, C).

        Returns:
            mu:    (P, 1, C) mean over B and patch positions
            sigma: (P, 1, C) std  over B and patch positions
        where P = seq_len // period_len.
        """
        bs, length, dim = x.shape
        x_s = x.reshape(bs, -1, self.period_len, dim)        # (B, P, period_len, C)
        mu    = x_s.mean(dim=-2)                              # (B, P, C)
        sigma = x_s.std(dim=-2).clamp(min=self.sigma_min)    # (B, P, C)
        # Average over batch → (P, 1, C)
        return mu.mean(dim=0, keepdim=False).unsqueeze(1), \
               sigma.mean(dim=0, keepdim=False).unsqueeze(1)

    def set_ttn_source_state(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        """Store calibrated source stats.

        Does NOT seed EMA memory — memory is zeroed by reset_ttn_state() before
        each eval pass and seeded from the first batch of that pass.

        Args:
            mu:    (P, 1, C)
            sigma: (P, 1, C)
        """
        self._ttn_source_mu.copy_(mu)
        self._ttn_source_sigma.copy_(sigma)
        self._ttn_source_ready.fill_(True)

    def reset_ttn_state(self) -> None:
        """Reset EMA memory before a new eval/test pass.

        Zeros memory and marks it as uninitialized so the first batch of the
        new pass seeds it from live stats.  Source readiness is preserved.
        """
        self._ttn_mem_mu.zero_()
        self._ttn_mem_sigma.zero_()
        self._ttn_initialized.fill_(False)
        # Reset diagnostics and batch counters
        self._last_ttn_drift   = 0.0
        self._last_ttn_alpha   = 0.0
        self._last_ttn_d_mu    = 0.0
        self._last_ttn_d_sigma = 0.0
        self._ttn_n_batches    = 0
        self._ttn_n_updated    = 0
        self._ttn_use_mu       = None
        self._ttn_use_sigma    = None

    def get_last_ttn_stats(self) -> dict:
        """Return TTN diagnostics from the most recent normalize() call."""
        update_rate = (self._ttn_n_updated / self._ttn_n_batches
                       if self._ttn_n_batches > 0 else 0.0)
        return {
            "drift":       self._last_ttn_drift,
            "alpha":       self._last_ttn_alpha,
            "d_mu":        self._last_ttn_d_mu,
            "d_sigma":     self._last_ttn_d_sigma,
            "update_rate": update_rate,
        }

    # ------------------------------------------------------------------
    def _spike_inpaint(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect spikes and replace them with the per-channel median.

        Args:
            x: (B, T, C) detached input tensor.
        Returns:
            x_used:     (B, T, C) inpainted tensor (spike → median).
            spike_rate: scalar float tensor, fraction of masked steps.
            thr_mean:   scalar float tensor, mean detection threshold.
        """
        B, T, C = x.shape

        # Per-sample, per-channel median
        center = torch.quantile(x, 0.5, dim=1, keepdim=True)  # (B, 1, C)

        if self.spike_mode == "mad":
            score = (x - center).abs()                          # (B, T, C)
        elif self.spike_mode == "diff":
            diff = (x[:, 1:, :] - x[:, :-1, :]).abs()         # (B, T-1, C)
            score = F.pad(diff, (0, 0, 1, 0))                   # (B, T, C)
        else:
            raise ValueError(f"Unknown spike_mode: {self.spike_mode!r}")

        # Per-(B, C) threshold
        thr = torch.quantile(score, self.spike_q, dim=1, keepdim=True)  # (B, 1, C)
        mask = score >= thr                                               # (B, T, C)

        # Optional temporal dilation
        if self.spike_dilate > 0:
            d = self.spike_dilate
            mask_f = mask.float().permute(0, 2, 1).reshape(B * C, 1, T)
            mask_f = F.max_pool1d(
                mask_f, kernel_size=2 * d + 1, stride=1, padding=d
            )
            mask = mask_f.reshape(B, C, T).permute(0, 2, 1).bool()  # (B, T, C)

        # Replace spike positions with channel median
        x_used = torch.where(mask, center.expand_as(x), x)

        return x_used, mask.float().mean(), thr.mean()

    # ------------------------------------------------------------------
    def _build_model(self):
        seq_len  = self.seq_len // self.period_len
        enc_in   = self.enc_in
        pred_len = self.pred_len_new
        self.model     = _MLP(seq_len, pred_len, enc_in, self.period_len, mode='mean').float()
        self.model_std = _MLP(seq_len, pred_len, enc_in, self.period_len, mode='std').float()

    # ------------------------------------------------------------------
    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """(B, T, N) → (B, T, N)  [stats stored in self._pred_stats]

        When spike_stats=False: behaviour is identical to vanilla SAN
        (x_used = input, no clamp, no z_clip applied).
        """
        if self.spike_stats:
            x_det = input.detach()
            x_inpainted, spike_rate, thr_mean = self._spike_inpaint(x_det)
            spike_rate_val = float(spike_rate.item())
            self._last_spike_rate    = spike_rate_val
            self._last_spike_thr_mean = float(thr_mean.item())

            if spike_rate_val > self.r_max:
                # Fail-safe: too many positions masked → revert to original
                x_used = x_det
            else:
                x_used = x_inpainted          # (B, T, C), detached
        else:
            # Original SAN path: keep gradient through the normalization output
            x_used = input
            self._last_spike_rate     = 0.0
            self._last_spike_thr_mean = 0.0

        if self.station_type == 'adaptive':
            bs, length, dim = input.shape

            # --- Statistics: always computed from detached x_used ---
            x_for_stats = x_used.detach()
            x_s  = x_for_stats.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(x_s, dim=-2, keepdim=True)   # (B, P, 1, C)
            std  = torch.std(x_s,  dim=-2, keepdim=True)   # (B, P, 1, C)

            # sigma_min guard (only applied when spike_stats is active)
            if self.spike_stats:
                std_clamped = torch.clamp(std, min=self.sigma_min)
                self._last_sigma_min_frac = float(
                    (std <= self.sigma_min).float().mean().item()
                )
            else:
                std_clamped = std
                self._last_sigma_min_frac = 0.0

            # --- TTN: eval-only test-time normalization stat blending ----------
            if self.ttn_enabled and not self.training:
                # Strict check: source must be calibrated before any eval pass
                if not bool(self._ttn_source_ready.item()):
                    raise RuntimeError(
                        "SAN TTN is enabled but source stats have not been calibrated. "
                        "Call calibrate_ttn() (which calls set_ttn_source_state()) "
                        "before running evaluation."
                    )

                # Batch-average target stats: (P, 1, C)
                tgt_mu    = mean.mean(dim=0)        # (P, 1, C)
                tgt_sigma = std_clamped.mean(dim=0)  # (P, 1, C)

                if self.ttn_use_mem:
                    if not bool(self._ttn_initialized.item()):
                        # First batch of this eval pass: seed memory from live stats
                        self._ttn_mem_mu.copy_(tgt_mu)
                        self._ttn_mem_sigma.copy_(tgt_sigma)
                        self._ttn_initialized.fill_(True)
                    else:
                        # EMA: retain momentum fraction of old, blend in (1-m) of new
                        m = self.ttn_momentum
                        self._ttn_mem_mu.copy_(
                            m * self._ttn_mem_mu + (1.0 - m) * tgt_mu
                        )
                        self._ttn_mem_sigma.copy_(
                            m * self._ttn_mem_sigma + (1.0 - m) * tgt_sigma
                        )
                    eff_mu    = self._ttn_mem_mu
                    eff_sigma = self._ttn_mem_sigma
                else:
                    eff_mu    = tgt_mu
                    eff_sigma = tgt_sigma
                    self._ttn_initialized.fill_(True)

                # Drift metrics
                # d_mu: per-element normalised by |src_sigma|
                d_mu_t = (eff_mu - self._ttn_source_mu).abs() / (
                    self._ttn_source_sigma.abs() + self.ttn_eps
                )
                d_mu = float(d_mu_t.mean().item())
                # d_sigma: log-space comparison
                d_sigma_t = (
                    torch.log(eff_sigma + self.ttn_eps)
                    - torch.log(self._ttn_source_sigma + self.ttn_eps)
                ).abs()
                d_sigma = float(d_sigma_t.mean().item())
                drift = d_mu + d_sigma

                # Alpha gate: linear ramp from 0 to ttn_alpha_max
                tau_lo, tau_hi = self.ttn_tau_low, self.ttn_tau_high
                if tau_hi <= tau_lo:
                    # Degenerate config: step function at tau_lo
                    alpha = self.ttn_alpha_max if drift > tau_lo else 0.0
                elif drift <= tau_lo:
                    alpha = 0.0
                elif drift >= tau_hi:
                    alpha = self.ttn_alpha_max
                else:
                    alpha = self.ttn_alpha_max * (drift - tau_lo) / (tau_hi - tau_lo)
                alpha = float(alpha)

                # Blending: use_* = (1-alpha)*source + alpha*eff
                use_mu    = (1.0 - alpha) * self._ttn_source_mu    + alpha * eff_mu
                use_sigma = (
                    (1.0 - alpha) * self._ttn_source_sigma + alpha * eff_sigma
                ).clamp(min=self.sigma_min)

                # NaN/Inf fallback: revert to source, mark alpha=0
                if not (torch.isfinite(use_mu).all() and torch.isfinite(use_sigma).all()):
                    use_mu    = self._ttn_source_mu.clone()
                    use_sigma = self._ttn_source_sigma.clone().clamp(min=self.sigma_min)
                    alpha     = 0.0

                # Cache blended stats for denorm consistency
                self._ttn_use_mu    = use_mu    # (P, 1, C)
                self._ttn_use_sigma = use_sigma  # (P, 1, C)

                # Override mean/std with blended stats (broadcast over batch dim)
                mean        = use_mu.unsqueeze(0).expand_as(mean)
                std_clamped = use_sigma.unsqueeze(0).expand_as(std_clamped)

                # Update per-pass counters and diagnostics
                self._ttn_n_batches += 1
                if alpha > 0.0:
                    self._ttn_n_updated += 1
                self._last_ttn_drift   = drift
                self._last_ttn_alpha   = alpha
                self._last_ttn_d_mu    = d_mu
                self._last_ttn_d_sigma = d_sigma
            else:
                # Not in TTN eval mode: clear cache and diagnostics
                self._ttn_use_mu    = None
                self._ttn_use_sigma = None
                self._last_ttn_drift   = 0.0
                self._last_ttn_alpha   = 0.0
                self._last_ttn_d_mu    = 0.0
                self._last_ttn_d_sigma = 0.0
            # --- end TTN ---

            # --- Normalize x_used with (optionally clamped / TTN-blended) statistics ---
            x_used_r  = x_used.reshape(bs, -1, self.period_len, dim)
            norm_input = (x_used_r - mean) / (std_clamped + self.epsilon)

            # z-score clipping (only when spike_stats is active)
            if self.spike_stats and self.z_clip > 0.0:
                norm_input = torch.clamp(norm_input, -self.z_clip, self.z_clip)
                self._last_clip_frac = float(
                    (norm_input.abs() >= self.z_clip - 1e-6).float().mean().item()
                )
            else:
                self._last_clip_frac = 0.0

            # --- MLP stat predictors: flat detached x_used as input ---
            x_flat   = x_for_stats                         # (bs, length, dim)
            mean_all = torch.mean(x_flat, dim=1, keepdim=True)
            outputs_mean = (
                self.model(mean.squeeze(2) - mean_all, x_flat - mean_all) * self.weight[0]
                + mean_all * self.weight[1]
            )
            outputs_std = self.model_std(std_clamped.squeeze(2), x_flat)
            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
            self._pred_stats = outputs[:, -self.pred_len_new:, :]

            # TTN consistency: override _pred_stats so denorm reads the exact same
            # use_mu/use_sigma that normalize applied.  No statistical summarisation
            # is performed; the only operations are positional index mapping and
            # expand (broadcast), preserving every period's TTN value unchanged.
            #
            # Mapping rule: future period i ← input period (P - Q + i), i.e. the
            # last Q input periods map one-to-one onto the Q future periods.
            # When Q > P (rare) the input is tiled before slicing.
            if self.ttn_enabled and not self.training and self._ttn_use_mu is not None:
                Q     = self.pred_len_new
                P_src = self._ttn_use_mu.shape[0]   # = seq_len_new
                if Q <= P_src:
                    pred_mu    = self._ttn_use_mu[-Q:, 0, :]    # (Q, C)
                    pred_sigma = self._ttn_use_sigma[-Q:, 0, :]  # (Q, C)
                else:
                    repeats    = (Q + P_src - 1) // P_src
                    pred_mu    = self._ttn_use_mu[:, 0, :].repeat(repeats, 1)[:Q]    # (Q, C)
                    pred_sigma = self._ttn_use_sigma[:, 0, :].repeat(repeats, 1)[:Q]  # (Q, C)
                ttn_pred_mean = pred_mu.unsqueeze(0).expand(bs, Q, self.channels)
                ttn_pred_std  = pred_sigma.unsqueeze(0).expand(bs, Q, self.channels)
                self._pred_stats = torch.cat([ttn_pred_mean, ttn_pred_std], dim=-1)

            return norm_input.reshape(bs, length, dim)
        else:
            self._pred_stats        = None
            self._last_clip_frac    = 0.0
            self._last_sigma_min_frac = 0.0
            return input

    # ------------------------------------------------------------------
    def denormalize(self, input: torch.Tensor, station_pred=None) -> torch.Tensor:
        # input: (B, O, N)
        # station_pred: (B, pred_len_new, 2*N) — uses self._pred_stats if None
        if station_pred is None:
            station_pred = self._pred_stats
        if self.station_type == 'adaptive' and station_pred is not None:
            bs, length, dim = input.shape
            x    = input.reshape(bs, -1, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std  = station_pred[:, :, self.channels:].unsqueeze(2)
            output = x * (std + self.epsilon) + mean
            return output.reshape(bs, length, dim)
        else:
            return input

    # ------------------------------------------------------------------
    def loss(self, true: torch.Tensor) -> torch.Tensor:
        """Supervision loss: MSE of predicted period mean/std vs oracle from true."""
        if self._pred_stats is None or self.station_type != 'adaptive':
            return torch.tensor(0.0, device=true.device)
        bs, pred_len, n = true.shape
        mean_pred = self._pred_stats[:, :, :n]
        std_pred  = self._pred_stats[:, :, n:]
        true_r = true.reshape(bs, -1, self.period_len, n)
        return F.mse_loss(mean_pred, true_r.mean(dim=2)) + F.mse_loss(std_pred, true_r.std(dim=2))

    # ------------------------------------------------------------------
    def get_last_spike_stats(self) -> dict:
        """Return spike diagnostics from the most recent normalize() call."""
        return {
            "spike_rate":      self._last_spike_rate,
            "spike_thr_mean":  self._last_spike_thr_mean,
            "clip_frac":       self._last_clip_frac,
            "sigma_min_frac":  self._last_sigma_min_frac,
        }

    # ------------------------------------------------------------------
    def forward(self, batch_x, mode='n', station_pred=None):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x, station_pred)


# ======================================================================
class _MLP(nn.Module):
    """Internal MLP for SAN (verbatim from FAN)."""

    def __init__(self, seq_len, pred_len, enc_in, period_len, mode):
        super(_MLP, self).__init__()
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.channels   = enc_in
        self.period_len = period_len
        self.mode       = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input     = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output     = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x     = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
