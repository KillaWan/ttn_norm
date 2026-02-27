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

            # --- Normalize x_used with (optionally clamped) statistics ---
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
