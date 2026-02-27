# Adapted from FAN/torch_timeseries/normalizations/SAN.py
# Changes vs FAN original:
#   - normalize() stores pred_stats internally (self._pred_stats) and returns
#     only the normalized tensor (so TTNModel.normalize() sees a single tensor).
#   - denormalize() reads from self._pred_stats when station_pred is None.
#   - loss(true) added: computes mean/std supervision loss using stored pred_stats.
# The FAN experiment used a tuple return + external storage in Model; here the
# stats are stored inside SAN itself so train.py/TTNModel stay generic.
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
        station_type='adaptive',
        spike_stats: bool = False,
        spike_q: float = 0.99,
        spike_dilate: int = 1,
        spike_mode: str = "mad",
        spike_eps: float = 1e-6,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.channels = enc_in
        self.enc_in = enc_in
        self.station_type = station_type

        # Spike-robust stat estimation options
        self.spike_stats = spike_stats
        self.spike_q = spike_q
        self.spike_dilate = spike_dilate
        self.spike_mode = spike_mode
        self.spike_eps = spike_eps

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

        # Adapted: internal storage of predicted statistics (set during normalize)
        self._pred_stats: Optional[torch.Tensor] = None

        # Diagnostic caches for spike detection (updated each normalize() call)
        self._last_spike_rate: float = 0.0
        self._last_spike_thr_mean: float = 0.0

    def _spike_inpaint_for_stats(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect spikes and replace them with the per-channel median for statistics.

        Args:
            x: (B, T, C) detached input tensor.
        Returns:
            x_stats:    (B, T, C) spike-inpainted tensor.
            spike_rate: scalar tensor, fraction of masked time steps.
            thr_mean:   scalar tensor, mean detection threshold.
        """
        B, T, C = x.shape

        # Per-sample, per-channel median (robust center)
        center = torch.quantile(x, 0.5, dim=1, keepdim=True)  # (B, 1, C)

        if self.spike_mode == "mad":
            score = (x - center).abs()                          # (B, T, C)
        elif self.spike_mode == "diff":
            diff = (x[:, 1:, :] - x[:, :-1, :]).abs()         # (B, T-1, C)
            score = F.pad(diff, (0, 0, 1, 0))                   # (B, T, C)
        else:
            raise ValueError(f"Unknown spike_mode: {self.spike_mode!r}")

        # Per-(B, C) threshold so each channel has its own sensitivity
        thr = torch.quantile(score, self.spike_q, dim=1, keepdim=True)  # (B, 1, C)
        mask = score >= thr                                               # (B, T, C)

        # Optional temporal dilation of the spike mask
        if self.spike_dilate > 0:
            d = self.spike_dilate
            # Reshape to (B*C, 1, T) for max_pool1d then reshape back
            mask_f = mask.float().permute(0, 2, 1).reshape(B * C, 1, T)
            mask_f = F.max_pool1d(
                mask_f, kernel_size=2 * d + 1, stride=1, padding=d
            )
            mask = mask_f.reshape(B, C, T).permute(0, 2, 1).bool()     # (B, T, C)

        # Replace spike positions with the channel median
        x_stats = torch.where(mask, center.expand_as(x), x)

        spike_rate = mask.float().mean()
        thr_mean = thr.mean()
        return x_stats, spike_rate, thr_mean

    def _build_model(self):
        seq_len = self.seq_len // self.period_len
        enc_in = self.enc_in
        pred_len = self.pred_len_new
        self.model = _MLP(seq_len, pred_len, enc_in, self.period_len, mode='mean').float()
        self.model_std = _MLP(seq_len, pred_len, enc_in, self.period_len, mode='std').float()

    def normalize(self, input):
        # (B, T, N) → (B, T, N)   [tuple return removed; stats stored in self._pred_stats]
        x_det = input.detach()
        if self.spike_stats:
            x_stats, spike_rate, thr_mean = self._spike_inpaint_for_stats(x_det)
            self._last_spike_rate = float(spike_rate.item())
            self._last_spike_thr_mean = float(thr_mean.item())
        else:
            x_stats = x_det
            self._last_spike_rate = 0.0
            self._last_spike_thr_mean = 0.0

        if self.station_type == 'adaptive':
            bs, length, dim = input.shape
            # Compute per-period statistics from spike-robust input
            x_s = x_stats.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(x_s, dim=-2, keepdim=True)
            std = torch.std(x_s, dim=-2, keepdim=True)
            # Normalize original x using the spike-robust statistics
            x = input.reshape(bs, -1, self.period_len, dim)
            norm_input = (x - mean) / (std + self.epsilon)
            # MLP stat predictors use spike-robust flat input
            mean_all = torch.mean(x_stats, dim=1, keepdim=True)
            outputs_mean = (
                self.model(mean.squeeze(2) - mean_all, x_stats - mean_all) * self.weight[0]
                + mean_all * self.weight[1]
            )
            outputs_std = self.model_std(std.squeeze(2), x_stats)
            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
            # Adapted: store stats internally instead of returning tuple
            self._pred_stats = outputs[:, -self.pred_len_new:, :]
            return norm_input.reshape(bs, length, dim)
        else:
            self._pred_stats = None
            return input

    def denormalize(self, input, station_pred=None):
        # input: (B, O, N)
        # station_pred: (B, pred_len_new, 2*N) — uses self._pred_stats if None
        if station_pred is None:
            station_pred = self._pred_stats
        if self.station_type == 'adaptive' and station_pred is not None:
            bs, length, dim = input.shape
            x = input.reshape(bs, -1, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std = station_pred[:, :, self.channels:].unsqueeze(2)
            output = x * (std + self.epsilon) + mean
            return output.reshape(bs, length, dim)
        else:
            return input

    def loss(self, true: torch.Tensor) -> torch.Tensor:
        """Supervision loss: MSE of predicted period mean/std vs oracle from true."""
        if self._pred_stats is None or self.station_type != 'adaptive':
            return torch.tensor(0.0, device=true.device)
        bs, pred_len, n = true.shape
        mean_pred = self._pred_stats[:, :, :n]
        std_pred = self._pred_stats[:, :, n:]
        true_r = true.reshape(bs, -1, self.period_len, n)
        return F.mse_loss(mean_pred, true_r.mean(dim=2)) + F.mse_loss(std_pred, true_r.std(dim=2))

    def get_last_spike_stats(self) -> dict:
        """Return the spike diagnostics from the most recent normalize() call."""
        return {
            "spike_rate": self._last_spike_rate,
            "spike_thr_mean": self._last_spike_thr_mean,
        }

    def forward(self, batch_x, mode='n', station_pred=None):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x, station_pred)


class _MLP(nn.Module):
    """Internal MLP for SAN (verbatim from FAN)."""

    def __init__(self, seq_len, pred_len, enc_in, period_len, mode):
        super(_MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.period_len = period_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
