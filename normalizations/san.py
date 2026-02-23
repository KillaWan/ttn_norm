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
    def __init__(self, seq_len, pred_len, period_len, enc_in, station_type='adaptive'):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.channels = enc_in
        self.enc_in = enc_in
        self.station_type = station_type

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

        # Adapted: internal storage of predicted statistics (set during normalize)
        self._pred_stats: Optional[torch.Tensor] = None

    def _build_model(self):
        seq_len = self.seq_len // self.period_len
        enc_in = self.enc_in
        pred_len = self.pred_len_new
        self.model = _MLP(seq_len, pred_len, enc_in, self.period_len, mode='mean').float()
        self.model_std = _MLP(seq_len, pred_len, enc_in, self.period_len, mode='std').float()

    def normalize(self, input):
        # (B, T, N) → (B, T, N)   [tuple return removed; stats stored in self._pred_stats]
        if self.station_type == 'adaptive':
            bs, length, dim = input.shape
            x = input.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(x, dim=-2, keepdim=True)
            std = torch.std(x, dim=-2, keepdim=True)
            norm_input = (x - mean) / (std + self.epsilon)
            x_flat = input  # already (bs, length, dim)
            mean_all = torch.mean(x_flat, dim=1, keepdim=True)
            outputs_mean = (
                self.model(mean.squeeze(2) - mean_all, x_flat - mean_all) * self.weight[0]
                + mean_all * self.weight[1]
            )
            outputs_std = self.model_std(std.squeeze(2), x_flat)
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
