# Single-level SAN base (SANRouteNorm).
#
# Derived from the original multi-level SAN (san.py) by removing all hierarchy logic:
#   - No multi-level hierarchy, no upper_mean_lift / upper_std_lift
#   - No future_norm_z_pred, no second-order norm reconstruction
#   - No coarse-level cache, no level-transition specs
#   - No level-weighted loss, no final_mean_loss, no top/non-top distinction
#
# Retains the core SAN design:
#   - Slice/patch-level mean and std computation from history windows
#   - Slice-level affine normalization:  z = (x - mu_hist_t) / (std_hist_t + eps)
#   - Future mean/std prediction via anchor+residual MLP (same as original SAN level predictor)
#   - Denormalization:  y = mu_fut_hat_t + (std_fut_hat_t + eps) * y_norm
#   - Simple MSE aux loss on predicted vs oracle future window stats
#
# This class is a clean single-level SAN base intended as the foundation for
# state-path routing experiments.  State/path routing logic is NOT implemented here.
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SANRouteNorm(nn.Module):
    """Single-level SAN base for state-path routing experiments.

    Simplified from the original multi-level SAN.  Keeps the same slice-level
    mean/std normalization and future stats prediction; removes all hierarchy.

    Args:
        seq_len:    Length of the historical input window.
        pred_len:   Length of the prediction horizon.
        period_len: Window (patch/slice) length for computing per-window stats.
        enc_in:     Number of input channels.
        stride:     Stride between consecutive windows.  Defaults to period_len
                    (non-overlapping windows, same as original SAN base_stride).
        sigma_min:  Minimum allowed std value (clamp floor).
        w_mu:       Weight for the mean component of the aux loss.
        w_std:      Weight for the std component of the aux loss.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        enc_in: int,
        stride: int = 0,
        sigma_min: float = 1e-3,
        w_mu: float = 1.0,
        w_std: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.window_len = int(period_len)
        self.stride = int(stride) if int(stride) > 0 else self.window_len
        self.enc_in = enc_in
        self.channels = enc_in
        self.sigma_min = float(sigma_min)
        self.epsilon = 1e-5
        self.w_mu = float(w_mu)
        self.w_std = float(w_std)

        self._validate_config()

        self.hist_stat_len = self._compute_n_windows(self.seq_len)
        self.pred_stat_len = self._compute_n_windows(self.pred_len)
        raw_hist_len = self.hist_stat_len * self.window_len

        self.predictor = _SANBasePredictor(
            hist_stat_len=self.hist_stat_len,
            raw_hist_len=raw_hist_len,
            pred_stat_len=self.pred_stat_len,
            enc_in=enc_in,
            sigma_min=sigma_min,
        )

        self._reset_cache()

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if self.seq_len < self.window_len:
            raise ValueError(
                f"SANRouteNorm requires seq_len >= window_len, "
                f"got seq_len={self.seq_len}, window_len={self.window_len}."
            )
        if self.pred_len < self.window_len:
            raise ValueError(
                f"SANRouteNorm requires pred_len >= window_len, "
                f"got pred_len={self.pred_len}, window_len={self.window_len}."
            )
        if (self.seq_len - self.window_len) % self.stride != 0:
            raise ValueError(
                f"SANRouteNorm requires (seq_len - window_len) % stride == 0, "
                f"got seq_len={self.seq_len}, window_len={self.window_len}, stride={self.stride}."
            )
        if (self.pred_len - self.window_len) % self.stride != 0:
            raise ValueError(
                f"SANRouteNorm requires (pred_len - window_len) % stride == 0, "
                f"got pred_len={self.pred_len}, window_len={self.window_len}, stride={self.stride}."
            )

    # ------------------------------------------------------------------
    # Window utilities
    # ------------------------------------------------------------------

    def _compute_n_windows(self, length: int) -> int:
        return (length - self.window_len) // self.stride + 1

    def _extract_windows(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, N, window_len, C)"""
        windows = x.unfold(dimension=1, size=self.window_len, step=self.stride)
        return windows.permute(0, 1, 3, 2).contiguous()

    def _compute_window_stats(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """windows: (B, N, window_len, C) -> (mean, std), each (B, N, C)"""
        mean = windows.mean(dim=2)
        std = windows.std(dim=2).clamp(min=self.sigma_min)
        return mean, std

    def _window_stats_to_time_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        total_length: int,
    ) -> torch.Tensor:
        """Average overlapping window stats back into per-timestep stats.

        Returns: (B, total_length, 2*C) = cat([mu_t, sigma_t], dim=-1)
        """
        B, N, C = mean.shape
        mu_t = torch.zeros(B, total_length, C, device=mean.device, dtype=mean.dtype)
        e2_t = torch.zeros_like(mu_t)
        counts = torch.zeros_like(mu_t)
        e2 = std.pow(2) + mean.pow(2)
        for i in range(N):
            s = i * self.stride
            e = s + self.window_len
            mu_t[:, s:e, :] += mean[:, i : i + 1, :]
            e2_t[:, s:e, :] += e2[:, i : i + 1, :]
            counts[:, s:e, :] += 1.0
        mu_t = mu_t / counts.clamp_min(1.0)
        e2_t = e2_t / counts.clamp_min(1.0)
        sigma_t = torch.sqrt(
            torch.clamp(e2_t - mu_t.pow(2), min=self.sigma_min ** 2)
        )
        return torch.cat([mu_t, sigma_t], dim=-1)

    def _split_stats(
        self, stats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return stats[:, :, : self.channels], stats[:, :, self.channels :]

    # ------------------------------------------------------------------
    # Internal cache
    # ------------------------------------------------------------------

    def _reset_cache(self) -> None:
        self._mu_hist: Optional[torch.Tensor] = None
        self._std_hist: Optional[torch.Tensor] = None
        self._mu_fut_hat: Optional[torch.Tensor] = None
        self._std_fut_hat: Optional[torch.Tensor] = None
        self._pred_time_stats: Optional[torch.Tensor] = None
        self._last_mu_loss: float = 0.0
        self._last_std_loss: float = 0.0
        self._last_aux_total: float = 0.0

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input and cache predicted future stats.

        Normalization:  z = (x - mu_hist_t) / (std_hist_t + eps)
        where mu_hist_t / std_hist_t are per-timestep stats reconstructed
        from window-level history statistics (same approach as original SAN).

        Returns z with the same shape as x.
        """
        self._reset_cache()
        B, T, C = x.shape
        if T != self.seq_len:
            raise ValueError(
                f"SANRouteNorm expected seq_len={self.seq_len}, got {T}."
            )

        # History stats — detached so the normalization is a fixed affine transform
        hist_windows = self._extract_windows(x.detach())
        mu_hist, std_hist = self._compute_window_stats(hist_windows)
        self._mu_hist = mu_hist
        self._std_hist = std_hist

        # Per-timestep normalization
        hist_time_stats = self._window_stats_to_time_stats(mu_hist, std_hist, T)
        hist_time_mean, hist_time_std = self._split_stats(hist_time_stats)
        z = (x - hist_time_mean) / (hist_time_std + self.epsilon)

        # xbar: flattened normalized-input windows  (B, N*window_len, C)
        # Matches the xbar_levels[0] in the original SAN (raw_hist_len = N*window_len).
        norm_windows = self._extract_windows(z.detach())
        xbar = norm_windows.reshape(B, -1, C)

        # Predict future window stats
        anchor = mu_hist.mean(dim=1, keepdim=True)  # (B, 1, C)
        mu_fut_hat, std_fut_hat = self.predictor.predict(mu_hist, std_hist, xbar, anchor)

        self._mu_fut_hat = mu_fut_hat
        self._std_fut_hat = std_fut_hat
        self._pred_time_stats = self._window_stats_to_time_stats(
            mu_fut_hat, std_fut_hat, self.pred_len
        )
        return z

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Restore y_norm to original scale.

        y = mu_fut_hat_t + (std_fut_hat_t + eps) * y_norm
        """
        if self._pred_time_stats is None:
            return y_norm
        T = y_norm.shape[1]
        if T == self._pred_time_stats.shape[1]:
            time_stats = self._pred_time_stats
        else:
            time_stats = self._window_stats_to_time_stats(
                self._mu_fut_hat, self._std_fut_hat, T
            )
        mean, std = self._split_stats(time_stats)
        return mean + (std + self.epsilon) * y_norm

    def loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Aux loss: MSE on predicted vs oracle future window stats.

        aux_total = w_mu * MSE(mu_fut_hat, oracle_mu)
                  + w_std * MSE(std_fut_hat, oracle_std)
        """
        if self._mu_fut_hat is None or self._std_fut_hat is None:
            return torch.tensor(0.0, device=y_true.device)

        true_windows = self._extract_windows(y_true)
        oracle_mu, oracle_std = self._compute_window_stats(true_windows)

        mu_loss = F.mse_loss(self._mu_fut_hat, oracle_mu)
        std_loss = F.mse_loss(self._std_fut_hat, oracle_std)
        aux_total = self.w_mu * mu_loss + self.w_std * std_loss

        self._last_mu_loss = float(mu_loss.detach().item())
        self._last_std_loss = float(std_loss.detach().item())
        self._last_aux_total = float(aux_total.detach().item())

        return aux_total

    def get_last_aux_stats(self) -> dict:
        return {
            "aux_total": self._last_aux_total,
            "mu_loss": self._last_mu_loss,
            "std_loss": self._last_std_loss,
            "mu_hist_mean": (
                float(self._mu_hist.mean().item()) if self._mu_hist is not None else 0.0
            ),
            "std_hist_mean": (
                float(self._std_hist.mean().item()) if self._std_hist is not None else 0.0
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "n",
        station_pred=None,
    ) -> torch.Tensor:
        if mode == "n":
            return self.normalize(x)
        if mode == "d":
            return self.denormalize(x)
        return x


# ---------------------------------------------------------------------------
# Internal MLP and predictor classes
# ---------------------------------------------------------------------------


class _SANBaseMLP(nn.Module):
    """Lightweight MLP for future-stats prediction.

    Matches the original SAN _MLP structure:
      mode='mu':    anchor + residual prediction with Tanh + learnable blend weight.
      mode='sigma': direct sigma prediction with GELU + ReLU.

    Input:
        x_stats: (B, N, C)              — per-window history stats sequence
        x_raw:   (B, raw_hist_len, C)   — xbar (flattened normalized input windows)
        anchor:  (B, 1, C)              — mean anchor, used only in 'mu' mode
    Output:
        (B, pred_stat_len, C)
    """

    def __init__(
        self,
        hist_stat_len: int,
        raw_hist_len: int,
        pred_stat_len: int,
        enc_in: int,
        mode: str,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.pred_stat_len = pred_stat_len
        self.mode = mode
        self.stats_input = nn.Linear(hist_stat_len, hidden_dim)
        self.raw_input = nn.Linear(raw_hist_len, hidden_dim)
        self.output = nn.Linear(2 * hidden_dim, pred_stat_len)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        if mode == "mu":
            self.activation = nn.Tanh()
            self.final_activation = nn.Identity()
            self.weight = nn.Parameter(torch.ones(2, enc_in))
        else:
            self.activation = nn.GELU()
            self.final_activation = nn.ReLU()
            self.weight = None

    def forward(
        self,
        x_stats: torch.Tensor,
        x_raw: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Permute to (B, C, N) / (B, C, raw_hist_len) for Linear over time dim
        xs = x_stats.permute(0, 2, 1)
        xr = x_raw.permute(0, 2, 1)
        feat = torch.cat([self.stats_input(xs), self.raw_input(xr)], dim=-1)
        pred = self.final_activation(
            self.output(self.activation(feat))
        ).permute(0, 2, 1)  # (B, pred_stat_len, C)

        if self.mode == "mu" and anchor is not None:
            anch = anchor.expand(-1, self.pred_stat_len, -1)
            pred = (
                pred * self.weight[0].view(1, 1, -1)
                + anch * self.weight[1].view(1, 1, -1)
            )
        return pred


class _SANBasePredictor(nn.Module):
    """Predicts future (mean, std) from historical window stats and normalized input.

    mean_head: anchor + residual style (same as original SAN _SANLevelPredictor).
    std_head:  direct sigma prediction.
    """

    def __init__(
        self,
        hist_stat_len: int,
        raw_hist_len: int,
        pred_stat_len: int,
        enc_in: int,
        sigma_min: float = 1e-3,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.mean_head = _SANBaseMLP(
            hist_stat_len=hist_stat_len,
            raw_hist_len=raw_hist_len,
            pred_stat_len=pred_stat_len,
            enc_in=enc_in,
            mode="mu",
            hidden_dim=hidden_dim,
        )
        self.std_head = _SANBaseMLP(
            hist_stat_len=hist_stat_len,
            raw_hist_len=raw_hist_len,
            pred_stat_len=pred_stat_len,
            enc_in=enc_in,
            mode="sigma",
            hidden_dim=hidden_dim,
        )

    def predict(
        self,
        mu_hist: torch.Tensor,   # (B, N_hist, C)
        std_hist: torch.Tensor,  # (B, N_hist, C)
        xbar: torch.Tensor,      # (B, N_hist * window_len, C)
        anchor: torch.Tensor,    # (B, 1, C)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_fut = self.mean_head(
            mu_hist - anchor.expand(-1, mu_hist.shape[1], -1),
            xbar,
            anchor,
        )
        std_fut = self.std_head(std_hist, xbar, None).clamp(min=self.sigma_min)
        return mu_fut, std_fut
