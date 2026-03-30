# Sliding-window SAN with full mean/std hierarchy prediction.
#
# Level-0 behavior depends on depth:
#   - num_levels == 1: keep original SAN absolute heads (self.model + self.model_std)
#   - num_levels >= 2: disable level0 absolute-mean head (self.model is None)
#     and obtain level0 mean only from hierarchy reconstruction future_mean_recon[0]
#     while keeping level0 std on the original model_std path
#
# Full mean/std hierarchy (num_levels >= 2):
#   - hier_stats_predictors[l-1] predicts absolute future_mean_pred[l] + future_std_pred[l]
#     for levels l = 1..L.  Both outputs have direct oracle loss (L_stats[l]).
#   - hier_norm_predictors[l] predicts normalized future_mean_norm_pred[l] for levels l = 0..L-1.
#     Each output has direct oracle loss (L_norm[l]).
#   - Top-down reconstruction: future_mean_recon[L] = future_mean_pred[L];
#     for l=L-1..0: future_mean_recon[l] = future_mean_norm_pred[l] * lift(future_std_pred[l+1])
#                                           + lift(future_mean_pred[l+1]).
#   - Middle-level consistency: L_cons[l] = MSE(future_mean_recon[l], future_mean_pred[l])
#     for l = 1..L-1.
#
# Level-0 denorm always uses future_mean_recon[0] and future_std0_pred (from model_std).
# OSTN remains compatible and only refines level-0 future stats at eval/test.
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _OSTNStatsCorrector(nn.Module):
    """GRU + MLP stats corrector for OSTN."""

    D_HIST: int = 4
    D_OLAP: int = 4

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pos_dim: int,
        pred_stat_len: int,
        use_patchwise_overlap: bool,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.pred_stat_len = pred_stat_len
        self.use_patchwise_overlap = use_patchwise_overlap

        self.hist_gru = nn.GRU(
            input_size=self.D_HIST,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.olap_enc = nn.Sequential(
            nn.Linear(self.D_OLAP, hidden_dim),
            nn.GELU(),
        )
        self.pos_emb = nn.Parameter(torch.randn(pred_stat_len, pos_dim) * 0.01)

        in_corr = 2 * hidden_dim + 2 + pos_dim
        self.delta_mu_head = nn.Sequential(
            nn.Linear(in_corr, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.delta_lsig_head = nn.Sequential(
            nn.Linear(in_corr, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.delta_mu_head[-1].weight)
        nn.init.zeros_(self.delta_mu_head[-1].bias)
        nn.init.zeros_(self.delta_lsig_head[-1].weight)
        nn.init.zeros_(self.delta_lsig_head[-1].bias)
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.constant_(self.alpha_head[-1].bias, -3.0)

    def forward(
        self,
        hist_stats_seq: torch.Tensor,
        base_future_mean: torch.Tensor,
        base_future_logsigma: torch.Tensor,
        prev_overlap_summary: torch.Tensor,
    ):
        batch_size, hist_len, channels, _ = hist_stats_seq.shape
        pred_len = base_future_mean.shape[1]

        x_gru = hist_stats_seq.permute(0, 2, 1, 3).reshape(
            batch_size * channels, hist_len, self.D_HIST
        )
        _, h_n = self.hist_gru(x_gru)
        hist_ctx = h_n[-1].reshape(batch_size, channels, self.hidden_dim)

        if self.use_patchwise_overlap and prev_overlap_summary.dim() == 4:
            olap_in = prev_overlap_summary.mean(dim=1)
        else:
            olap_in = prev_overlap_summary
        olap_ctx = self.olap_enc(olap_in)

        ctx = torch.cat([hist_ctx, olap_ctx], dim=-1)
        alpha = torch.sigmoid(
            self.alpha_head(ctx.reshape(batch_size * channels, 2 * self.hidden_dim)).reshape(
                batch_size, channels, 1
            )
        ).permute(0, 2, 1)

        ctx_exp = ctx.unsqueeze(1).expand(batch_size, pred_len, channels, 2 * self.hidden_dim)
        pos_exp = self.pos_emb[:pred_len].unsqueeze(0).unsqueeze(2).expand(
            batch_size, pred_len, channels, self.pos_dim
        )
        base_in = torch.stack([base_future_mean, base_future_logsigma], dim=-1)
        feat = torch.cat([ctx_exp, base_in, pos_exp], dim=-1)
        feat_flat = feat.reshape(batch_size * pred_len * channels, -1)

        delta_mu = self.delta_mu_head(feat_flat).reshape(batch_size, pred_len, channels)
        delta_lsig = self.delta_lsig_head(feat_flat).reshape(batch_size, pred_len, channels)
        return delta_mu, delta_lsig, alpha


class SAN(nn.Module):
    MAX_EXTRA_LEVELS: int = 2
    LEVEL_LOSS_WEIGHTS: tuple[float, float, float] = (1.0, 0.2, 0.05)

    def __init__(
        self,
        seq_len,
        pred_len,
        period_len,
        enc_in,
        station_type: str = 'adaptive',
        stride: int = 0,
        base_stride: int = 0,
        force_extra_levels: int = -1,
        sigma_min: float = 1e-3,
        ostn_enabled: bool = False,
        ostn_hidden_dim: int = 64,
        ostn_num_layers: int = 2,
        ostn_dropout: float = 0.0,
        ostn_pos_dim: int = 16,
        ostn_use_patchwise_overlap_summary: bool = True,
        ostn_alpha_l1: float = 1e-3,
        ostn_overlap_weight: float = 1.0,
        ostn_stats_weight: float = 1.0,
        ostn_logsigma_min: float = -6.0,
        ostn_logsigma_max: float = 6.0,
        ostn_reset_each_eval: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.window_len = self.period_len
        self.base_stride = int(base_stride) if int(base_stride) > 0 else self.window_len
        self.hier_stride = int(stride) if int(stride) > 0 else 0
        self.stride = self.hier_stride
        self.channels = enc_in
        self.enc_in = enc_in
        self.station_type = station_type
        self.force_extra_levels = int(force_extra_levels)
        self.sigma_min = float(sigma_min)

        if self.force_extra_levels not in {-1, 0, 1, 2}:
            raise ValueError(
                "SAN force_extra_levels must be one of {-1, 0, 1, 2}, "
                f"got {self.force_extra_levels}."
            )

        self.ostn_enabled = bool(ostn_enabled)
        self.ostn_hidden_dim = int(ostn_hidden_dim)
        self.ostn_num_layers = int(ostn_num_layers)
        self.ostn_dropout = float(ostn_dropout)
        self.ostn_pos_dim = int(ostn_pos_dim)
        self.ostn_use_patchwise_overlap_summary = bool(ostn_use_patchwise_overlap_summary)
        self.ostn_alpha_l1 = float(ostn_alpha_l1)
        self.ostn_overlap_weight = float(ostn_overlap_weight)
        self.ostn_stats_weight = float(ostn_stats_weight)
        self.ostn_logsigma_min = float(ostn_logsigma_min)
        self.ostn_logsigma_max = float(ostn_logsigma_max)
        self.ostn_reset_each_eval = bool(ostn_reset_each_eval)

        self._validate_config_lengths()
        self.hist_stat_len = self._compute_n_windows(self.seq_len)
        self.pred_stat_len = self._compute_n_windows(self.pred_len)
        self.epsilon = 1e-5
        self._build_level_layout()
        self._build_model()
        self._reset_prediction_cache()

        self._prev_stream_final_pred: Optional[torch.Tensor] = None
        self._prev_stream_refined_mean: Optional[torch.Tensor] = None
        self._prev_stream_refined_logsigma: Optional[torch.Tensor] = None
        self._prev_stream_base_final_pred: Optional[torch.Tensor] = None
        self._prev_stream_refined_final_pred: Optional[torch.Tensor] = None
        self._prev_stream_overlap_summary: Optional[torch.Tensor] = None
        self._prev_stream_valid: bool = False

        self._curr_refined_mean: Optional[torch.Tensor] = None
        self._curr_refined_logsigma: Optional[torch.Tensor] = None
        self._curr_base_mean: Optional[torch.Tensor] = None
        self._curr_base_logsigma: Optional[torch.Tensor] = None

        self._last_ostn_applied: bool = False
        self._last_ostn_alpha_mean: float = 0.0
        self._last_ostn_alpha_max: float = 0.0
        self._last_ostn_delta_mu_abs_mean: float = 0.0
        self._last_ostn_delta_lsig_abs_mean: float = 0.0
        self._last_ostn_base_overlap_loss: float = 0.0
        self._last_ostn_refined_overlap_loss: float = 0.0
        self._last_ostn_base_mu_abs_mean: float = 0.0
        self._last_ostn_refined_mu_abs_mean: float = 0.0
        self._last_ostn_base_sigma_mean: float = 0.0
        self._last_ostn_refined_sigma_mean: float = 0.0

        self._oracle_pred_stats: Optional[torch.Tensor] = None
        self._oracle_mode: str = 'both'

    def _reset_prediction_cache(self) -> None:
        self._pred_stats: Optional[torch.Tensor] = None
        self._base_pred_stats: Optional[torch.Tensor] = None
        self._refined_pred_stats: Optional[torch.Tensor] = None
        self._pred_time_stats: Optional[torch.Tensor] = None
        self._base_pred_time_stats: Optional[torch.Tensor] = None
        self._refined_pred_time_stats: Optional[torch.Tensor] = None

        self._level0_hist_mu: Optional[torch.Tensor] = None
        self._level0_hist_std: Optional[torch.Tensor] = None
        self._level0_hist_lambda: Optional[torch.Tensor] = None
        self._level0_future_base_mu: Optional[torch.Tensor] = None
        self._level0_future_base_lambda: Optional[torch.Tensor] = None
        self._level0_future_final_mu: Optional[torch.Tensor] = None
        self._level0_future_final_lambda: Optional[torch.Tensor] = None
        self._level0_future_ostn_mu: Optional[torch.Tensor] = None
        self._level0_future_ostn_lambda: Optional[torch.Tensor] = None
        self._final_future_time_stats: Optional[torch.Tensor] = None

        self._hierarchy_cache: list[dict] = []
        self._coarse_level_cache: list[dict] = []
        self._last_level_losses: dict[int, float] = {}
        self._last_total_stats_loss: float = 0.0
        self._last_recon0_loss: float = 0.0
        self._last_std0_loss: float = 0.0
        # Per-level diagnostics for the full mean/std hierarchy
        self._last_norm_losses: dict[int, float] = {}    # L_norm[l] for l=0..L-1
        self._last_stats_losses: dict[int, float] = {}   # L_stats[l] for l=1..L
        self._last_cons_losses: dict[int, float] = {}    # L_cons[l] for l=1..L-1
        self._last_weighted_hier_loss: float = 0.0
        self._last_hier_to_level0_ratio: float = 0.0

    def _select_num_coarse_levels(self, level0_hist_len: int) -> int:
        if level0_hist_len <= 8:
            return 0
        if level0_hist_len <= 24:
            return 1
        return self.MAX_EXTRA_LEVELS

    def _select_meta_slicing(self, length: int) -> Optional[tuple[int, int]]:
        if length <= 8:
            return None
        if length <= 16:
            meta_patch = 3
        else:
            meta_patch = 4

        if self.hier_stride > 0:
            meta_stride = min(self.hier_stride, meta_patch)
        else:
            meta_stride = meta_patch
        return meta_patch, meta_stride

    def _compute_meta_output_len(self, length: int, meta_patch: int, meta_stride: int) -> int:
        if length < meta_patch:
            return 0
        return (length - meta_patch) // meta_stride + 1

    def _build_level_layout(self) -> None:
        if self.force_extra_levels >= 0:
            allowed_extra_levels = self.force_extra_levels
        else:
            allowed_extra_levels = self._select_num_coarse_levels(self.hist_stat_len)
        self.level_hist_lens = [self.hist_stat_len]
        self.level_pred_lens = [self.pred_stat_len]
        self.level_transition_specs: list[dict[str, int]] = []

        current_hist_len = self.hist_stat_len
        current_pred_len = self.pred_stat_len
        for _ in range(allowed_extra_levels):
            meta = self._select_meta_slicing(current_hist_len)
            if meta is None:
                break
            meta_patch, meta_stride = meta
            next_hist_len = self._compute_meta_output_len(
                current_hist_len,
                meta_patch,
                meta_stride,
            )
            next_pred_len = self._compute_meta_output_len(
                current_pred_len,
                meta_patch,
                meta_stride,
            )
            if next_hist_len <= 0 or next_pred_len <= 0:
                break

            self.level_transition_specs.append(
                {
                    'meta_patch': meta_patch,
                    'meta_stride': meta_stride,
                }
            )
            self.level_hist_lens.append(next_hist_len)
            self.level_pred_lens.append(next_pred_len)
            current_hist_len = next_hist_len
            current_pred_len = next_pred_len

        self.num_levels = len(self.level_hist_lens)

    def _std_to_logsigma(self, std: torch.Tensor) -> torch.Tensor:
        return torch.log(std.clamp(min=self.sigma_min)).clamp(
            min=self.ostn_logsigma_min,
            max=self.ostn_logsigma_max,
        )

    def _logsigma_to_std(self, logsigma: torch.Tensor) -> torch.Tensor:
        return logsigma.exp().clamp(min=self.sigma_min)

    def _validate_config_lengths(self) -> None:
        if self.seq_len < self.window_len:
            raise ValueError(
                f"SAN requires seq_len >= window_len, got seq_len={self.seq_len}, window_len={self.window_len}."
            )
        if self.pred_len < self.window_len:
            raise ValueError(
                f"SAN requires pred_len >= window_len, got pred_len={self.pred_len}, window_len={self.window_len}."
            )
        if (self.seq_len - self.window_len) % self.base_stride != 0:
            raise ValueError(
                "SAN requires (seq_len - window_len) % base_stride == 0, "
                f"got seq_len={self.seq_len}, window_len={self.window_len}, base_stride={self.base_stride}."
            )
        if (self.pred_len - self.window_len) % self.base_stride != 0:
            raise ValueError(
                "SAN requires (pred_len - window_len) % base_stride == 0, "
                f"got pred_len={self.pred_len}, window_len={self.window_len}, base_stride={self.base_stride}."
            )

    def _compute_n_windows(self, length: int) -> int:
        if length < self.window_len:
            raise ValueError(
                f"Length must be >= window_len, got length={length}, window_len={self.window_len}."
            )
        return (length - self.window_len) // self.base_stride + 1

    def _extract_windows(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B, T, C), got {tuple(x.shape)}.")
        if x.shape[1] < self.window_len:
            raise ValueError(
                f"Expected T >= window_len, got T={x.shape[1]}, window_len={self.window_len}."
            )
        if (x.shape[1] - self.window_len) % self.base_stride != 0:
            raise ValueError(
                "Sliding-window SAN requires (T - window_len) % base_stride == 0 at runtime, "
                f"got T={x.shape[1]}, window_len={self.window_len}, base_stride={self.base_stride}."
            )
        windows = x.unfold(dimension=1, size=self.window_len, step=self.base_stride)
        return windows.permute(0, 1, 3, 2).contiguous()

    def _extract_norm_windows(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self._extract_windows(x_norm)

    def _compute_window_stats(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = windows.mean(dim=2)
        std = windows.std(dim=2).clamp(min=self.sigma_min)
        return mean, std

    def _window_stats_to_time_stats(
        self,
        window_mean: torch.Tensor,
        window_std: torch.Tensor,
        total_length: int,
    ) -> torch.Tensor:
        batch_size, n_windows, channels = window_mean.shape
        expected = self._compute_n_windows(total_length)
        if n_windows != expected:
            raise ValueError(
                f"Expected {expected} windows for total_length={total_length}, got {n_windows}."
            )

        sum_mean = torch.zeros(
            batch_size,
            total_length,
            channels,
            device=window_mean.device,
            dtype=window_mean.dtype,
        )
        sum_second = torch.zeros_like(sum_mean)
        counts = torch.zeros_like(sum_mean)
        second_per_window = window_std.pow(2) + window_mean.pow(2)

        for index in range(n_windows):
            start = index * self.base_stride
            end = start + self.window_len
            sum_mean[:, start:end, :] += window_mean[:, index:index + 1, :]
            sum_second[:, start:end, :] += second_per_window[:, index:index + 1, :]
            counts[:, start:end, :] += 1.0

        mu_t = sum_mean / counts.clamp_min(1.0)
        second_t = sum_second / counts.clamp_min(1.0)
        sigma_t = torch.sqrt(
            torch.clamp(second_t - mu_t.pow(2), min=self.sigma_min * self.sigma_min)
        )
        return torch.cat([mu_t, sigma_t], dim=-1)

    def _split_stats(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return stats[:, :, :self.channels], stats[:, :, self.channels:]

    def _stats_to_time_stats(self, stats: torch.Tensor, total_length: int) -> torch.Tensor:
        if stats.shape[1] == total_length:
            return stats
        expected = self._compute_n_windows(total_length)
        if stats.shape[1] != expected:
            raise ValueError(
                f"Stats length {stats.shape[1]} is incompatible with total_length={total_length}."
            )
        mean, std = self._split_stats(stats)
        return self._window_stats_to_time_stats(mean, std, total_length)

    def _slice_stats_sequence(
        self,
        stats_seq: torch.Tensor,
        meta_patch: int,
        meta_stride: int,
    ) -> torch.Tensor:
        if stats_seq.dim() != 3:
            raise ValueError(
                f"Expected stats_seq with shape (B, N, C), got {tuple(stats_seq.shape)}."
            )
        if meta_patch <= 0 or meta_stride <= 0:
            raise ValueError(
                f"meta_patch and meta_stride must be positive, got {meta_patch}, {meta_stride}."
            )
        if stats_seq.shape[1] < meta_patch:
            return stats_seq[:, :0, :]
        windows = stats_seq.unfold(dimension=1, size=meta_patch, step=meta_stride)
        return windows.mean(dim=-1).contiguous()

    def _recompute_seq_stats(
        self,
        seq: torch.Tensor,
        meta_patch: int,
        meta_stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq.dim() != 3:
            raise ValueError(f"Expected seq with shape (B, N, C), got {tuple(seq.shape)}.")
        if meta_patch <= 0 or meta_stride <= 0:
            raise ValueError(
                f"meta_patch and meta_stride must be positive, got {meta_patch}, {meta_stride}."
            )
        if seq.shape[1] < meta_patch:
            empty = seq[:, :0, :]
            return empty, empty

        windows = seq.unfold(dimension=1, size=meta_patch, step=meta_stride)
        seq_mean = windows.mean(dim=-1).contiguous()
        seq_std = windows.std(dim=-1, unbiased=False).clamp(min=self.sigma_min).contiguous()
        return seq_mean, seq_std

    def _lift_seq_stats(
        self,
        stats_seq: torch.Tensor,
        target_len: int,
        meta_patch: int,
        meta_stride: int,
    ) -> torch.Tensor:
        if stats_seq.dim() != 3:
            raise ValueError(
                f"Expected stats_seq with shape (B, N, C), got {tuple(stats_seq.shape)}."
            )
        if stats_seq.shape[1] <= 0:
            raise ValueError("Cannot lift an empty stats sequence.")
        if meta_patch <= 0 or meta_stride <= 0:
            raise ValueError(
                f"meta_patch and meta_stride must be positive, got {meta_patch}, {meta_stride}."
            )

        batch_size, coarse_len, channels = stats_seq.shape
        sum_tensor = torch.zeros(
            batch_size,
            target_len,
            channels,
            device=stats_seq.device,
            dtype=stats_seq.dtype,
        )
        count_tensor = torch.zeros_like(sum_tensor)

        for index in range(coarse_len):
            start = index * meta_stride
            end = min(start + meta_patch, target_len)
            if start >= target_len or end <= start:
                continue
            sum_tensor[:, start:end, :] += stats_seq[:, index:index + 1, :]
            count_tensor[:, start:end, :] += 1.0

        return sum_tensor / count_tensor.clamp_min(1.0)

    def _upsample_stats_sequence(
        self,
        stats_seq: torch.Tensor,
        target_len: int,
        meta_patch: int,
        meta_stride: int,
    ) -> torch.Tensor:
        return self._lift_seq_stats(stats_seq, target_len, meta_patch, meta_stride)

    def _compute_roughness(self, stats_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if stats_seq.shape[1] <= 1:
            roughness = torch.zeros(
                stats_seq.shape[0],
                1,
                stats_seq.shape[2],
                device=stats_seq.device,
                dtype=stats_seq.dtype,
            )
        else:
            delta = stats_seq[:, 1:, :] - stats_seq[:, :-1, :]
            seq_std = stats_seq.std(dim=1, keepdim=True, unbiased=False)
            roughness = delta.abs().mean(dim=1, keepdim=True) / (seq_std + self.epsilon)
        alpha = 1.0 / (1.0 + roughness)
        return roughness, alpha

    def _build_recursive_mean_hierarchy(
        self,
        level0_mean_seq: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        level_mean_seqs: list[torch.Tensor] = [level0_mean_seq]
        level_std_seqs: list[torch.Tensor] = [torch.zeros_like(level0_mean_seq)]
        current_mean_seq = level0_mean_seq

        for spec in self.level_transition_specs:
            next_mean_seq, next_std_seq = self._recompute_seq_stats(
                current_mean_seq,
                meta_patch=spec['meta_patch'],
                meta_stride=spec['meta_stride'],
            )
            level_mean_seqs.append(next_mean_seq)
            level_std_seqs.append(next_std_seq)
            current_mean_seq = next_mean_seq

        return level_mean_seqs, level_std_seqs

    def _build_xbar_hierarchy(self, norm_input: torch.Tensor) -> list[torch.Tensor]:
        if norm_input.dim() != 3:
            raise ValueError(
                f"Expected norm_input with shape (B, T, C), got {tuple(norm_input.shape)}."
            )

        current_blocks = self._extract_norm_windows(norm_input)
        levels = [current_blocks.reshape(current_blocks.shape[0], -1, current_blocks.shape[-1])]

        for spec in self.level_transition_specs:
            block_windows = current_blocks.unfold(
                dimension=1,
                size=spec['meta_patch'],
                step=spec['meta_stride'],
            )
            current_blocks = block_windows.mean(dim=-1).contiguous()
            levels.append(
                current_blocks.reshape(current_blocks.shape[0], -1, current_blocks.shape[-1])
            )

        return levels

    def _build_hist_stats_seq_from_stats(
        self,
        hist_mean: torch.Tensor,
        hist_std: torch.Tensor,
    ) -> torch.Tensor:
        hist_logsigma = self._std_to_logsigma(hist_std)
        delta_mean = torch.zeros_like(hist_mean)
        delta_logsigma = torch.zeros_like(hist_logsigma)
        delta_mean[:, 1:, :] = hist_mean[:, 1:, :] - hist_mean[:, :-1, :]
        delta_logsigma[:, 1:, :] = hist_logsigma[:, 1:, :] - hist_logsigma[:, :-1, :]
        return torch.stack([hist_mean, hist_logsigma, delta_mean, delta_logsigma], dim=-1)

    def _build_hist_stats_seq(self, x: torch.Tensor) -> torch.Tensor:
        hist_windows = self._extract_windows(x)
        hist_mean, hist_std = self._compute_window_stats(hist_windows)
        return self._build_hist_stats_seq_from_stats(hist_mean, hist_std)

    def _build_model(self):
        self.level_raw_hist_lens = [hist_len * self.window_len for hist_len in self.level_hist_lens]
        # level0 absolute-mean head exists only in single-level SAN.
        self.model: Optional[nn.Module]
        if self.num_levels == 1:
            self.model = _MLP(
                hist_stat_len=self.level_hist_lens[0],
                raw_hist_len=self.level_raw_hist_lens[0],
                pred_stat_len=self.level_pred_lens[0],
                enc_in=self.enc_in,
                mode='mu',
            ).float()
        else:
            self.model = None

        # level0 std head is always active and remains separate from hierarchy stats predictors.
        self.model_std = _MLP(
            hist_stat_len=self.level_hist_lens[0],
            raw_hist_len=self.level_raw_hist_lens[0],
            pred_stat_len=self.level_pred_lens[0],
            enc_in=self.enc_in,
            mode='lambda',
        ).float()
        # hier_norm_predictors[l] handles level l (0..num_levels-2): predicts future_mean_norm_pred[l]
        # hier_stats_predictors[l-1] handles level l (1..num_levels-1): predicts future_mean_pred[l] + future_std_pred[l]
        self.hier_norm_predictors = nn.ModuleList()
        self.hier_stats_predictors = nn.ModuleList()

        if self.num_levels >= 2:
            # Norm predictors for levels 0..L-1 (one per level that has an upper context)
            self.hier_norm_predictors = nn.ModuleList(
                [
                    _HierNormPredictor(
                        hist_stat_len=self.level_hist_lens[level_idx],
                        raw_hist_len=self.level_raw_hist_lens[level_idx],
                        pred_stat_len=self.level_pred_lens[level_idx],
                        enc_in=self.enc_in,
                    ).float()
                    for level_idx in range(self.num_levels - 1)   # 0 .. L-1
                ]
            )
            # Stats predictors for levels 1..L (one per level above zero)
            self.hier_stats_predictors = nn.ModuleList(
                [
                    _HierStatsPredictor(
                        hist_stat_len=self.level_hist_lens[level_idx],
                        raw_hist_len=self.level_raw_hist_lens[level_idx],
                        pred_stat_len=self.level_pred_lens[level_idx],
                        enc_in=self.enc_in,
                        sigma_min=self.sigma_min,
                    ).float()
                    for level_idx in range(1, self.num_levels)    # 1 .. L
                ]
            )
        if self.ostn_enabled:
            self.ostn_corrector = _OSTNStatsCorrector(
                hidden_dim=self.ostn_hidden_dim,
                num_layers=self.ostn_num_layers,
                dropout=self.ostn_dropout,
                pos_dim=self.ostn_pos_dim,
                pred_stat_len=self.pred_stat_len,
                use_patchwise_overlap=self.ostn_use_patchwise_overlap_summary,
            )

    def set_oracle_stats(self, batch_y: torch.Tensor, mode: str = 'both') -> None:
        if self.station_type != 'adaptive':
            return
        oracle_windows = self._extract_windows(batch_y)
        oracle_mean, oracle_std = self._compute_window_stats(oracle_windows)
        self._oracle_pred_stats = torch.cat([oracle_mean, oracle_std], dim=-1)
        self._oracle_mode = mode

    def clear_oracle_stats(self) -> None:
        self._oracle_pred_stats = None
        self._oracle_mode = 'both'

    def extract_state(self, x: torch.Tensor):
        hist_windows = self._extract_windows(x)
        mean, sigma = self._compute_window_stats(hist_windows)
        return mean.mean(0).unsqueeze(1), sigma.mean(0).unsqueeze(1)

    def _get_mu_predictor(self, level_idx: int) -> nn.Module:
        if level_idx != 0:
            raise ValueError("_get_mu_predictor is only valid for level0.")
        if self.model is None:
            raise ValueError(
                "hierarchy 模式下 level0 absolute mean head 已禁用; "
                "level0 mean 必须通过 hier_norm_predictors[0] + upper stats denorm 获得."
            )
        return self.model

    def _get_lambda_predictor(self, level_idx: int) -> nn.Module:
        if level_idx != 0:
            raise ValueError("_get_lambda_predictor is only valid for level0.")
        return self.model_std

    def _predict_level0_absolute_mean_std(
        self,
        hist_mu: torch.Tensor,
        hist_scale: torch.Tensor,
        xbar_raw_like: torch.Tensor,
        global_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-level SAN only: predict level0 absolute mean + logsigma."""
        if self.num_levels != 1:
            raise ValueError(
                "_predict_level0_absolute_mean_std is only valid when num_levels == 1."
            )
        global_hist = global_mean.expand(-1, hist_mu.shape[1], -1)
        mu_pred = self._get_mu_predictor(0)(
            hist_mu - global_hist,
            xbar_raw_like,
            global_mean,
        )
        scale_pred = self._get_lambda_predictor(0)(
            hist_scale,
            xbar_raw_like,
            None,
        ).clamp(min=self.sigma_min)
        return mu_pred, self._std_to_logsigma(scale_pred)

    def _predict_level0_std_only(
        self,
        hist_scale: torch.Tensor,
        xbar_raw_like: torch.Tensor,
    ) -> torch.Tensor:
        """Predict level0 logsigma via model_std. Valid for both single-level and hierarchy modes."""
        scale_pred = self._get_lambda_predictor(0)(
            hist_scale,
            xbar_raw_like,
            None,
        ).clamp(min=self.sigma_min)
        return self._std_to_logsigma(scale_pred)

    def _predict_hierarchical_future_stats(
        self,
        hist_window_mean: torch.Tensor,
        hist_window_std: torch.Tensor,
        norm_input: torch.Tensor,
        global_mean: torch.Tensor,
    ) -> list[dict]:
        max_level = self.num_levels - 1
        hist_mean_seq, hist_std_seq = self._build_recursive_mean_hierarchy(hist_window_mean)
        xbar_levels = self._build_xbar_hierarchy(norm_input)

        hist_lambda0 = self._std_to_logsigma(hist_window_std)
        # Level-0 std prediction: always use model_std (unchanged path, output endpoint)
        future_std0_logsigma = self._predict_level0_std_only(
            hist_lambda0,
            xbar_levels[0],
        )
        future_std0_pred = self._logsigma_to_std(future_std0_logsigma)

        future_mean_norm_pred: dict[int, torch.Tensor] = {}
        future_mean_pred: dict[int, torch.Tensor] = {}
        future_std_pred: dict[int, torch.Tensor] = {}
        future_mean_recon: dict[int, torch.Tensor] = {}
        hist_mean_norm_under: dict[tuple[int, int], torch.Tensor] = {}

        if max_level == 0:
            # No hierarchy: self.model predicts absolute mean
            future_mean_recon_0, _ = self._predict_level0_absolute_mean_std(
                hist_window_mean,
                hist_lambda0,
                xbar_levels[0],
                global_mean,
            )
            future_mean_recon[0] = future_mean_recon_0
        else:
            # Build hist hierarchy-normalized representations for levels 0..L-1
            for level_idx in range(max_level):
                spec = self.level_transition_specs[level_idx]
                lifted_mean = self._lift_seq_stats(
                    hist_mean_seq[level_idx + 1],
                    target_len=hist_mean_seq[level_idx].shape[1],
                    meta_patch=spec['meta_patch'],
                    meta_stride=spec['meta_stride'],
                )
                lifted_std = self._lift_seq_stats(
                    hist_std_seq[level_idx + 1],
                    target_len=hist_mean_seq[level_idx].shape[1],
                    meta_patch=spec['meta_patch'],
                    meta_stride=spec['meta_stride'],
                )
                hist_mean_norm_under[(level_idx + 1, level_idx)] = (
                    hist_mean_seq[level_idx] - lifted_mean
                ) / (lifted_std + self.epsilon)

            # Step 1: predict absolute future stats (mean + std) for levels 1..L
            for level_idx in range(1, self.num_levels):
                m_pred, s_pred = self.hier_stats_predictors[level_idx - 1](
                    hist_mean_seq[level_idx],
                    hist_std_seq[level_idx],
                    xbar_levels[level_idx],
                )
                future_mean_pred[level_idx] = m_pred
                future_std_pred[level_idx] = s_pred

            # Step 2: predict normalized future mean for levels 0..L-1
            # level-0: uses lifted hist_std_seq[1] as upper-std context branch
            # levels 1..L-1: uses hist_std_seq[level] directly
            for level_idx in range(max_level):
                spec = self.level_transition_specs[level_idx]
                if level_idx == 0:
                    lifted_upper_std = self._lift_seq_stats(
                        hist_std_seq[1],
                        target_len=hist_mean_seq[0].shape[1],
                        meta_patch=spec['meta_patch'],
                        meta_stride=spec['meta_stride'],
                    )
                    future_mean_norm_pred[0] = self.hier_norm_predictors[0](
                        hist_mean_seq[0],
                        lifted_upper_std,
                        xbar_levels[0],
                        hist_mean_norm_under[(1, 0)],
                    )
                else:
                    future_mean_norm_pred[level_idx] = self.hier_norm_predictors[level_idx](
                        hist_mean_seq[level_idx],
                        hist_std_seq[level_idx],
                        xbar_levels[level_idx],
                        hist_mean_norm_under[(level_idx + 1, level_idx)],
                    )

            # Step 3: top-down reconstruction
            # future_mean_recon[L] = future_mean_pred[L]
            # future_mean_recon[l] = future_mean_norm_pred[l] * lift(future_std_pred[l+1]) + lift(future_mean_pred[l+1])
            future_mean_recon[max_level] = future_mean_pred[max_level]
            for level_idx in range(max_level - 1, -1, -1):
                spec = self.level_transition_specs[level_idx]
                upper_mean_lift = self._lift_seq_stats(
                    future_mean_pred[level_idx + 1],
                    target_len=self.level_pred_lens[level_idx],
                    meta_patch=spec['meta_patch'],
                    meta_stride=spec['meta_stride'],
                )
                upper_std_lift = self._lift_seq_stats(
                    future_std_pred[level_idx + 1],
                    target_len=self.level_pred_lens[level_idx],
                    meta_patch=spec['meta_patch'],
                    meta_stride=spec['meta_stride'],
                )
                future_mean_recon[level_idx] = (
                    future_mean_norm_pred[level_idx] * (upper_std_lift + self.epsilon)
                    + upper_mean_lift
                )

        level_cache: list[dict] = []
        for level_idx in range(self.num_levels):
            cache = {
                'level': level_idx,
                'enabled': True,
                'hist_len': self.level_hist_lens[level_idx],
                'future_len': self.level_pred_lens[level_idx],
                'meta_patch': self.level_transition_specs[level_idx]['meta_patch']
                if level_idx < len(self.level_transition_specs)
                else 0,
                'meta_stride': self.level_transition_specs[level_idx]['meta_stride']
                if level_idx < len(self.level_transition_specs)
                else 0,
                'hist_mean_seq': hist_mean_seq[level_idx],
                'hist_std_seq': hist_std_seq[level_idx],
                'hist_xbar': xbar_levels[level_idx],
                'future_mean_recon': future_mean_recon[level_idx],
                'stats_loss': 0.0,
            }
            if max_level > 0 and level_idx < max_level:
                cache['hist_mean_norm_under'] = hist_mean_norm_under[(level_idx + 1, level_idx)]
                cache['future_mean_norm_pred'] = future_mean_norm_pred[level_idx]
            if max_level > 0 and level_idx >= 1:
                cache['future_mean_pred'] = future_mean_pred[level_idx]
                cache['future_std_pred'] = future_std_pred[level_idx]
            level_cache.append(cache)

        level_cache[0]['future_std0_pred'] = future_std0_pred
        level_cache[0]['future_std0_logsigma'] = self._std_to_logsigma(future_std0_pred)
        return level_cache

    def _build_overlap_summary(
        self,
        prev_pred: torch.Tensor,
        curr_pred: torch.Tensor,
    ) -> torch.Tensor:
        horizon = prev_pred.shape[1]
        if horizon <= 1:
            if self.ostn_use_patchwise_overlap_summary:
                return torch.zeros(
                    1,
                    0,
                    prev_pred.shape[2],
                    4,
                    device=prev_pred.device,
                    dtype=prev_pred.dtype,
                )
            return torch.zeros(
                1,
                prev_pred.shape[2],
                4,
                device=prev_pred.device,
                dtype=prev_pred.dtype,
            )

        residual = prev_pred[:, 1:, :] - curr_pred[:, :-1, :]
        if self.ostn_use_patchwise_overlap_summary:
            return torch.stack(
                [residual, residual.abs(), residual.pow(2), residual.sign()],
                dim=-1,
            )
        return torch.stack(
            [
                residual.mean(dim=1),
                residual.std(dim=1),
                residual.abs().mean(dim=1),
                residual.abs().max(dim=1)[0],
            ],
            dim=-1,
        )

    def _denorm_from_stats(self, pred_norm: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        time_stats = self._stats_to_time_stats(stats, pred_norm.shape[1])
        mean, std = self._split_stats(time_stats)
        return pred_norm * (std + self.epsilon) + mean

    def _norm_from_stats(self, pred_denorm: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        time_stats = self._stats_to_time_stats(stats, pred_denorm.shape[1])
        mean, std = self._split_stats(time_stats)
        return (pred_denorm - mean) / (std + self.epsilon)

    def reset_ostn_eval_state(self) -> None:
        self._prev_stream_final_pred = None
        self._prev_stream_base_final_pred = None
        self._prev_stream_refined_final_pred = None
        self._prev_stream_refined_mean = None
        self._prev_stream_refined_logsigma = None
        self._prev_stream_overlap_summary = None
        self._prev_stream_valid = False
        self._curr_refined_mean = None
        self._curr_refined_logsigma = None
        self._curr_base_mean = None
        self._curr_base_logsigma = None
        self._last_ostn_applied = False
        self._last_ostn_alpha_mean = 0.0
        self._last_ostn_alpha_max = 0.0
        self._last_ostn_delta_mu_abs_mean = 0.0
        self._last_ostn_delta_lsig_abs_mean = 0.0
        self._last_ostn_base_overlap_loss = 0.0
        self._last_ostn_refined_overlap_loss = 0.0
        self._last_ostn_base_mu_abs_mean = 0.0
        self._last_ostn_refined_mu_abs_mean = 0.0
        self._last_ostn_base_sigma_mean = 0.0
        self._last_ostn_refined_sigma_mean = 0.0

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        if self.station_type != 'adaptive':
            self._reset_prediction_cache()
            return input

        batch_size, length, dim = input.shape
        if length != self.seq_len:
            raise ValueError(f"SAN expected input length {self.seq_len}, got {length}.")

        self._reset_prediction_cache()
        x_for_stats = input.detach()
        hist_windows = self._extract_windows(x_for_stats)
        hist_mean, hist_std = self._compute_window_stats(hist_windows)
        hist_time_stats = self._window_stats_to_time_stats(hist_mean, hist_std, length)
        hist_time_mean, hist_time_std = self._split_stats(hist_time_stats)
        norm_input = (input - hist_time_mean) / (hist_time_std + self.epsilon)
        global_mean = input.mean(dim=1, keepdim=True)

        hierarchy_cache = self._predict_hierarchical_future_stats(
            hist_mean,
            hist_std,
            norm_input,
            global_mean,
        )
        level0_cache = hierarchy_cache[0]

        self._hierarchy_cache = hierarchy_cache
        self._coarse_level_cache = hierarchy_cache[1:]
        self._level0_hist_mu = hist_mean
        self._level0_hist_std = hist_std
        self._level0_hist_lambda = self._std_to_logsigma(hist_std)
        self._level0_future_final_mu = level0_cache['future_mean_recon']
        self._level0_future_final_lambda = level0_cache['future_std0_logsigma']
        self._level0_future_base_mu = self._level0_future_final_mu
        self._level0_future_base_lambda = self._level0_future_final_lambda

        base_std = level0_cache['future_std0_pred']
        base_stats = torch.cat([self._level0_future_final_mu, base_std], dim=-1)
        base_time_stats = self._window_stats_to_time_stats(
            self._level0_future_final_mu,
            base_std,
            self.pred_len,
        )
        fused_std = base_std
        fused_stats = torch.cat([self._level0_future_final_mu, fused_std], dim=-1)
        fused_time_stats = self._window_stats_to_time_stats(
            self._level0_future_final_mu,
            fused_std,
            self.pred_len,
        )
        self._base_pred_stats = base_stats
        self._base_pred_time_stats = base_time_stats
        self._refined_pred_stats = fused_stats
        self._refined_pred_time_stats = fused_time_stats

        use_ostn = (
            not self.training
            and self.ostn_enabled
            and self._prev_stream_valid
            and self._prev_stream_overlap_summary is not None
        )

        if use_ostn:
            hist_stats_seq = self._build_hist_stats_seq_from_stats(hist_mean, hist_std)
            overlap_summary = self._prev_stream_overlap_summary.expand(
                batch_size, *self._prev_stream_overlap_summary.shape[1:]
            )
            with torch.no_grad():
                delta_mu, delta_lsig, alpha = self.ostn_corrector(
                    hist_stats_seq,
                    self._level0_future_final_mu.detach(),
                    self._level0_future_final_lambda.detach(),
                    overlap_summary,
                )

            refined_mean = self._level0_future_final_mu + alpha * delta_mu
            refined_logsigma = self._level0_future_final_lambda + alpha * delta_lsig
            refined_std = self._logsigma_to_std(refined_logsigma)

            refined_stats = torch.cat([refined_mean, refined_std], dim=-1)
            refined_time_stats = self._window_stats_to_time_stats(
                refined_mean,
                refined_std,
                self.pred_len,
            )
            self._pred_stats = refined_stats
            self._pred_time_stats = refined_time_stats
            self._last_ostn_applied = True

            self._level0_future_ostn_mu = refined_mean
            self._level0_future_ostn_lambda = refined_logsigma
            self._curr_refined_mean = refined_mean.detach()
            self._curr_refined_logsigma = refined_logsigma.detach()
            self._curr_base_mean = self._level0_future_final_mu.detach()
            self._curr_base_logsigma = self._level0_future_final_lambda.detach()

            with torch.no_grad():
                self._last_ostn_alpha_mean = float(alpha.detach().mean().item())
                self._last_ostn_alpha_max = float(alpha.detach().max().item())
                self._last_ostn_delta_mu_abs_mean = float(delta_mu.detach().abs().mean().item())
                self._last_ostn_delta_lsig_abs_mean = float(delta_lsig.detach().abs().mean().item())
                self._last_ostn_base_mu_abs_mean = float(
                    self._level0_future_final_mu.abs().mean().item()
                )
                self._last_ostn_refined_mu_abs_mean = float(refined_mean.abs().mean().item())
                self._last_ostn_base_sigma_mean = float(fused_std.mean().item())
                self._last_ostn_refined_sigma_mean = float(refined_std.mean().item())
        else:
            self._pred_stats = fused_stats
            self._pred_time_stats = fused_time_stats
            self._last_ostn_applied = False

            self._level0_future_ostn_mu = self._level0_future_final_mu
            self._level0_future_ostn_lambda = self._level0_future_final_lambda
            self._curr_refined_mean = self._level0_future_final_mu.detach()
            self._curr_refined_logsigma = self._level0_future_final_lambda.detach()
            self._curr_base_mean = self._level0_future_final_mu.detach()
            self._curr_base_logsigma = self._level0_future_final_lambda.detach()

            with torch.no_grad():
                self._last_ostn_base_mu_abs_mean = float(
                    self._level0_future_final_mu.abs().mean().item()
                )
                self._last_ostn_base_sigma_mean = float(fused_std.mean().item())
                self._last_ostn_refined_mu_abs_mean = self._last_ostn_base_mu_abs_mean
                self._last_ostn_refined_sigma_mean = self._last_ostn_base_sigma_mean
                self._last_ostn_alpha_mean = 0.0
                self._last_ostn_alpha_max = 0.0
                self._last_ostn_delta_mu_abs_mean = 0.0
                self._last_ostn_delta_lsig_abs_mean = 0.0

        if self._oracle_pred_stats is not None:
            channels = self.channels
            if self._oracle_mode == 'mean_only':
                self._pred_stats = torch.cat(
                    [
                        self._oracle_pred_stats[:, :, :channels],
                        self._pred_stats[:, :, channels:],
                    ],
                    dim=-1,
                )
            elif self._oracle_mode == 'std_only':
                self._pred_stats = torch.cat(
                    [
                        self._pred_stats[:, :, :channels],
                        self._oracle_pred_stats[:, :, channels:],
                    ],
                    dim=-1,
                )
            else:
                self._pred_stats = self._oracle_pred_stats
            self._pred_time_stats = self._stats_to_time_stats(self._pred_stats, self.pred_len)

        self._final_future_time_stats = self._pred_time_stats
        return norm_input

    def update_ostn_stream_cache(self, final_pred: torch.Tensor) -> None:
        if not self.ostn_enabled or self.station_type != 'adaptive':
            return

        curr_tail = final_pred[-1:].detach()
        curr_base_tail = curr_tail
        curr_refined_tail = curr_tail
        if (
            self._curr_base_mean is not None
            and self._curr_base_logsigma is not None
            and self._curr_refined_mean is not None
            and self._curr_refined_logsigma is not None
        ):
            curr_base_std = self._curr_base_logsigma[-1:].detach().exp().clamp(min=self.sigma_min)
            curr_refined_std = self._curr_refined_logsigma[-1:].detach().exp().clamp(min=self.sigma_min)
            base_stats = torch.cat([self._curr_base_mean[-1:].detach(), curr_base_std], dim=-1)
            refined_stats = torch.cat(
                [self._curr_refined_mean[-1:].detach(), curr_refined_std],
                dim=-1,
            )
            norm_from_refined = self._norm_from_stats(curr_tail, refined_stats)
            curr_base_tail = self._denorm_from_stats(norm_from_refined, base_stats)
            curr_refined_tail = self._denorm_from_stats(norm_from_refined, refined_stats)

        if (
            self._prev_stream_valid
            and self._prev_stream_base_final_pred is not None
            and self._prev_stream_refined_final_pred is not None
        ):
            self._prev_stream_overlap_summary = self._build_overlap_summary(
                self._prev_stream_refined_final_pred,
                curr_refined_tail,
            )

            if self._prev_stream_base_final_pred.shape[1] > 1:
                base_residual = (
                    self._prev_stream_base_final_pred[:, 1:, :] - curr_base_tail[:, :-1, :]
                )
                self._last_ostn_base_overlap_loss = float(base_residual.pow(2).mean().item())
            else:
                self._last_ostn_base_overlap_loss = 0.0

            if self._prev_stream_refined_final_pred.shape[1] > 1:
                refined_residual = (
                    self._prev_stream_refined_final_pred[:, 1:, :] - curr_refined_tail[:, :-1, :]
                )
                self._last_ostn_refined_overlap_loss = float(
                    refined_residual.pow(2).mean().item()
                )
            else:
                self._last_ostn_refined_overlap_loss = 0.0

        if self._curr_refined_mean is not None:
            self._prev_stream_refined_mean = self._curr_refined_mean[-1:].detach()
        if self._curr_refined_logsigma is not None:
            self._prev_stream_refined_logsigma = self._curr_refined_logsigma[-1:].detach()

        self._prev_stream_base_final_pred = curr_base_tail
        self._prev_stream_refined_final_pred = curr_refined_tail
        self._prev_stream_final_pred = curr_tail
        self._prev_stream_valid = True

    def ostn_train_loss(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        x_t1: torch.Tensor,
        y_t1: torch.Tensor,
        prev_overlap_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, channels = x_t.shape
        device = x_t.device

        with torch.no_grad():
            hist_windows_t = self._extract_windows(x_t.detach())
            hist_mean_t, hist_std_t = self._compute_window_stats(hist_windows_t)
            hist_windows_t1 = self._extract_windows(x_t1.detach())
            hist_mean_t1, hist_std_t1 = self._compute_window_stats(hist_windows_t1)

            hist_time_stats_t = self._window_stats_to_time_stats(hist_mean_t, hist_std_t, x_t.shape[1])
            hist_time_mean_t, hist_time_std_t = self._split_stats(hist_time_stats_t)
            norm_input_t = (x_t - hist_time_mean_t) / (hist_time_std_t + self.epsilon)
            global_mean_t = x_t.mean(dim=1, keepdim=True)

            hist_time_stats_t1 = self._window_stats_to_time_stats(hist_mean_t1, hist_std_t1, x_t1.shape[1])
            hist_time_mean_t1, hist_time_std_t1 = self._split_stats(hist_time_stats_t1)
            norm_input_t1 = (x_t1 - hist_time_mean_t1) / (hist_time_std_t1 + self.epsilon)
            global_mean_t1 = x_t1.mean(dim=1, keepdim=True)

            base_level_t = self._predict_hierarchical_future_stats(
                hist_mean_t,
                hist_std_t,
                norm_input_t,
                global_mean_t,
            )[0]
            base_level_t1 = self._predict_hierarchical_future_stats(
                hist_mean_t1,
                hist_std_t1,
                norm_input_t1,
                global_mean_t1,
            )[0]
            base_mean_t = base_level_t['future_mean_recon']
            base_lsig_t = base_level_t['future_std0_logsigma']
            base_mean_t1 = base_level_t1['future_mean_recon']
            base_lsig_t1 = base_level_t1['future_std0_logsigma']
            hist_t = self._build_hist_stats_seq_from_stats(hist_mean_t, hist_std_t)
            hist_t1 = self._build_hist_stats_seq_from_stats(hist_mean_t1, hist_std_t1)

        oracle_windows_t = self._extract_windows(y_t)
        oracle_mu_t, oracle_std_t = self._compute_window_stats(oracle_windows_t)
        oracle_lsig_t = self._std_to_logsigma(oracle_std_t)
        oracle_windows_t1 = self._extract_windows(y_t1)
        oracle_mu_t1, oracle_std_t1 = self._compute_window_stats(oracle_windows_t1)
        oracle_lsig_t1 = self._std_to_logsigma(oracle_std_t1)

        delta_mu_oracle_t = oracle_mu_t - base_mean_t.detach()
        delta_lsig_oracle_t = oracle_lsig_t - base_lsig_t.detach()
        delta_mu_oracle_t1 = oracle_mu_t1 - base_mean_t1.detach()
        delta_lsig_oracle_t1 = oracle_lsig_t1 - base_lsig_t1.detach()

        if prev_overlap_summary is None:
            if self.ostn_use_patchwise_overlap_summary:
                overlap_t = torch.zeros(
                    batch_size,
                    max(self.pred_len - 1, 0),
                    channels,
                    4,
                    device=device,
                    dtype=x_t.dtype,
                )
            else:
                overlap_t = torch.zeros(batch_size, channels, 4, device=device, dtype=x_t.dtype)
        else:
            overlap_t = prev_overlap_summary.expand(batch_size, *prev_overlap_summary.shape[1:])

        delta_mu_t, delta_lsig_t, alpha_t = self.ostn_corrector(
            hist_t,
            base_mean_t.detach(),
            base_lsig_t.detach(),
            overlap_t,
        )

        teacher_residual = (
            base_mean_t.detach() + alpha_t * delta_mu_t
        )[:, 1:, :] - base_mean_t1.detach()[:, :-1, :]
        if self.ostn_use_patchwise_overlap_summary:
            overlap_t1 = torch.stack(
                [
                    teacher_residual,
                    teacher_residual.abs(),
                    teacher_residual.pow(2),
                    teacher_residual.sign(),
                ],
                dim=-1,
            )
        else:
            overlap_t1 = torch.stack(
                [
                    teacher_residual.mean(dim=1),
                    teacher_residual.std(dim=1),
                    teacher_residual.abs().mean(dim=1),
                    teacher_residual.abs().max(dim=1)[0],
                ],
                dim=-1,
            )

        delta_mu_t1, delta_lsig_t1, alpha_t1 = self.ostn_corrector(
            hist_t1,
            base_mean_t1.detach(),
            base_lsig_t1.detach(),
            overlap_t1.detach(),
        )

        refined_mean_t = base_mean_t.detach() + alpha_t * delta_mu_t
        refined_lsig_t = base_lsig_t.detach() + alpha_t * delta_lsig_t
        refined_std_t = self._logsigma_to_std(refined_lsig_t)

        refined_mean_t1 = base_mean_t1.detach() + alpha_t1 * delta_mu_t1
        refined_lsig_t1 = base_lsig_t1.detach() + alpha_t1 * delta_lsig_t1
        refined_std_t1 = self._logsigma_to_std(refined_lsig_t1)

        stats_t = torch.cat([refined_mean_t, refined_std_t], dim=-1)
        stats_t1 = torch.cat([refined_mean_t1, refined_std_t1], dim=-1)

        y_t_norm = self._norm_from_stats(y_t, stats_t)
        y_t1_norm = self._norm_from_stats(y_t1, stats_t1)
        y_t_corr = self._denorm_from_stats(y_t_norm, stats_t)
        y_t1_corr = self._denorm_from_stats(y_t1_norm, stats_t1)

        loss_stat = (
            F.mse_loss(delta_mu_t, delta_mu_oracle_t)
            + F.mse_loss(delta_lsig_t, delta_lsig_oracle_t)
            + F.mse_loss(delta_mu_t1, delta_mu_oracle_t1)
            + F.mse_loss(delta_lsig_t1, delta_lsig_oracle_t1)
        )
        if y_t_corr.shape[1] > 1 and y_t1_corr.shape[1] > 1:
            loss_overlap = F.mse_loss(y_t_corr[:, 1:, :], y_t1_corr[:, :-1, :])
        else:
            loss_overlap = torch.tensor(0.0, device=device)

        loss_alpha = alpha_t.mean() + alpha_t1.mean()
        return (
            self.ostn_stats_weight * loss_stat
            + self.ostn_overlap_weight * loss_overlap
            + self.ostn_alpha_l1 * loss_alpha
        )

    def denormalize(self, input: torch.Tensor, station_pred=None) -> torch.Tensor:
        if station_pred is None:
            station_pred = self._pred_time_stats if self._pred_time_stats is not None else self._pred_stats
        if self.station_type == 'adaptive' and station_pred is not None:
            time_stats = self._stats_to_time_stats(station_pred, input.shape[1])
            mean, std = self._split_stats(time_stats)
            return input * (std + self.epsilon) + mean
        return input

    def loss(self, true: torch.Tensor) -> torch.Tensor:
        if not self._hierarchy_cache or self.station_type != 'adaptive':
            return torch.tensor(0.0, device=true.device)

        max_level = self.num_levels - 1
        true_windows = self._extract_windows(true)
        oracle_mean0, oracle_std0 = self._compute_window_stats(true_windows)
        # _build_recursive_mean_hierarchy: oracle_std_seq[0] is zeros placeholder;
        # oracle_std_seq[l>=1] are real computed stds of level l-1 mean windowed slices.
        oracle_mean_seq, oracle_std_seq = self._build_recursive_mean_hierarchy(oracle_mean0)

        # Oracle normalized means for levels 0..L-1
        oracle_mean_norm: dict[int, torch.Tensor] = {}
        for level_idx in range(max_level):
            spec = self.level_transition_specs[level_idx]
            oracle_mean_lift = self._lift_seq_stats(
                oracle_mean_seq[level_idx + 1],
                target_len=oracle_mean_seq[level_idx].shape[1],
                meta_patch=spec['meta_patch'],
                meta_stride=spec['meta_stride'],
            )
            oracle_std_lift = self._lift_seq_stats(
                oracle_std_seq[level_idx + 1],
                target_len=oracle_mean_seq[level_idx].shape[1],
                meta_patch=spec['meta_patch'],
                meta_stride=spec['meta_stride'],
            )
            oracle_mean_norm[level_idx] = (
                oracle_mean_seq[level_idx] - oracle_mean_lift
            ) / (oracle_std_lift + self.epsilon)

        self._last_level_losses = {}
        level0_cache = self._hierarchy_cache[0]

        # L0_main = L_recon0 + L_std0
        recon0_loss = F.mse_loss(level0_cache['future_mean_recon'], oracle_mean_seq[0])
        std0_loss = F.mse_loss(level0_cache['future_std0_pred'], oracle_std0)
        l0_main = recon0_loss + std0_loss

        # Hierarchy losses (only when max_level >= 1)
        norm_losses: dict[int, torch.Tensor] = {}
        stats_losses: dict[int, torch.Tensor] = {}
        cons_losses: dict[int, torch.Tensor] = {}

        if max_level >= 1:
            # L_norm[l] for l = 0..L-1
            for level_idx in range(max_level):
                norm_losses[level_idx] = F.mse_loss(
                    self._hierarchy_cache[level_idx]['future_mean_norm_pred'],
                    oracle_mean_norm[level_idx],
                )
            # L_stats[l] = 0.5*(MSE_mean + MSE_std) for l = 1..L
            for level_idx in range(1, self.num_levels):
                cache_l = self._hierarchy_cache[level_idx]
                stats_losses[level_idx] = 0.5 * (
                    F.mse_loss(cache_l['future_mean_pred'], oracle_mean_seq[level_idx])
                    + F.mse_loss(cache_l['future_std_pred'], oracle_std_seq[level_idx])
                )
            # L_cons[l] = MSE(future_mean_recon[l], future_mean_pred[l]) for middle l = 1..L-1
            for level_idx in range(1, max_level):
                cache_l = self._hierarchy_cache[level_idx]
                cons_losses[level_idx] = F.mse_loss(
                    cache_l['future_mean_recon'],
                    cache_l['future_mean_pred'].detach(),
                )

        # Weighted hier total
        weighted_hier = torch.tensor(0.0, device=true.device, dtype=true.dtype)
        if max_level == 1:
            # L=1: norm[0](0.5) + stats[1](0.2)
            weighted_hier = (
                0.50 * norm_losses[0]
                + 0.20 * stats_losses[1]
            )
        elif max_level >= 2:
            # L>=2: norm[0](0.5) + norm[1](0.2) + stats[1](0.2) + stats[2](0.1) + cons[1](0.05)
            weighted_hier = (
                0.50 * norm_losses[0]
                + 0.20 * norm_losses[1]
                + 0.20 * stats_losses[1]
                + 0.10 * stats_losses[2]
                + 0.05 * cons_losses.get(1, torch.tensor(0.0, device=true.device, dtype=true.dtype))
            )
            for level_idx in range(3, self.num_levels):
                w = 0.05 / (2 ** (level_idx - 2))
                weighted_hier = weighted_hier + w * stats_losses[level_idx]

        total_loss = l0_main + weighted_hier

        # --- Update scalar diagnostics ---
        self._last_recon0_loss = float(recon0_loss.detach().item())
        self._last_std0_loss = float(std0_loss.detach().item())
        self._last_norm_losses = {l: float(v.detach().item()) for l, v in norm_losses.items()}
        self._last_stats_losses = {l: float(v.detach().item()) for l, v in stats_losses.items()}
        self._last_cons_losses = {l: float(v.detach().item()) for l, v in cons_losses.items()}
        self._last_weighted_hier_loss = float(weighted_hier.detach().item())
        self._last_total_stats_loss = float(total_loss.detach().item())
        self._last_hier_to_level0_ratio = self._last_weighted_hier_loss / max(
            float(l0_main.detach().item()), 1e-8
        )

        # --- Per-level cache diagnostics ---
        # Determine weighted_norm / weighted_stats per level
        _norm_w: dict[int, float] = {0: 0.50, 1: 0.20}
        _stats_w: dict[int, float] = {1: 0.20, 2: 0.10}
        _cons_w: dict[int, float] = {1: 0.05}
        if max_level == 1:
            _stats_w = {1: 0.20}
        for level_idx in range(self.num_levels):
            cache = self._hierarchy_cache[level_idx]
            nl = float(norm_losses[level_idx].detach().item()) if level_idx in norm_losses else 0.0
            sl = float(stats_losses[level_idx].detach().item()) if level_idx in stats_losses else 0.0
            cl = float(cons_losses[level_idx].detach().item()) if level_idx in cons_losses else 0.0
            sm_l = (
                float(F.mse_loss(cache['future_mean_pred'], oracle_mean_seq[level_idx]).detach().item())
                if level_idx in stats_losses
                else 0.0
            )
            ss_l = (
                float(F.mse_loss(cache['future_std_pred'], oracle_std_seq[level_idx]).detach().item())
                if level_idx in stats_losses
                else 0.0
            )
            w_nl = _norm_w.get(level_idx, 0.0) * nl
            w_sl = _stats_w.get(level_idx, 0.0) * sl
            w_cl = _cons_w.get(level_idx, 0.0) * cl
            cache['norm_loss'] = nl
            cache['stats_mean_loss'] = sm_l
            cache['stats_std_loss'] = ss_l
            cache['cons_loss'] = cl
            cache['weighted_stats_loss'] = w_nl + w_sl + w_cl
            cache['stats_loss'] = nl + sl
            self._last_level_losses[level_idx] = cache['stats_loss']

        return total_loss

    def get_last_hierarchical_stats(self) -> dict:
        levels = []
        for level_idx in range(self.MAX_EXTRA_LEVELS + 1):
            if level_idx < self.num_levels:
                cached = self._hierarchy_cache[level_idx] if level_idx < len(self._hierarchy_cache) else None
                levels.append(
                    {
                        'level': level_idx,
                        'enabled': True,
                        'hist_len': self.level_hist_lens[level_idx],
                        'future_len': self.level_pred_lens[level_idx],
                        'meta_patch': self.level_transition_specs[level_idx]['meta_patch']
                        if level_idx < len(self.level_transition_specs)
                        else 0,
                        'meta_stride': self.level_transition_specs[level_idx]['meta_stride']
                        if level_idx < len(self.level_transition_specs)
                        else 0,
                        'stats_loss': self._last_level_losses.get(level_idx, 0.0),
                        'weighted_stats_loss': float(cached.get('weighted_stats_loss', 0.0))
                        if cached is not None else 0.0,
                        'norm_loss': float(cached.get('norm_loss', 0.0))
                        if cached is not None else 0.0,
                        'stats_mean_loss': float(cached.get('stats_mean_loss', 0.0))
                        if cached is not None else 0.0,
                        'stats_std_loss': float(cached.get('stats_std_loss', 0.0))
                        if cached is not None else 0.0,
                        'cons_loss': float(cached.get('cons_loss', 0.0))
                        if cached is not None else 0.0,
                    }
                )
            else:
                levels.append(
                    {
                        'level': level_idx,
                        'enabled': False,
                        'hist_len': 0,
                        'future_len': 0,
                        'meta_patch': 0,
                        'meta_stride': 0,
                        'stats_loss': 0.0,
                        'weighted_stats_loss': 0.0,
                        'norm_loss': 0.0,
                        'stats_mean_loss': 0.0,
                        'stats_std_loss': 0.0,
                        'cons_loss': 0.0,
                    }
                )

        return {
            'num_levels': self.num_levels,
            'force_extra_levels': self.force_extra_levels,
            'extra_levels': self.num_levels - 1,
            'actual_extra_levels': self.num_levels - 1,
            'has_level0_absolute_mean_head': self.model is not None,
            'num_norm_predictors': len(self.hier_norm_predictors),
            'num_stats_predictors': len(self.hier_stats_predictors),
            'total_stats_loss': self._last_total_stats_loss,
            'recon0_loss': self._last_recon0_loss,
            'std0_loss': self._last_std0_loss,
            'norm_losses': dict(self._last_norm_losses),
            'stats_losses': dict(self._last_stats_losses),
            'cons_losses': dict(self._last_cons_losses),
            'weighted_hier_loss': self._last_weighted_hier_loss,
            'hier_to_level0_ratio': self._last_hier_to_level0_ratio,
            'levels': levels,
        }

    def get_last_ostn_stats(self) -> dict:
        zero = {
            'enabled': False,
            'applied': False,
            'alpha_mean': 0.0,
            'alpha_max': 0.0,
            'delta_mu_abs_mean': 0.0,
            'delta_logsigma_abs_mean': 0.0,
            'base_overlap_loss': 0.0,
            'refined_overlap_loss': 0.0,
            'base_mu_abs_mean': 0.0,
            'refined_mu_abs_mean': 0.0,
            'base_sigma_mean': 0.0,
            'refined_sigma_mean': 0.0,
            'hierarchical': self.get_last_hierarchical_stats(),
        }
        if not self.ostn_enabled:
            return zero
        return {
            'enabled': True,
            'applied': self._last_ostn_applied,
            'alpha_mean': self._last_ostn_alpha_mean,
            'alpha_max': self._last_ostn_alpha_max,
            'delta_mu_abs_mean': self._last_ostn_delta_mu_abs_mean,
            'delta_logsigma_abs_mean': self._last_ostn_delta_lsig_abs_mean,
            'base_overlap_loss': self._last_ostn_base_overlap_loss,
            'refined_overlap_loss': self._last_ostn_refined_overlap_loss,
            'base_mu_abs_mean': self._last_ostn_base_mu_abs_mean,
            'refined_mu_abs_mean': self._last_ostn_refined_mu_abs_mean,
            'base_sigma_mean': self._last_ostn_base_sigma_mean,
            'refined_sigma_mean': self._last_ostn_refined_sigma_mean,
            'hierarchical': self.get_last_hierarchical_stats(),
        }

    def forward(self, batch_x, mode='n', station_pred=None):
        if mode == 'n':
            return self.normalize(batch_x)
        if mode == 'd':
            return self.denormalize(batch_x, station_pred)
        return batch_x


class _MLP(nn.Module):
    def __init__(
        self,
        hist_stat_len: int,
        raw_hist_len: int,
        pred_stat_len: int,
        enc_in: int,
        mode: str,
    ):
        super().__init__()
        self.hist_stat_len = hist_stat_len
        self.raw_hist_len = raw_hist_len
        self.pred_stat_len = pred_stat_len
        self.channels = enc_in
        self.mode = mode
        hidden_dim = 512
        self.stats_input = nn.Linear(hist_stat_len, hidden_dim)
        self.raw_input = nn.Linear(raw_hist_len, hidden_dim)
        self.output = nn.Linear(2 * hidden_dim, pred_stat_len)

        if mode == 'mu':
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
        x_raw_like: torch.Tensor,
        global_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_stats = x_stats.permute(0, 2, 1)
        x_raw_like = x_raw_like.permute(0, 2, 1)

        stats_feat = self.stats_input(x_stats)
        raw_feat = self.raw_input(x_raw_like)
        pred = self.output(self.activation(torch.cat([stats_feat, raw_feat], dim=-1)))
        pred = self.final_activation(pred).permute(0, 2, 1)

        if self.mode == 'mu' and global_anchor is not None:
            anchor = global_anchor.expand(-1, self.pred_stat_len, -1)
            pred = (
                pred * self.weight[0].view(1, 1, -1)
                + anchor * self.weight[1].view(1, 1, -1)
            )

        return pred


class _HierStatsPredictor(nn.Module):
    """Predict absolute future mean and std for hierarchy levels >= 1.

    Used as `hier_stats_predictors[l-1]` for level l in 1..L.
    All three inputs are fused through a shared trunk before forking into
    two output heads.

    Inputs  (B, N_l, C):
        hist_mu_mean  – hist_mean_seq[l]
        hist_mu_std   – hist_std_seq[l]
        xbar_raw_like – patch-averaged raw-input representation at level l
    Outputs (B, pred_l, C):
        future_mean_pred  – absolute future mean at level l
        future_std_pred   – absolute future std at level l (always >= sigma_min)
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
        self.pred_stat_len = pred_stat_len
        self.sigma_min = sigma_min

        self.enc_mu_mean = nn.Sequential(nn.Linear(hist_stat_len, hidden_dim), nn.GELU())
        self.enc_mu_std  = nn.Sequential(nn.Linear(hist_stat_len, hidden_dim), nn.GELU())
        self.enc_xbar    = nn.Sequential(nn.Linear(raw_hist_len,  hidden_dim), nn.GELU())

        self.trunk = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU())

        self.head_mean = nn.Linear(hidden_dim, pred_stat_len)
        self.head_std  = nn.Linear(hidden_dim, pred_stat_len)

    def forward(
        self,
        hist_mu_mean: torch.Tensor,
        hist_mu_std: torch.Tensor,
        xbar_raw_like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, L, C) -> (B, C, L) for time-axis linear
        h_mean = self.enc_mu_mean(hist_mu_mean.permute(0, 2, 1))
        h_std  = self.enc_mu_std(hist_mu_std.permute(0, 2, 1))
        h_xbar = self.enc_xbar(xbar_raw_like.permute(0, 2, 1))

        h_trunk = self.trunk(torch.cat([h_mean, h_std, h_xbar], dim=-1))

        # (B, C, pred_stat_len) -> (B, pred_stat_len, C)
        future_mean_pred = self.head_mean(h_trunk).permute(0, 2, 1)
        future_std_pred  = (
            F.softplus(self.head_std(h_trunk)).permute(0, 2, 1).clamp(min=self.sigma_min)
        )
        return future_mean_pred, future_std_pred


class _HierNormPredictor(nn.Module):
    """Predict normalized future mean for non-top hierarchy levels."""

    def __init__(
        self,
        hist_stat_len: int,
        raw_hist_len: int,
        pred_stat_len: int,
        enc_in: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.pred_stat_len = pred_stat_len

        self.enc_mu_mean = nn.Sequential(nn.Linear(hist_stat_len, hidden_dim), nn.GELU())
        self.enc_mu_std = nn.Sequential(nn.Linear(hist_stat_len, hidden_dim), nn.GELU())
        self.enc_xbar = nn.Sequential(nn.Linear(raw_hist_len, hidden_dim), nn.GELU())
        self.enc_norm = nn.Sequential(nn.Linear(hist_stat_len, hidden_dim), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU())
        self.head_norm = nn.Linear(hidden_dim, pred_stat_len)

    def forward(
        self,
        hist_mu_mean: torch.Tensor,
        hist_mu_std: torch.Tensor,
        xbar_raw_like: torch.Tensor,
        hist_mean_norm_under: torch.Tensor,
    ) -> torch.Tensor:
        h_mean = self.enc_mu_mean(hist_mu_mean.permute(0, 2, 1))
        h_std = self.enc_mu_std(hist_mu_std.permute(0, 2, 1))
        h_xbar = self.enc_xbar(xbar_raw_like.permute(0, 2, 1))
        h_norm = self.enc_norm(hist_mean_norm_under.permute(0, 2, 1))
        h_trunk = self.trunk(torch.cat([h_mean, h_std, h_xbar, h_norm], dim=-1))
        return self.head_norm(h_trunk).permute(0, 2, 1)
