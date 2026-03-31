# Sliding-window SAN with uniform multi-level hierarchy prediction.
#
# Design principle: every level first performs a full original-SAN-style base
# prediction using that level's own history stats, xbar signal, and that level's
# own history-mean anchor. Hierarchy then acts only outside the base predictor,
# adding a second-order norm on mean for non-top levels.
#
# Per-level structure (every level l = 0..L):
#   - mean head: anchor + residual style (same as original single-level SAN)
#       anchor_l = hist_mean_seq[l].mean(dim=1, keepdim=True)
#       future_mean_base[l] = mean_head_l(hist_mean_seq[l], xbar_levels[l], anchor_l)
#   - std head:  direct sigma prediction style
#       future_std_base[l] = std_head_l(hist_std[l], xbar[l])
#   Both heads receive the level's own xbar_levels[l] auxiliary signal.
#   Only the top level base-mean head is directly supervised against oracle
#   future mean. Std heads remain supervised at every level. Non-top mean
#   supervision is applied only on future_mean_final[l]. The z branch remains
#   an implicit latent path used to construct future_mean_final[l], but it no
#   longer receives its own explicit oracle-MSE target.
#
# Second-order norm (for non-top levels l < L only):
#   hist_norm_z[l]        = (hist_mean_seq[l] - lift(hist_mean_seq[l+1]))
#                           / (lift(hist_std_seq[l+1]) + eps)
#   future_norm_z_pred[l] = norm_z_head_l(...)
#   Final mean:           future_mean_final[l] = lift(future_mean_final[l+1])
#                         + future_norm_z_pred[l] * (lift(future_std_base[l+1]) + eps)
#
# Top level (l == L):
#   future_mean_final[L] = future_mean_base[L]  (no upper level)
#
# Level-0 denorm uses future_mean_final[0] and future_std_base[0].
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self._validate_config_lengths()
        self.hist_stat_len = self._compute_n_windows(self.seq_len)
        self.pred_stat_len = self._compute_n_windows(self.pred_len)
        self.epsilon = 1e-5
        self._build_level_layout()
        self._build_model()
        self._reset_prediction_cache()

    def _reset_prediction_cache(self) -> None:
        self._pred_stats: Optional[torch.Tensor] = None
        self._pred_time_stats: Optional[torch.Tensor] = None

        self._level0_hist_mu: Optional[torch.Tensor] = None
        self._level0_hist_std: Optional[torch.Tensor] = None
        self._level0_hist_lambda: Optional[torch.Tensor] = None
        self._level0_future_final_mu: Optional[torch.Tensor] = None
        self._level0_future_final_lambda: Optional[torch.Tensor] = None

        self._hierarchy_cache: list[dict] = []
        self._coarse_level_cache: list[dict] = []
        self._last_level_losses: dict[int, float] = {}
        self._last_total_stats_loss: float = 0.0
        self._last_recon0_loss: float = 0.0
        self._last_std0_loss: float = 0.0
        # Per-level diagnostics
        self._last_base_mean_losses: dict[int, float] = {}   # L_base_mean[l] for l=0..L
        self._last_std_losses: dict[int, float] = {}         # L_std[l] for l=0..L
        self._last_final_mean_losses: dict[int, float] = {}  # L_final_mean[l] for l=0..L-1
        self._last_top_base_mean_loss: float = 0.0
        self._last_weighted_base_loss: float = 0.0
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
        return torch.log(std.clamp(min=self.sigma_min)).clamp(min=-6.0, max=6.0)

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

    def _build_recursive_stats_hierarchy(
        self,
        level0_mean_seq: torch.Tensor,
        level0_std_seq: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Build hierarchy where level0 std is real window std and higher-level std
        is the patch std of the previous level mean sequence.
        """
        level_mean_seqs: list[torch.Tensor] = [level0_mean_seq]
        level_std_seqs: list[torch.Tensor] = [level0_std_seq]
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

    def _build_model(self):
        self.level_raw_hist_lens = [hist_len * self.window_len for hist_len in self.level_hist_lens]
        # One _SANLevelPredictor per level, all structurally identical.
        # Each predictor has: mean head (anchor+residual), std head, and — for non-top levels —
        # a norm-z head that predicts the second-order normed residual under the upper level.
        self.level_predictors = nn.ModuleList(
            [
                _SANLevelPredictor(
                    hist_stat_len=self.level_hist_lens[l],
                    raw_hist_len=self.level_raw_hist_lens[l],
                    pred_stat_len=self.level_pred_lens[l],
                    enc_in=self.enc_in,
                    sigma_min=self.sigma_min,
                    has_norm_z_head=(l < self.num_levels - 1),
                ).float()
                for l in range(self.num_levels)
            ]
        )
        # Keep legacy attribute references for external code that inspects self.model / self.model_std.
        # Both point into level_predictors[0] sub-modules so no extra parameters are created.
        self.model = self.level_predictors[0].mean_head
        self.model_std = self.level_predictors[0].std_head

    def _predict_hierarchical_future_stats(
        self,
        hist_window_mean: torch.Tensor,
        hist_window_std: torch.Tensor,
        norm_input: torch.Tensor,
        global_mean: torch.Tensor,
    ) -> list[dict]:
        """Predict future stats for all levels using the uniform SAN-style design.

        Every level first produces its own original-SAN-style base outputs using:
            hist_mean_seq[l], hist_std_seq[l], xbar_levels[l], anchor_l
        where anchor_l = hist_mean_seq[l].mean(dim=1, keepdim=True).

        Non-top levels then apply an outer mean-only second-order norm:
            future_mean_final[l] = lift(future_mean_final[l+1])
                                   + zhat_l * (lift(future_std_base[l+1]) + eps)
        while the top level keeps future_mean_final[top] = future_mean_base[top].
        """
        max_level = self.num_levels - 1
        hist_mean_seq, hist_std_seq = self._build_recursive_stats_hierarchy(
            hist_window_mean,
            hist_window_std,
        )
        xbar_levels = self._build_xbar_hierarchy(norm_input)

        # Storage
        future_mean_base: dict[int, torch.Tensor] = {}
        future_std_base: dict[int, torch.Tensor] = {}
        future_mean_final: dict[int, torch.Tensor] = {}
        # For non-top levels: hist normed-z and predicted normed-z
        hist_norm_z: dict[int, torch.Tensor] = {}       # z_l computed from hist
        future_norm_z_pred: dict[int, torch.Tensor] = {}  # zhat_l predicted

        # ---------------------------------------------------------------
        # Step 1: every level independently performs a base SAN prediction using
        # that level's own history-mean anchor.
        # ---------------------------------------------------------------
        for level_idx in range(max_level, -1, -1):
            predictor = self.level_predictors[level_idx]
            anchor = hist_mean_seq[level_idx].mean(dim=1, keepdim=True)

            mb, sb = predictor.predict_base(
                hist_mean_seq[level_idx],
                hist_std_seq[level_idx],
                xbar_levels[level_idx],
                anchor,
            )
            future_mean_base[level_idx] = mb
            future_std_base[level_idx] = sb

        # ---------------------------------------------------------------
        # Step 2: compute hist normed-z for non-top levels (needed as input to norm_z_head).
        # z_l = (hist_mean[l] - lift(hist_mean[l+1])) / (lift(hist_std[l+1]) + eps)
        # level0 std comes from real window std; level>=1 std comes from patch std
        # of the previous level mean sequence.
        # ---------------------------------------------------------------
        for level_idx in range(max_level):
            spec = self.level_transition_specs[level_idx]
            upper_hist_mean_lift = self._lift_seq_stats(
                hist_mean_seq[level_idx + 1],
                target_len=hist_mean_seq[level_idx].shape[1],
                meta_patch=spec['meta_patch'],
                meta_stride=spec['meta_stride'],
            )
            upper_hist_std_lift = self._lift_seq_stats(
                hist_std_seq[level_idx + 1],
                target_len=hist_mean_seq[level_idx].shape[1],
                meta_patch=spec['meta_patch'],
                meta_stride=spec['meta_stride'],
            )
            hist_norm_z[level_idx] = (
                hist_mean_seq[level_idx] - upper_hist_mean_lift
            ) / (upper_hist_std_lift + self.epsilon)

        # ---------------------------------------------------------------
        # Step 3: for non-top levels, predict zhat_l and compute final mean.
        # final_mean[l] = lift(final_mean[l+1]) + zhat_l * (lift(std_base[l+1]) + eps)
        # ---------------------------------------------------------------
        future_mean_final[max_level] = future_mean_base[max_level]

        for level_idx in range(max_level - 1, -1, -1):
            spec = self.level_transition_specs[level_idx]
            upper_mean_lift = self._lift_seq_stats(
                future_mean_final[level_idx + 1],
                target_len=self.level_pred_lens[level_idx],
                meta_patch=spec['meta_patch'],
                meta_stride=spec['meta_stride'],
            )
            upper_std_lift = self._lift_seq_stats(
                future_std_base[level_idx + 1],
                target_len=self.level_pred_lens[level_idx],
                meta_patch=spec['meta_patch'],
                meta_stride=spec['meta_stride'],
            )
            # Upper-std context in hist domain (for norm_z_head input)
            upper_hist_std_lift = self._lift_seq_stats(
                hist_std_seq[level_idx + 1],
                target_len=hist_mean_seq[level_idx].shape[1],
                meta_patch=spec['meta_patch'],
                meta_stride=spec['meta_stride'],
            )
            zhat = self.level_predictors[level_idx].predict_norm_z(
                hist_mean_seq[level_idx],
                upper_hist_std_lift,
                xbar_levels[level_idx],
                hist_norm_z[level_idx],
            )
            future_norm_z_pred[level_idx] = zhat
            future_mean_final[level_idx] = upper_mean_lift + zhat * (upper_std_lift + self.epsilon)

        if max_level == 0:
            # Single-level: no upper level, final == base.
            future_mean_final[0] = future_mean_base[0]

        # ---------------------------------------------------------------
        # Assemble level cache
        # ---------------------------------------------------------------
        level_cache: list[dict] = []
        for level_idx in range(self.num_levels):
            cache = {
                'level': level_idx,
                'enabled': True,
                'hist_len': self.level_hist_lens[level_idx],
                'future_len': self.level_pred_lens[level_idx],
                'meta_patch': self.level_transition_specs[level_idx]['meta_patch']
                if level_idx < len(self.level_transition_specs) else 0,
                'meta_stride': self.level_transition_specs[level_idx]['meta_stride']
                if level_idx < len(self.level_transition_specs) else 0,
                'hist_mean_seq': hist_mean_seq[level_idx],
                'hist_std_seq': hist_std_seq[level_idx],
                'hist_xbar': xbar_levels[level_idx],
                'future_mean_base': future_mean_base[level_idx],
                'future_std_base': future_std_base[level_idx],
                'future_mean_final': future_mean_final[level_idx],
                # legacy key used by denorm / loss callers
                'future_mean_recon': future_mean_final[level_idx],
                'stats_loss': 0.0,
            }
            if level_idx < max_level:
                cache['hist_norm_z'] = hist_norm_z[level_idx]
                cache['future_norm_z_pred'] = future_norm_z_pred[level_idx]
            level_cache.append(cache)

        # Level-0 std bookkeeping (used by denorm)
        level_cache[0]['future_std0_pred'] = future_std_base[0]
        level_cache[0]['future_std0_logsigma'] = self._std_to_logsigma(future_std_base[0])
        return level_cache

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
        self._level0_future_final_mu = level0_cache['future_mean_final']
        self._level0_future_final_lambda = level0_cache['future_std0_logsigma']

        future_mu = level0_cache['future_mean_final']
        future_std = level0_cache['future_std_base']
        self._pred_stats = torch.cat([future_mu, future_std], dim=-1)
        self._pred_time_stats = self._window_stats_to_time_stats(future_mu, future_std, self.pred_len)

        return norm_input

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
        oracle_mean_seq_l0, oracle_std_seq_l0 = self._compute_window_stats(true_windows)
        oracle_mean_seq, oracle_std_seq = self._build_recursive_stats_hierarchy(
            oracle_mean_seq_l0,
            oracle_std_seq_l0,
        )

        self._last_level_losses = {}

        # --- Per-level losses ---
        # Top-level explicit mean supervision: L_base_mean[top]
        # Non-top explicit mean supervision:   L_final_mean[l]
        # All-level sigma supervision:         L_std[l]
        # future_norm_z_pred remains in the forward path, but is trained only
        # implicitly through final_mean_loss.
        base_mean_losses: dict[int, torch.Tensor] = {}
        std_losses: dict[int, torch.Tensor] = {}
        final_mean_losses: dict[int, torch.Tensor] = {}

        for level_idx in range(self.num_levels):
            cache_l = self._hierarchy_cache[level_idx]
            if level_idx == max_level:
                base_mean_losses[level_idx] = F.mse_loss(
                    cache_l['future_mean_base'], oracle_mean_seq[level_idx]
                )
            std_losses[level_idx] = F.mse_loss(
                cache_l['future_std_base'], oracle_std_seq[level_idx]
            )

        for level_idx in range(max_level):
            cache_l = self._hierarchy_cache[level_idx]
            final_mean_losses[level_idx] = F.mse_loss(
                cache_l['future_mean_final'], oracle_mean_seq[level_idx]
            )

        top_level_weight = 0.20 / (2.0 ** max(max_level - 1, 0)) if self.num_levels > 1 else 1.0
        weighted_base = top_level_weight * base_mean_losses[max_level]
        for level_idx in range(self.num_levels):
            lw = 1.0 if level_idx == 0 else 0.20 / (2.0 ** (level_idx - 1))
            weighted_base = weighted_base + lw * std_losses[level_idx]

        weighted_hier = torch.tensor(0.0, device=true.device, dtype=true.dtype)
        for level_idx in range(max_level):
            lw = 0.50 / (2.0 ** level_idx)
            weighted_hier = weighted_hier + lw * final_mean_losses[level_idx]

        total_loss = weighted_base + weighted_hier

        # --- Scalar diagnostics ---
        recon0_loss = final_mean_losses.get(0, base_mean_losses[max_level])
        std0_loss = std_losses[0]
        self._last_recon0_loss = float(recon0_loss.detach().item())
        self._last_std0_loss = float(std0_loss.detach().item())
        self._last_base_mean_losses = {l: float(v.detach().item()) for l, v in base_mean_losses.items()}
        self._last_std_losses = {l: float(v.detach().item()) for l, v in std_losses.items()}
        self._last_final_mean_losses = {l: float(v.detach().item()) for l, v in final_mean_losses.items()}
        self._last_top_base_mean_loss = float(base_mean_losses[max_level].detach().item())
        self._last_weighted_base_loss = float(weighted_base.detach().item())
        self._last_weighted_hier_loss = float(weighted_hier.detach().item())
        self._last_total_stats_loss = float(total_loss.detach().item())
        self._last_hier_to_level0_ratio = self._last_weighted_hier_loss / max(
            float(weighted_base.detach().item()), 1e-8
        )

        # --- Per-level cache diagnostics ---
        for level_idx in range(self.num_levels):
            cache = self._hierarchy_cache[level_idx]
            bml = float(base_mean_losses[level_idx].detach().item()) if level_idx in base_mean_losses else 0.0
            sl = float(std_losses[level_idx].detach().item())
            fml = float(final_mean_losses[level_idx].detach().item()) if level_idx in final_mean_losses else 0.0
            cache['base_mean_loss'] = bml
            cache['std_loss'] = sl
            cache['final_mean_loss'] = fml
            cache['stats_loss'] = bml + sl + fml
            if level_idx == max_level:
                cache['weighted_stats_loss'] = top_level_weight * bml + (
                    (1.0 if level_idx == 0 else 0.20 / (2.0 ** (level_idx - 1))) * sl
                )
            elif level_idx == 0:
                cache['weighted_stats_loss'] = sl + 0.50 * fml
            else:
                level_weight = 0.20 / (2.0 ** (level_idx - 1))
                cache['weighted_stats_loss'] = level_weight * sl + (
                    0.50 / (2.0 ** level_idx)
                ) * fml
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
                        if level_idx < len(self.level_transition_specs) else 0,
                        'meta_stride': self.level_transition_specs[level_idx]['meta_stride']
                        if level_idx < len(self.level_transition_specs) else 0,
                        'stats_loss': self._last_level_losses.get(level_idx, 0.0),
                        'weighted_stats_loss': float(cached.get('weighted_stats_loss', 0.0))
                        if cached is not None else 0.0,
                        'base_mean_loss': float(cached.get('base_mean_loss', 0.0))
                        if cached is not None else 0.0,
                        'std_loss': float(cached.get('std_loss', 0.0))
                        if cached is not None else 0.0,
                        'final_mean_loss': float(cached.get('final_mean_loss', 0.0))
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
                        'base_mean_loss': 0.0,
                        'std_loss': 0.0,
                        'final_mean_loss': 0.0,
                    }
                )

        return {
            'num_levels': self.num_levels,
            'force_extra_levels': self.force_extra_levels,
            'extra_levels': self.num_levels - 1,
            'actual_extra_levels': self.num_levels - 1,
            'has_level0_absolute_mean_head': True,
            'num_level_predictors': len(self.level_predictors),
            'total_stats_loss': self._last_total_stats_loss,
            'recon0_loss': self._last_recon0_loss,
            'std0_loss': self._last_std0_loss,
            'top_base_mean_loss': self._last_top_base_mean_loss,
            'weighted_base': self._last_weighted_base_loss,
            'base_mean_losses': dict(self._last_base_mean_losses),
            'std_losses': dict(self._last_std_losses),
            'final_mean_losses': dict(self._last_final_mean_losses),
            'weighted_hier_loss': self._last_weighted_hier_loss,
            'hier_to_level0_ratio': self._last_hier_to_level0_ratio,
            'levels': levels,
        }

    def get_last_aux_stats(self) -> dict:
        stats: dict[str, float] = {
            'aux_total': self._last_total_stats_loss,
            'weighted_base': self._last_weighted_base_loss,
            'weighted_hier': self._last_weighted_hier_loss,
            'total_stats_loss': self._last_total_stats_loss,
            'top_base_mean_loss': self._last_top_base_mean_loss,
            'L_easy': 0.0,
            'L_white': 0.0,
            'L_js': 0.0,
            'L_w1': 0.0,
            'L_min': 0.0,
            'L_e_tv': 0.0,
            'ratio_n_bc_mean': 0.0,
            'ratio_n_bc_min': 0.0,
            'ratio_n_bc_max': 0.0,
            'loss_n_ratio_budget': 0.0,
            'pred_n_loss': 0.0,
            'L_sparse': 0.0,
            'L_u_tv': 0.0,
        }
        for level_idx in range(self.MAX_EXTRA_LEVELS + 1):
            stats[f'std_loss_l{level_idx}'] = self._last_std_losses.get(level_idx, 0.0)
            stats[f'final_mean_loss_l{level_idx}'] = self._last_final_mean_losses.get(level_idx, 0.0)
        return stats

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
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

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



class _SANLevelPredictor(nn.Module):
    """Uniform SAN-style predictor for a single hierarchy level.

    Every level has the same structure:
      - mean_head: anchor + residual prediction of future mean (same as original SAN).
      - std_head:  direct sigma prediction of future std.
      - norm_z_head (non-top levels only): predicts the normed residual under the upper level.

    Inputs to predict_base (B, N_l, C):
        hist_mean   – hist_mean_seq[l]
        hist_std    – hist_std_seq[l]
        xbar        – xbar_levels[l] (normed input signal at this level)
        anchor      – (B, 1, C) anchor for mean head, always from this level's own
                      history mean: hist_mean_seq[l].mean(dim=1, keepdim=True)

    Inputs to predict_norm_z (non-top only):
        hist_mean        – hist_mean_seq[l]
        upper_hist_std   – lifted hist_std_seq[l+1] (upper-level std context)
        xbar             – xbar_levels[l]
        hist_norm_z      – hist normed residual z_l = (hist_mean[l] - upper_hist_mean_lift)/(upper_std+eps)
    """

    def __init__(
        self,
        hist_stat_len: int,
        raw_hist_len: int,
        pred_stat_len: int,
        enc_in: int,
        sigma_min: float = 1e-3,
        has_norm_z_head: bool = True,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.pred_stat_len = pred_stat_len
        self.sigma_min = sigma_min
        self.has_norm_z_head = has_norm_z_head

        # mean_head: reuses _MLP in 'mu' mode (anchor + residual)
        self.mean_head = _MLP(
            hist_stat_len=hist_stat_len,
            raw_hist_len=raw_hist_len,
            pred_stat_len=pred_stat_len,
            enc_in=enc_in,
            mode='mu',
        )

        # std_head: reuses _MLP in 'lambda' mode (direct sigma prediction)
        self.std_head = _MLP(
            hist_stat_len=hist_stat_len,
            raw_hist_len=raw_hist_len,
            pred_stat_len=pred_stat_len,
            enc_in=enc_in,
            mode='lambda',
        )

        # norm_z_head: predicts zhat = z (normed residual under upper level); non-top only
        if has_norm_z_head:
            self.enc_mean  = nn.Sequential(nn.Linear(hist_stat_len, hidden_dim), nn.GELU())
            self.enc_ustd  = nn.Sequential(nn.Linear(hist_stat_len, hidden_dim), nn.GELU())
            self.enc_xbar  = nn.Sequential(nn.Linear(raw_hist_len, hidden_dim), nn.GELU())
            self.enc_normz = nn.Sequential(nn.Linear(hist_stat_len, hidden_dim), nn.GELU())
            self.trunk_z   = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU())
            self.head_z    = nn.Linear(hidden_dim, pred_stat_len)
            nn.init.zeros_(self.head_z.weight)
            nn.init.zeros_(self.head_z.bias)

    def predict_base(
        self,
        hist_mean: torch.Tensor,
        hist_std: torch.Tensor,
        xbar: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (future_mean_base, future_std_base) for this level."""
        future_mean = self.mean_head(
            hist_mean - anchor.expand(-1, hist_mean.shape[1], -1),
            xbar,
            anchor,
        )
        future_std = self.std_head(
            hist_std,
            xbar,
            None,
        ).clamp(min=self.sigma_min)
        return future_mean, future_std

    def predict_norm_z(
        self,
        hist_mean: torch.Tensor,
        upper_hist_std: torch.Tensor,
        xbar: torch.Tensor,
        hist_norm_z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict zhat_l (normed residual under upper level). Only valid when has_norm_z_head."""
        if not self.has_norm_z_head:
            raise ValueError("predict_norm_z called on a top-level predictor without norm_z_head.")
        h_mean  = self.enc_mean(hist_mean.permute(0, 2, 1))
        h_ustd  = self.enc_ustd(upper_hist_std.permute(0, 2, 1))
        h_xbar  = self.enc_xbar(xbar.permute(0, 2, 1))
        h_normz = self.enc_normz(hist_norm_z.permute(0, 2, 1))
        h_trunk = self.trunk_z(torch.cat([h_mean, h_ustd, h_xbar, h_normz], dim=-1))
        return self.head_z(h_trunk).permute(0, 2, 1)
