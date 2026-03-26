# Adapted from FAN/torch_timeseries/normalizations/FAN.py
# Changes vs FAN original:
#   - main_freq_part accepts ablation_mode and returns (norm_input, x_filtered, indices).
#   - FAN.__init__ accepts ablation_mode; adds per-phase frequency-bin stat accumulators.
#   - FAN.normalize() accumulates per-batch selected_bin_mean/std/low_ratio stats.
#   - FAN exposes reset_freq_stats() / get_freq_stats() for train.py hooks.
import torch
import torch.nn as nn


def main_freq_part(x, k, rfft=True, ablation_mode="original"):
    """Select and reconstruct the frequency-based 'main' component of x.

    Args:
        x:             (B, T, N) input tensor.
        k:             number of frequency bins to select.
        rfft:          use rfft (True) or fft (False).
        ablation_mode: "original"          — top-K bins by amplitude (original FAN).
                       "low_only"          — lowest K bins by frequency index (DC-first),
                                            no amplitude sorting.
                       "topk_exclude_low"  — mask the lowest K bins, then top-K by
                                            amplitude on the remaining bins.

    Returns:
        norm_input:  x minus the filtered signal.
        x_filtered:  reconstructed main component.
        indices:     (B, K, N) int64 — selected bin indices (same shape as original).
    """
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)

    if ablation_mode == "original":
        k_values = torch.topk(xf.abs(), k, dim=1)
        indices = k_values.indices                           # (B, K, N)

    elif ablation_mode == "low_only":
        # Select the K lowest-frequency bins (0, 1, ..., K-1) for every
        # (sample, channel) pair — same selection granularity as original top-K.
        B, _F, N = xf.shape
        indices = (
            torch.arange(k, device=xf.device, dtype=torch.long)
            .view(1, k, 1)
            .expand(B, k, N)
        )                                                    # (B, K, N)

    elif ablation_mode == "topk_exclude_low":
        # Zero out the K lowest-frequency bins in the amplitude tensor, then
        # select the top-K bins by amplitude from the remainder.
        abs_val = xf.abs().clone()
        abs_val[:, :k, :] = 0.0
        k_values = torch.topk(abs_val, k, dim=1)
        indices = k_values.indices                           # (B, K, N)

    else:
        raise ValueError(f"Unknown fan_ablation_mode: {ablation_mode!r}")

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    norm_input = x - x_filtered
    return norm_input, x_filtered, indices


class FAN(nn.Module):
    """FAN first subtracts the bottom-k frequency component from the original series."""

    def __init__(self, seq_len, pred_len, enc_in, freq_topk=20, rfft=True,
                 ablation_mode="original", **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        print("freq_topk : ", self.freq_topk)
        self.rfft = rfft
        self.ablation_mode = ablation_mode

        # Per-phase frequency-bin statistics accumulators.
        # Reset by reset_freq_stats() at the start of each phase (train/val/test).
        self._freq_stat_sum_mean:      float = 0.0
        self._freq_stat_sum_std:       float = 0.0
        self._freq_stat_sum_low_ratio: float = 0.0
        self._freq_stat_count:         int   = 0

        # Per-batch prediction quality diagnostics (set in loss(), read via get_debug_scalars())
        self._last_fan_main_mse:          float = float("nan")
        self._last_fan_res_mse:           float = float("nan")
        self._last_fan_main_energy_ratio: float = float("nan")
        self._last_fan_res_energy_ratio:  float = float("nan")

        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def _build_model(self):
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)

    def reset_freq_stats(self) -> None:
        """Reset per-phase frequency-bin statistics before each train/val/test pass."""
        self._freq_stat_sum_mean      = 0.0
        self._freq_stat_sum_std       = 0.0
        self._freq_stat_sum_low_ratio = 0.0
        self._freq_stat_count         = 0

    def get_freq_stats(self) -> dict:
        """Return averaged frequency-bin stats accumulated since last reset_freq_stats()."""
        if self._freq_stat_count == 0:
            return {
                "selected_bin_mean":  float("nan"),
                "selected_bin_std":   float("nan"),
                "selected_low_ratio": float("nan"),
            }
        n = self._freq_stat_count
        return {
            "selected_bin_mean":  self._freq_stat_sum_mean      / n,
            "selected_bin_std":   self._freq_stat_sum_std       / n,
            "selected_low_ratio": self._freq_stat_sum_low_ratio / n,
        }

    def loss(self, true):
        # freq normalization
        B, O, N = true.shape
        residual, pred_main, _ = main_freq_part(
            true, self.freq_topk, self.rfft, self.ablation_mode
        )
        lf = nn.functional.mse_loss
        main_mse = lf(self.pred_main_freq_signal, pred_main)
        res_mse  = lf(self.pred_residual, residual)

        with torch.no_grad():
            sum_y2 = true.pow(2).sum(dim=1).clamp(min=1e-8)          # (B, N)
            self._last_fan_main_mse          = float(main_mse.item())
            self._last_fan_res_mse           = float(res_mse.item())
            self._last_fan_main_energy_ratio = float(
                (pred_main.pow(2).sum(dim=1) / sum_y2).mean().item()
            )
            self._last_fan_res_energy_ratio  = float(
                (residual.pow(2).sum(dim=1) / sum_y2).mean().item()
            )

        return main_mse + res_mse

    def get_debug_scalars(self) -> dict:
        """Return per-batch prediction quality metrics (epoch-averageable)."""
        return {
            "fan_main_mse":          self._last_fan_main_mse,
            "fan_res_mse":           self._last_fan_res_mse,
            "fan_main_energy_ratio": self._last_fan_main_energy_ratio,
            "fan_res_energy_ratio":  self._last_fan_res_energy_ratio,
        }

    def normalize(self, input):
        # (B, T, N)
        bs, len, dim = input.shape
        norm_input, x_filtered, indices = main_freq_part(
            input, self.freq_topk, self.rfft, self.ablation_mode
        )

        # Accumulate per-batch frequency-bin statistics (no gradient tracking needed)
        with torch.no_grad():
            idx_f = indices.float()
            self._freq_stat_sum_mean      += float(idx_f.mean().item())
            self._freq_stat_sum_std       += float(idx_f.std().item())
            self._freq_stat_sum_low_ratio += float(
                (indices < self.freq_topk).float().mean().item()
            )
            self._freq_stat_count += 1

        self.pred_main_freq_signal = self.model_freq(
            x_filtered.transpose(1, 2), input.transpose(1, 2)
        ).transpose(1, 2)

        return norm_input.reshape(bs, len, dim)

    def denormalize(self, input_norm):
        # input:  (B, O, N)
        bs, len, dim = input_norm.shape
        # freq denormalize
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_freq_signal

        return output.reshape(bs, len, dim)

    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x)


class MLPfreq(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(MLPfreq, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )

        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len),
        )

    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)
