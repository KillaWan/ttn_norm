from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .san_route_norm import _PureSANSeqPredictor


_LFAP_PER_FREQ_MAX_K = 8


def _valid_values(x: Optional[torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        if x is None:
            return torch.empty(0, dtype=torch.float32)
        if not torch.is_tensor(x):
            try:
                return torch.tensor([float(x)], dtype=torch.float32)
            except Exception:
                return torch.empty(0, dtype=torch.float32)
        flat = x.detach().reshape(-1)
        if flat.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=flat.device)
        flat = flat.float()
        return flat[torch.isfinite(flat)]


def _safe_scalar(x: torch.Tensor) -> float:
    with torch.no_grad():
        valid = _valid_values(x)
        if valid.numel() == 0:
            return 0.0
        return float(valid.mean().item())


def _wrap_phase(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    with torch.no_grad():
        valid = _valid_values(x)
        if valid.numel() == 0:
            device = x.device if torch.is_tensor(x) else "cpu"
            return torch.tensor(0.0, device=device)
        return torch.sqrt(valid.pow(2).mean() + float(eps))


def _mean_abs(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        valid = _valid_values(x)
        if valid.numel() == 0:
            device = x.device if torch.is_tensor(x) else "cpu"
            return torch.tensor(0.0, device=device)
        return valid.abs().mean()


def _q(x: torch.Tensor, q: float) -> torch.Tensor:
    with torch.no_grad():
        valid = _valid_values(x)
        if valid.numel() == 0:
            device = x.device if torch.is_tensor(x) else "cpu"
            return torch.tensor(0.0, device=device)
        return torch.quantile(valid.float(), q)


def _finite_ratio(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if x is None or not torch.is_tensor(x):
            return torch.tensor(0.0)
        flat = x.detach().reshape(-1)
        if flat.numel() == 0:
            return torch.tensor(0.0, device=flat.device)
        return torch.isfinite(flat).float().mean()


def _std_safe(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        valid = _valid_values(x)
        if valid.numel() == 0:
            device = x.device if torch.is_tensor(x) else "cpu"
            return torch.tensor(0.0, device=device)
        return valid.std(unbiased=False)


def _max_abs_safe(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        valid = _valid_values(x)
        if valid.numel() == 0:
            device = x.device if torch.is_tensor(x) else "cpu"
            return torch.tensor(0.0, device=device)
        return valid.abs().max()


def _max_safe(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        valid = _valid_values(x)
        if valid.numel() == 0:
            device = x.device if torch.is_tensor(x) else "cpu"
            return torch.tensor(0.0, device=device)
        return valid.max()


def _bool_ratio(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if x is None or not torch.is_tensor(x):
            return torch.tensor(0.0)
        flat = x.detach().reshape(-1)
        if flat.numel() == 0:
            return torch.tensor(0.0, device=flat.device)
        return flat.float().mean()


def _lfap_diag_keys(max_k: int = _LFAP_PER_FREQ_MAX_K) -> list[str]:
    keys = [
        "lfap_carrier_logA_mean",
        "lfap_carrier_logA_std",
        "lfap_carrier_logA_p05",
        "lfap_carrier_logA_p50",
        "lfap_carrier_logA_p95",
        "lfap_pred_logA_mean",
        "lfap_pred_logA_std",
        "lfap_pred_logA_p05",
        "lfap_pred_logA_p50",
        "lfap_pred_logA_p95",
        "lfap_oracle_logA_mean",
        "lfap_oracle_logA_std",
        "lfap_oracle_logA_p05",
        "lfap_oracle_logA_p50",
        "lfap_oracle_logA_p95",
        "lfap_pred_minus_carrier_logA_mean",
        "lfap_pred_minus_carrier_logA_std",
        "lfap_pred_minus_carrier_logA_rms",
        "lfap_pred_minus_carrier_logA_p05",
        "lfap_pred_minus_carrier_logA_p50",
        "lfap_pred_minus_carrier_logA_p95",
        "lfap_amp_ratio_mean",
        "lfap_amp_ratio_p50",
        "lfap_amp_ratio_p95",
        "lfap_amp_ratio_p99",
        "lfap_amp_ratio_max",
        "lfap_pred_minus_oracle_logA_mean",
        "lfap_pred_minus_oracle_logA_std",
        "lfap_pred_minus_oracle_logA_rms",
        "lfap_pred_minus_oracle_logA_p05",
        "lfap_pred_minus_oracle_logA_p50",
        "lfap_pred_minus_oracle_logA_p95",
        "lfap_pred_oracle_amp_ratio_mean",
        "lfap_pred_oracle_amp_ratio_p50",
        "lfap_pred_oracle_amp_ratio_p95",
        "lfap_pred_oracle_amp_ratio_p99",
        "lfap_pred_oracle_amp_ratio_max",
        "lfap_carrier_minus_oracle_logA_rms",
        "lfap_carrier_oracle_amp_ratio_mean",
        "lfap_carrier_oracle_amp_ratio_p50",
        "lfap_carrier_oracle_amp_ratio_p95",
        "lfap_carrier_oracle_amp_ratio_p99",
        "lfap_carrier_oracle_amp_ratio_max",
        "lfap_oracle_carrier_amp_ratio_p50",
        "lfap_oracle_carrier_amp_ratio_p95",
        "lfap_oracle_carrier_amp_ratio_p99",
        "lfap_oracle_carrier_amp_ratio_max",
        "lfap_delta_phase_mean",
        "lfap_delta_phase_abs_mean",
        "lfap_delta_phase_rms",
        "lfap_delta_phase_p50_abs",
        "lfap_delta_phase_p95_abs",
        "lfap_delta_phase_p99_abs",
        "lfap_delta_phase_max_abs",
        "lfap_target_delta_phase_mean",
        "lfap_target_delta_phase_abs_mean",
        "lfap_target_delta_phase_rms",
        "lfap_target_delta_phase_p50_abs",
        "lfap_target_delta_phase_p95_abs",
        "lfap_target_delta_phase_p99_abs",
        "lfap_target_delta_phase_max_abs",
        "lfap_phase_err_abs_mean",
        "lfap_phase_err_rms",
        "lfap_phase_err_p50_abs",
        "lfap_phase_err_p95_abs",
        "lfap_phase_err_p99_abs",
        "lfap_phase_cos",
        "lfap_phase_sin_err_rms",
        "lfap_phase_cos_err_rms",
        "lfap_phase_active_ratio_0p1",
        "lfap_phase_active_ratio_0p5",
        "lfap_phase_active_ratio_1p0",
        "lfap_y_before_rms",
        "lfap_y_dc_only_rms",
        "lfap_y_amp_only_rms",
        "lfap_y_phase_only_rms",
        "lfap_y_amp_phase_rms",
        "lfap_y_final_rms",
        "lfap_dc_effect_rms",
        "lfap_amp_only_effect_rms",
        "lfap_phase_only_effect_rms",
        "lfap_amp_phase_effect_rms",
        "lfap_final_effect_rms",
        "lfap_dc_effect_ratio",
        "lfap_amp_only_effect_ratio",
        "lfap_phase_only_effect_ratio",
        "lfap_amp_phase_effect_ratio",
        "lfap_final_effect_ratio",
        "lfap_y_before_mse",
        "lfap_y_dc_only_mse",
        "lfap_y_amp_only_mse",
        "lfap_y_phase_only_mse",
        "lfap_y_amp_phase_mse",
        "lfap_y_final_mse",
        "lfap_y_before_mae",
        "lfap_y_dc_only_mae",
        "lfap_y_amp_only_mae",
        "lfap_y_phase_only_mae",
        "lfap_y_amp_phase_mae",
        "lfap_y_final_mae",
        "lfap_amp_only_mse_delta",
        "lfap_phase_only_mse_delta",
        "lfap_amp_phase_mse_delta",
        "lfap_final_mse_delta",
        "lfap_nan_ratio_pred_logA",
        "lfap_inf_ratio_pred_logA",
        "lfap_nan_ratio_delta_phase",
        "lfap_inf_ratio_delta_phase",
        "lfap_nan_ratio_y_final",
        "lfap_inf_ratio_y_final",
        "lfap_pred_logA_abs_max",
        "lfap_delta_phase_abs_max",
        "lfap_y_final_abs_max",
        "lfap_grad_sensitive_amp_ratio_gt_10",
        "lfap_grad_sensitive_amp_ratio_gt_100",
    ]
    for k in range(1, max_k + 1):
        keys.extend(
            [
                f"lfap_f{k}_pred_minus_carrier_logA_rms",
                f"lfap_f{k}_pred_minus_oracle_logA_rms",
                f"lfap_f{k}_carrier_minus_oracle_logA_rms",
                f"lfap_f{k}_amp_ratio_p95",
                f"lfap_f{k}_delta_phase_rms",
                f"lfap_f{k}_target_delta_phase_rms",
                f"lfap_f{k}_phase_err_rms",
                f"lfap_f{k}_phase_cos",
                f"lfap_f{k}_carrier_complex_mse",
                f"lfap_f{k}_pred_complex_mse",
                f"lfap_f{k}_complex_gain",
                f"lfap_f{k}_rel_complex_gain",
                f"lfap_f{k}_pred_better_ratio",
                f"lfap_f{k}_oracle_energy_share",
                f"lfap_f{k}_oracle_carrier_amp_ratio_p95",
            ]
        )
    for rank in range(1, 5):
        keys.extend(
            [
                f"lfap_tail{rank}_bin_idx",
                f"lfap_tail{rank}_energy_share",
                f"lfap_tail{rank}_carrier_complex_mse",
                f"lfap_tail{rank}_pred_complex_mse",
                f"lfap_tail{rank}_complex_gain",
                f"lfap_tail{rank}_rel_complex_gain",
                f"lfap_tail{rank}_pred_better_ratio",
                f"lfap_tail{rank}_phase_err_rms",
                f"lfap_tail{rank}_phase_cos",
            ]
        )
    return keys


def _auto_lfap_tcn_channels(enc_in: int, dataset: Optional[str]) -> list[int]:
    key = (dataset or "").lower()
    if "traffic" in key:
        return [512, 1024, 1024, 512, enc_in]
    if "electricity" in key or "ecl" in key:
        return [256, 512, 1024, 512, enc_in]
    if "weather" in key:
        return [32, 64, 32, enc_in]
    return [16, 32, 64, 32, enc_in]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _LFPhaseTCNBranch(nn.Module):
    def __init__(
        self,
        enc_in: int,
        hist_stat_len: int,
        pred_stat_len: int,
        d_model: int,
        phase_max_delta: float,
        dataset: Optional[str] = None,
        tcn_channels: Optional[list[int]] = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.enc_in = int(enc_in)
        self.hist_stat_len = int(hist_stat_len)
        self.pred_stat_len = int(pred_stat_len)
        self.phase_max_delta = float(phase_max_delta)
        channels = (
            list(tcn_channels)
            if tcn_channels is not None
            else _auto_lfap_tcn_channels(self.enc_in, dataset)
        )
        if not channels:
            channels = [self.enc_in]
        if channels[-1] != self.enc_in:
            channels = channels[:-1] + [self.enc_in]

        self.in_proj = nn.Linear(2 * self.hist_stat_len, int(d_model))
        self.tcn = TemporalConvNet(
            num_inputs=self.enc_in,
            num_channels=channels,
            kernel_size=3,
            dropout=float(dropout),
        )
        self.out_proj = nn.Linear(int(d_model), self.pred_stat_len)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, K, C, 2 * P_hist)
        B, K, C, _ = feat.shape
        x = feat.reshape(B * K, C, -1)
        x = self.in_proj(x)
        x = self.tcn(x)
        raw = self.out_proj(x)
        raw = raw.reshape(B, K, C, self.pred_stat_len).permute(0, 3, 1, 2)
        return self.phase_max_delta * torch.tanh(raw)


class SANLowFreqAPNorm(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        enc_in: int,
        dataset: Optional[str] = None,
        lfap_k: int = 2,
        lfap_d_model: int = 128,
        lfap_amp_loss_weight: float = 0.1,
        lfap_phase_loss_weight: float = 0.1,
        lfap_phase_max_delta: float = 3.141592653589793,
        lfap_detach_final_phase: bool = True,
        lfap_detach_final_amp: bool = False,
        eps: float = 1e-6,
        san_w_mu: float = 1.0,
        lfap_tcn_channels: Optional[list[int]] = None,
        lfap_dropout: float = 0.05,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.window_len = int(period_len)
        self.enc_in = int(enc_in)
        self.channels = self.enc_in
        self.dataset = dataset
        self.eps = float(eps)
        self.san_w_mu = float(san_w_mu)
        self.lfap_amp_loss_weight = float(lfap_amp_loss_weight)
        self.lfap_phase_loss_weight = float(lfap_phase_loss_weight)
        self.lfap_detach_final_phase = bool(lfap_detach_final_phase)
        self.lfap_detach_final_amp = bool(lfap_detach_final_amp)
        self.lfap_k_requested = max(0, int(lfap_k))
        self.lfap_k = min(self.lfap_k_requested, self.window_len // 2)
        self.lfap_phase_max_delta = float(lfap_phase_max_delta)
        self.route_path = "none"
        self.route_state = "none"
        self.route_state_loss_scale = 0.0
        self.san_ablation_mode = "none"

        if self.window_len <= 0:
            raise ValueError("period_len must be positive.")
        if self.seq_len % self.window_len != 0:
            raise ValueError(
                "SANLowFreqAPNorm requires seq_len divisible by period_len."
            )
        if self.pred_len % self.window_len != 0:
            raise ValueError(
                "SANLowFreqAPNorm requires pred_len divisible by period_len."
            )

        self.hist_stat_len = self.seq_len // self.window_len
        self.pred_stat_len = self.pred_len // self.window_len

        self.dc_predictor = _PureSANSeqPredictor(
            hist_stat_len=self.hist_stat_len,
            pred_stat_len=self.pred_stat_len,
            window_len=self.window_len,
            enc_in=self.enc_in,
            sigma_min=eps,
            hidden_dim=128,
            use_phase=False,
            phase_k=max(1, self.lfap_k),
            phase_zero_init=True,
            state_residual_scale_init=0.1,
        )
        self.amp_predictor = (
            _PureSANSeqPredictor(
                hist_stat_len=self.hist_stat_len,
                pred_stat_len=self.pred_stat_len,
                window_len=self.window_len,
                enc_in=self.enc_in * max(1, self.lfap_k),
                sigma_min=eps,
                hidden_dim=128,
                use_phase=False,
                phase_k=max(1, self.lfap_k),
                phase_zero_init=True,
                state_residual_scale_init=0.1,
            )
            if self.lfap_k > 0
            else None
        )
        self.phase_tcn = (
            _LFPhaseTCNBranch(
                enc_in=self.enc_in,
                hist_stat_len=self.hist_stat_len,
                pred_stat_len=self.pred_stat_len,
                d_model=int(lfap_d_model),
                phase_max_delta=self.lfap_phase_max_delta,
                dataset=dataset,
                tcn_channels=lfap_tcn_channels,
                dropout=lfap_dropout,
            )
            if self.lfap_k > 0
            else None
        )

        self._reset_cache()

    def _diag_zeros(self) -> dict[str, float]:
        return {key: 0.0 for key in _lfap_diag_keys()}

    def _reset_cache(self) -> None:
        self._mu_hist: Optional[torch.Tensor] = None
        self._std_hist: Optional[torch.Tensor] = None
        self._base_mu_fut_hat: Optional[torch.Tensor] = None
        self._base_std_fut_hat: Optional[torch.Tensor] = None
        self._lfap_hist_logA: Optional[torch.Tensor] = None
        self._lfap_hist_phase: Optional[torch.Tensor] = None
        self._lfap_hist_amp_mean: Optional[torch.Tensor] = None
        self._lfap_hist_amp_rms: Optional[torch.Tensor] = None
        self._lfap_loga_fut: Optional[torch.Tensor] = None
        self._lfap_delta_loga_fut: Optional[torch.Tensor] = None
        self._lfap_delta_phase: Optional[torch.Tensor] = None
        self._lfap_carrier_fft: Optional[torch.Tensor] = None
        self._lfap_carrier_fft_low: Optional[torch.Tensor] = None
        self._lfap_carrier_logA: Optional[torch.Tensor] = None
        self._lfap_carrier_phase: Optional[torch.Tensor] = None
        self._lfap_pred_logA: Optional[torch.Tensor] = None
        self._lfap_pred_delta_logA: Optional[torch.Tensor] = None
        self._lfap_pred_delta_phase: Optional[torch.Tensor] = None
        self._lfap_y_dc_only: Optional[torch.Tensor] = None
        self._lfap_y_amp_only: Optional[torch.Tensor] = None
        self._lfap_y_phase_only: Optional[torch.Tensor] = None
        self._lfap_y_amp_phase: Optional[torch.Tensor] = None
        self._lfap_y_before: Optional[torch.Tensor] = None
        self._lfap_y_after: Optional[torch.Tensor] = None
        self._last_lfap_diag_stats: dict[str, float] = self._diag_zeros()
        self._last_mu_loss: float = 0.0
        self._last_std_loss: float = 0.0
        self._last_phase_loss: float = 0.0
        self._last_route_state_loss: float = 0.0
        self._last_base_aux_loss: float = 0.0
        self._last_aux_total: float = 0.0
        self._last_lfap_mu_loss: float = 0.0
        self._last_lfap_amp_loss: float = 0.0
        self._last_lfap_phase_loss: float = 0.0
        self._last_lfap_state_loss: float = 0.0
        self._last_lfap_delta_loga_rms: float = 0.0
        self._last_lfap_delta_phase_rms: float = 0.0
        self._last_lfap_effect_rms: float = 0.0
        self._last_lfap_effect_ratio: float = 0.0
        self._last_lfap_pred_residual_rms_before: float = 0.0
        self._last_lfap_pred_residual_rms_after: float = 0.0
        self._last_lfap_oracle_residual_rms: float = 0.0
        self._last_lfap_residual_rms_ratio: float = 0.0
        self._last_lfap_task_mse_from_cache: float = 0.0
        self._last_lfap_recon_gain_vs_dc: float = 0.0
        self._last_lfap_recon_gain_vs_amp: float = 0.0

    def parameters_base_predictor(self) -> list:
        params = list(self.dc_predictor.parameters())
        if self.amp_predictor is not None:
            params.extend(list(self.amp_predictor.parameters()))
        return params

    def parameters_lfap(self) -> list:
        if self.phase_tcn is None:
            return []
        return list(self.phase_tcn.parameters())

    def parameters_route_modules(self) -> list:
        return self.parameters_lfap()

    def freeze_base_predictor(self) -> None:
        for p in self.dc_predictor.parameters():
            p.requires_grad_(False)
        if self.amp_predictor is not None:
            for p in self.amp_predictor.parameters():
                p.requires_grad_(False)

    def unfreeze_base_predictor(self) -> None:
        for p in self.dc_predictor.parameters():
            p.requires_grad_(True)
        if self.amp_predictor is not None:
            for p in self.amp_predictor.parameters():
                p.requires_grad_(True)

    def freeze_route_modules(self) -> None:
        if self.phase_tcn is not None:
            for p in self.phase_tcn.parameters():
                p.requires_grad_(False)

    def unfreeze_route_modules(self) -> None:
        if self.phase_tcn is not None:
            for p in self.phase_tcn.parameters():
                p.requires_grad_(True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._reset_cache()
        B, T, C = x.shape
        if T != self.seq_len:
            raise ValueError(
                f"SANLowFreqAPNorm expected seq_len={self.seq_len}, got {T}."
            )
        P_hist = self.hist_stat_len
        W = self.window_len
        x_patches = x.reshape(B, P_hist, W, C)
        mu_hist = x_patches.mean(dim=2)
        std_hist = x_patches.std(dim=2).clamp_min(self.eps)
        z_patches = x_patches - mu_hist.unsqueeze(2)
        z = z_patches.reshape(B, T, C)

        self._mu_hist = mu_hist
        self._std_hist = std_hist
        anchor = mu_hist.mean(dim=1, keepdim=True)
        mu_fut_hat, _ = self.dc_predictor.predict(
            mu_hist=mu_hist,
            std_hist=std_hist,
            anchor=anchor,
            raw_patches=x_patches,
            eps=self.eps,
        )
        self._base_mu_fut_hat = mu_fut_hat
        self._base_std_fut_hat = std_hist.mean(dim=1, keepdim=True).expand(
            -1, self.pred_stat_len, -1
        ).contiguous()

        if self.lfap_k > 0:
            r_hist = torch.fft.rfft(z_patches, dim=2)
            low = r_hist[:, :, 1 : self.lfap_k + 1, :]
            loga_hist = torch.log(low.abs() + self.eps)
            self._lfap_hist_logA = loga_hist.detach()
            self._lfap_hist_phase = torch.angle(low).detach()
            self._lfap_hist_amp_mean = low.abs().detach().mean()
            self._lfap_hist_amp_rms = _rms(low.abs().detach(), self.eps)
            loga_hist_flat = loga_hist.permute(0, 1, 3, 2).reshape(
                B, P_hist, C * self.lfap_k
            )
            amp_anchor = torch.zeros(
                B,
                1,
                C * self.lfap_k,
                device=x.device,
                dtype=x.dtype,
            )
            amp_raw_patches = loga_hist_flat.unsqueeze(2).expand(
                B, P_hist, self.window_len, C * self.lfap_k
            )
            if self.amp_predictor is None:
                raise RuntimeError("amp_predictor must exist when lfap_k > 0.")
            amp_fut_flat, _ = self.amp_predictor.predict(
                mu_hist=loga_hist_flat,
                std_hist=torch.ones_like(loga_hist_flat),
                anchor=amp_anchor,
                raw_patches=amp_raw_patches,
                eps=self.eps,
            )
            self._lfap_delta_loga_fut = amp_fut_flat.reshape(
                B, self.pred_stat_len, C, self.lfap_k
            ).permute(0, 1, 3, 2).contiguous()
            self._lfap_loga_fut = None

            phase_hist = torch.angle(low)
            feat = torch.stack(
                [torch.cos(phase_hist), torch.sin(phase_hist)],
                dim=-1,
            )
            feat = feat.permute(0, 2, 3, 1, 4).reshape(
                B, self.lfap_k, C, 2 * self.hist_stat_len
            )
            if self.phase_tcn is None:
                raise RuntimeError("phase_tcn must exist when lfap_k > 0.")
            delta_phase = self.phase_tcn(feat)
            self._lfap_delta_phase = delta_phase
            self._last_lfap_delta_loga_rms = float(
                _rms(self._lfap_delta_loga_fut.detach(), self.eps).item()
            )
            self._last_lfap_delta_phase_rms = float(
                _rms(delta_phase.detach(), self.eps).item()
            )
        return z

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        if self._base_mu_fut_hat is None:
            return y_norm
        B, T, C = y_norm.shape
        if T != self.pred_len:
            raise ValueError(
                f"SANLowFreqAPNorm expected pred_len={self.pred_len}, got {T}."
            )
        P_fut = self.pred_stat_len
        W = self.window_len
        y_norm_patches = y_norm.reshape(B, P_fut, W, C)
        y_res = y_norm_patches - y_norm_patches.mean(dim=2, keepdim=True)
        Y = torch.fft.rfft(y_res, dim=2)
        self._lfap_carrier_fft = Y.detach().clone()

        Y_dc = Y.clone()
        Y_dc[:, :, 0, :] = W * self._base_mu_fut_hat
        y_before = torch.fft.irfft(Y_dc, n=W, dim=2)
        y_before_flat = y_before.reshape(B, self.pred_len, C).contiguous()
        self._lfap_y_before = y_before_flat.detach()
        self._lfap_y_dc_only = y_before_flat.detach()
        before_residual = y_before - self._base_mu_fut_hat.unsqueeze(2)
        before_rms = _rms(before_residual.detach(), self.eps)
        self._last_lfap_pred_residual_rms_before = float(before_rms.item())

        Y_final = Y_dc.clone()
        if (
            self.lfap_k > 0
            and self._lfap_delta_loga_fut is not None
            and self._lfap_delta_phase is not None
        ):
            carrier_low = Y[:, :, 1 : self.lfap_k + 1, :]
            carrier_logA = torch.log(carrier_low.abs() + self.eps)
            carrier_phase = torch.angle(carrier_low)
            delta_logA_final = (
                self._lfap_delta_loga_fut.detach()
                if self.lfap_detach_final_amp
                else self._lfap_delta_loga_fut
            )
            logA_final = carrier_logA + delta_logA_final
            delta_for_final = (
                self._lfap_delta_phase.detach()
                if self.lfap_detach_final_phase
                else self._lfap_delta_phase
            )
            phi_final = carrier_phase + delta_for_final
            Y_final[:, :, 1 : self.lfap_k + 1, :] = torch.polar(
                torch.exp(logA_final),
                phi_final,
            )
            with torch.no_grad():
                self._lfap_carrier_fft_low = carrier_low.detach()
                self._lfap_carrier_logA = carrier_logA.detach()
                self._lfap_carrier_phase = carrier_phase.detach()
                self._lfap_pred_logA = (
                    carrier_logA + self._lfap_delta_loga_fut
                ).detach()
                self._lfap_pred_delta_logA = self._lfap_delta_loga_fut.detach()
                self._lfap_pred_delta_phase = self._lfap_delta_phase.detach()
                Y_amp = Y_dc.detach().clone()
                Y_amp[:, :, 1 : self.lfap_k + 1, :] = torch.polar(
                    torch.exp(self._lfap_pred_logA),
                    self._lfap_carrier_phase,
                )
                Y_phase = Y_dc.detach().clone()
                Y_phase[:, :, 1 : self.lfap_k + 1, :] = torch.polar(
                    torch.exp(self._lfap_carrier_logA),
                    self._lfap_carrier_phase + self._lfap_pred_delta_phase,
                )
                Y_amp_phase = Y_dc.detach().clone()
                Y_amp_phase[:, :, 1 : self.lfap_k + 1, :] = torch.polar(
                    torch.exp(self._lfap_pred_logA),
                    self._lfap_carrier_phase + self._lfap_pred_delta_phase,
                )
                self._lfap_y_amp_only = torch.fft.irfft(
                    Y_amp, n=W, dim=2
                ).reshape(B, self.pred_len, C).contiguous()
                self._lfap_y_phase_only = torch.fft.irfft(
                    Y_phase, n=W, dim=2
                ).reshape(B, self.pred_len, C).contiguous()
                self._lfap_y_amp_phase = torch.fft.irfft(
                    Y_amp_phase, n=W, dim=2
                ).reshape(B, self.pred_len, C).contiguous()
        else:
            self._lfap_carrier_fft_low = None
            self._lfap_carrier_logA = None
            self._lfap_carrier_phase = None
            self._lfap_pred_logA = None
            self._lfap_pred_delta_logA = None
            self._lfap_pred_delta_phase = None
            self._lfap_y_amp_only = None
            self._lfap_y_phase_only = None
            self._lfap_y_amp_phase = None

        y_after = torch.fft.irfft(Y_final, n=W, dim=2)
        y_after_flat = y_after.reshape(B, self.pred_len, C).contiguous()
        self._lfap_y_after = y_after_flat.detach()
        if self._lfap_y_amp_phase is None:
            self._lfap_y_amp_phase = y_after_flat.detach()
        after_residual = y_after - self._base_mu_fut_hat.unsqueeze(2)
        after_rms = _rms(after_residual.detach(), self.eps)
        self._last_lfap_pred_residual_rms_after = float(after_rms.item())

        effect = y_after - y_before
        effect_rms = _rms(effect.detach(), self.eps)
        if self.lfap_k > 0:
            self._last_lfap_effect_rms = float(effect_rms.item())
            self._last_lfap_effect_ratio = float(
                (effect_rms / (before_rms + self.eps)).item()
            )
        else:
            self._last_lfap_effect_rms = 0.0
            self._last_lfap_effect_ratio = 0.0

        return y_after_flat

    def compute_base_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        if self._base_mu_fut_hat is None:
            return torch.tensor(0.0, device=y_true.device)

        B, _, C = y_true.shape
        P_fut = self.pred_stat_len
        W = self.window_len
        y_true_patches = y_true.reshape(B, P_fut, W, C)
        mu_true = y_true_patches.mean(dim=2)
        true_res = y_true_patches - mu_true.unsqueeze(2)
        Y_true = torch.fft.rfft(true_res, dim=2)
        Y_true_low = Y_true[:, :, 1 : self.lfap_k + 1, :]

        mu_loss = F.mse_loss(self._base_mu_fut_hat, mu_true)
        amp_loss = torch.tensor(0.0, device=y_true.device)
        phase_loss = torch.tensor(0.0, device=y_true.device)
        if (
            self.lfap_k > 0
            and self._lfap_delta_loga_fut is not None
            and self._lfap_carrier_logA is not None
        ):
            target_logA = torch.log(
                Y_true_low.abs() + self.eps
            )
            target_delta_logA = target_logA - self._lfap_carrier_logA
            amp_loss = F.mse_loss(
                self._lfap_delta_loga_fut,
                target_delta_logA.detach(),
            )

        lfap_trainable = (
            self.phase_tcn is not None
            and any(p.requires_grad for p in self.phase_tcn.parameters())
        )
        if (
            self.lfap_k > 0
            and lfap_trainable
            and self._lfap_carrier_fft is not None
            and self._lfap_delta_phase is not None
        ):
            target_delta_phase = _wrap_phase(
                torch.angle(Y_true_low)
                - torch.angle(self._lfap_carrier_fft[:, :, 1 : self.lfap_k + 1, :])
            )
            phase_loss = (
                F.mse_loss(
                    torch.cos(self._lfap_delta_phase),
                    torch.cos(target_delta_phase.detach()),
                )
                + F.mse_loss(
                    torch.sin(self._lfap_delta_phase),
                    torch.sin(target_delta_phase.detach()),
                )
            )

        state_loss = (
            self.san_w_mu * mu_loss
            + self.lfap_amp_loss_weight * amp_loss
            + self.lfap_phase_loss_weight * phase_loss
        )

        oracle_rms = _rms(true_res.detach(), self.eps)
        self._last_lfap_oracle_residual_rms = float(oracle_rms.item())
        if self._last_lfap_pred_residual_rms_after > 0.0:
            self._last_lfap_residual_rms_ratio = (
                self._last_lfap_pred_residual_rms_after
                / (self._last_lfap_oracle_residual_rms + self.eps)
            )
        else:
            self._last_lfap_residual_rms_ratio = 0.0

        self._last_mu_loss = float(mu_loss.detach().item())
        self._last_std_loss = 0.0
        self._last_phase_loss = 0.0
        self._last_route_state_loss = float(state_loss.detach().item())
        self._last_base_aux_loss = float(state_loss.detach().item())
        self._last_aux_total = float(state_loss.detach().item())
        self._last_lfap_mu_loss = self._last_mu_loss
        self._last_lfap_amp_loss = float(amp_loss.detach().item())
        self._last_lfap_phase_loss = float(phase_loss.detach().item())
        self._last_lfap_state_loss = float(state_loss.detach().item())
        self._last_lfap_task_mse_from_cache = 0.0
        self._last_lfap_recon_gain_vs_dc = 0.0
        self._last_lfap_recon_gain_vs_amp = 0.0

        diag = self._diag_zeros()
        try:
            with torch.no_grad():
                if (
                    self.lfap_k > 0
                    and self._lfap_carrier_logA is not None
                    and self._lfap_carrier_phase is not None
                    and self._lfap_pred_logA is not None
                    and self._lfap_pred_delta_phase is not None
                ):
                    oracle_logA = torch.log(Y_true_low.abs() + self.eps).detach()
                    oracle_phase = torch.angle(Y_true_low).detach()
                    carrier_logA = self._lfap_carrier_logA.detach()
                    carrier_phase = self._lfap_carrier_phase.detach()
                    pred_logA = self._lfap_pred_logA.detach()
                    pred_delta_phase = self._lfap_pred_delta_phase.detach()
                    target_delta_phase = _wrap_phase(oracle_phase - carrier_phase)
                    phase_err = _wrap_phase(pred_delta_phase - target_delta_phase)
                    pred_minus_carrier = pred_logA - carrier_logA
                    pred_minus_oracle = pred_logA - oracle_logA
                    carrier_minus_oracle = carrier_logA - oracle_logA
                    amp_ratio = torch.exp(pred_minus_carrier)
                    pred_oracle_amp_ratio = torch.exp(pred_minus_oracle)
                    carrier_oracle_amp_ratio = torch.exp(carrier_minus_oracle)
                    oracle_carrier_amp_ratio = torch.exp(oracle_logA - carrier_logA)

                    def _fill_dist(prefix: str, x: torch.Tensor) -> None:
                        diag[f"{prefix}_mean"] = _safe_scalar(x)
                        diag[f"{prefix}_std"] = _safe_scalar(_std_safe(x))
                        diag[f"{prefix}_p05"] = _safe_scalar(_q(x, 0.05))
                        diag[f"{prefix}_p50"] = _safe_scalar(_q(x, 0.50))
                        diag[f"{prefix}_p95"] = _safe_scalar(_q(x, 0.95))

                    _fill_dist("lfap_carrier_logA", carrier_logA)
                    _fill_dist("lfap_pred_logA", pred_logA)
                    _fill_dist("lfap_oracle_logA", oracle_logA)
                    _fill_dist("lfap_pred_minus_carrier_logA", pred_minus_carrier)
                    _fill_dist("lfap_pred_minus_oracle_logA", pred_minus_oracle)

                    diag["lfap_pred_minus_carrier_logA_rms"] = _safe_scalar(
                        _rms(pred_minus_carrier, self.eps)
                    )
                    diag["lfap_pred_minus_oracle_logA_rms"] = _safe_scalar(
                        _rms(pred_minus_oracle, self.eps)
                    )
                    diag["lfap_carrier_minus_oracle_logA_rms"] = _safe_scalar(
                        _rms(carrier_minus_oracle, self.eps)
                    )
                    diag["lfap_amp_ratio_mean"] = _safe_scalar(amp_ratio)
                    diag["lfap_amp_ratio_p50"] = _safe_scalar(_q(amp_ratio, 0.50))
                    diag["lfap_amp_ratio_p95"] = _safe_scalar(_q(amp_ratio, 0.95))
                    diag["lfap_amp_ratio_p99"] = _safe_scalar(_q(amp_ratio, 0.99))
                    diag["lfap_amp_ratio_max"] = _safe_scalar(_max_safe(amp_ratio))
                    diag["lfap_pred_oracle_amp_ratio_mean"] = _safe_scalar(
                        pred_oracle_amp_ratio
                    )
                    diag["lfap_pred_oracle_amp_ratio_p50"] = _safe_scalar(
                        _q(pred_oracle_amp_ratio, 0.50)
                    )
                    diag["lfap_pred_oracle_amp_ratio_p95"] = _safe_scalar(
                        _q(pred_oracle_amp_ratio, 0.95)
                    )
                    diag["lfap_pred_oracle_amp_ratio_p99"] = _safe_scalar(
                        _q(pred_oracle_amp_ratio, 0.99)
                    )
                    diag["lfap_pred_oracle_amp_ratio_max"] = _safe_scalar(
                        _max_safe(pred_oracle_amp_ratio)
                    )
                    diag["lfap_carrier_oracle_amp_ratio_mean"] = _safe_scalar(
                        carrier_oracle_amp_ratio
                    )
                    diag["lfap_carrier_oracle_amp_ratio_p50"] = _safe_scalar(
                        _q(carrier_oracle_amp_ratio, 0.50)
                    )
                    diag["lfap_carrier_oracle_amp_ratio_p95"] = _safe_scalar(
                        _q(carrier_oracle_amp_ratio, 0.95)
                    )
                    diag["lfap_carrier_oracle_amp_ratio_p99"] = _safe_scalar(
                        _q(carrier_oracle_amp_ratio, 0.99)
                    )
                    diag["lfap_carrier_oracle_amp_ratio_max"] = _safe_scalar(
                        _max_safe(carrier_oracle_amp_ratio)
                    )
                    diag["lfap_oracle_carrier_amp_ratio_p50"] = _safe_scalar(
                        _q(oracle_carrier_amp_ratio, 0.50)
                    )
                    diag["lfap_oracle_carrier_amp_ratio_p95"] = _safe_scalar(
                        _q(oracle_carrier_amp_ratio, 0.95)
                    )
                    diag["lfap_oracle_carrier_amp_ratio_p99"] = _safe_scalar(
                        _q(oracle_carrier_amp_ratio, 0.99)
                    )
                    diag["lfap_oracle_carrier_amp_ratio_max"] = _safe_scalar(
                        _max_safe(oracle_carrier_amp_ratio)
                    )

                    abs_delta = pred_delta_phase.abs()
                    abs_target = target_delta_phase.abs()
                    abs_err = phase_err.abs()
                    diag["lfap_delta_phase_mean"] = _safe_scalar(pred_delta_phase)
                    diag["lfap_delta_phase_abs_mean"] = _safe_scalar(_mean_abs(pred_delta_phase))
                    diag["lfap_delta_phase_rms"] = _safe_scalar(_rms(pred_delta_phase, self.eps))
                    diag["lfap_delta_phase_p50_abs"] = _safe_scalar(_q(abs_delta, 0.50))
                    diag["lfap_delta_phase_p95_abs"] = _safe_scalar(_q(abs_delta, 0.95))
                    diag["lfap_delta_phase_p99_abs"] = _safe_scalar(_q(abs_delta, 0.99))
                    diag["lfap_delta_phase_max_abs"] = _safe_scalar(_max_abs_safe(pred_delta_phase))
                    diag["lfap_target_delta_phase_mean"] = _safe_scalar(target_delta_phase)
                    diag["lfap_target_delta_phase_abs_mean"] = _safe_scalar(_mean_abs(target_delta_phase))
                    diag["lfap_target_delta_phase_rms"] = _safe_scalar(_rms(target_delta_phase, self.eps))
                    diag["lfap_target_delta_phase_p50_abs"] = _safe_scalar(_q(abs_target, 0.50))
                    diag["lfap_target_delta_phase_p95_abs"] = _safe_scalar(_q(abs_target, 0.95))
                    diag["lfap_target_delta_phase_p99_abs"] = _safe_scalar(_q(abs_target, 0.99))
                    diag["lfap_target_delta_phase_max_abs"] = _safe_scalar(_max_abs_safe(target_delta_phase))
                    diag["lfap_phase_err_abs_mean"] = _safe_scalar(_mean_abs(phase_err))
                    diag["lfap_phase_err_rms"] = _safe_scalar(_rms(phase_err, self.eps))
                    diag["lfap_phase_err_p50_abs"] = _safe_scalar(_q(abs_err, 0.50))
                    diag["lfap_phase_err_p95_abs"] = _safe_scalar(_q(abs_err, 0.95))
                    diag["lfap_phase_err_p99_abs"] = _safe_scalar(_q(abs_err, 0.99))
                    diag["lfap_phase_cos"] = _safe_scalar(torch.cos(phase_err))
                    diag["lfap_phase_sin_err_rms"] = _safe_scalar(
                        _rms(torch.sin(pred_delta_phase) - torch.sin(target_delta_phase), self.eps)
                    )
                    diag["lfap_phase_cos_err_rms"] = _safe_scalar(
                        _rms(torch.cos(pred_delta_phase) - torch.cos(target_delta_phase), self.eps)
                    )
                    diag["lfap_phase_active_ratio_0p1"] = _safe_scalar(_bool_ratio(abs_delta > 0.1))
                    diag["lfap_phase_active_ratio_0p5"] = _safe_scalar(_bool_ratio(abs_delta > 0.5))
                    diag["lfap_phase_active_ratio_1p0"] = _safe_scalar(_bool_ratio(abs_delta > 1.0))

                    y_true_flat = y_true.detach()
                    y_before = self._lfap_y_before if self._lfap_y_before is not None else torch.zeros_like(y_true_flat)
                    y_dc = self._lfap_y_dc_only if self._lfap_y_dc_only is not None else y_before
                    y_amp = self._lfap_y_amp_only if self._lfap_y_amp_only is not None else y_dc
                    y_phase = self._lfap_y_phase_only if self._lfap_y_phase_only is not None else y_dc
                    y_amp_phase = self._lfap_y_amp_phase if self._lfap_y_amp_phase is not None else y_dc
                    y_final = self._lfap_y_after if self._lfap_y_after is not None else y_amp_phase

                    def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                        return ((a - b).detach().float().pow(2)).mean()

                    def _mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                        return (a - b).detach().float().abs().mean()

                    diag["lfap_y_before_rms"] = _safe_scalar(_rms(y_before, self.eps))
                    diag["lfap_y_dc_only_rms"] = _safe_scalar(_rms(y_dc, self.eps))
                    diag["lfap_y_amp_only_rms"] = _safe_scalar(_rms(y_amp, self.eps))
                    diag["lfap_y_phase_only_rms"] = _safe_scalar(_rms(y_phase, self.eps))
                    diag["lfap_y_amp_phase_rms"] = _safe_scalar(_rms(y_amp_phase, self.eps))
                    diag["lfap_y_final_rms"] = _safe_scalar(_rms(y_final, self.eps))

                    denom = float(self._last_lfap_pred_residual_rms_before) + self.eps
                    diag["lfap_dc_effect_rms"] = _safe_scalar(_rms(y_dc - y_before, self.eps))
                    diag["lfap_amp_only_effect_rms"] = _safe_scalar(_rms(y_amp - y_dc, self.eps))
                    diag["lfap_phase_only_effect_rms"] = _safe_scalar(_rms(y_phase - y_dc, self.eps))
                    diag["lfap_amp_phase_effect_rms"] = _safe_scalar(_rms(y_amp_phase - y_dc, self.eps))
                    diag["lfap_final_effect_rms"] = _safe_scalar(_rms(y_final - y_dc, self.eps))
                    diag["lfap_dc_effect_ratio"] = diag["lfap_dc_effect_rms"] / denom
                    diag["lfap_amp_only_effect_ratio"] = diag["lfap_amp_only_effect_rms"] / denom
                    diag["lfap_phase_only_effect_ratio"] = diag["lfap_phase_only_effect_rms"] / denom
                    diag["lfap_amp_phase_effect_ratio"] = diag["lfap_amp_phase_effect_rms"] / denom
                    diag["lfap_final_effect_ratio"] = diag["lfap_final_effect_rms"] / denom

                    diag["lfap_y_before_mse"] = _safe_scalar(_mse(y_before, y_true_flat))
                    diag["lfap_y_dc_only_mse"] = _safe_scalar(_mse(y_dc, y_true_flat))
                    diag["lfap_y_amp_only_mse"] = _safe_scalar(_mse(y_amp, y_true_flat))
                    diag["lfap_y_phase_only_mse"] = _safe_scalar(_mse(y_phase, y_true_flat))
                    diag["lfap_y_amp_phase_mse"] = _safe_scalar(_mse(y_amp_phase, y_true_flat))
                    diag["lfap_y_final_mse"] = _safe_scalar(_mse(y_final, y_true_flat))
                    diag["lfap_y_before_mae"] = _safe_scalar(_mae(y_before, y_true_flat))
                    diag["lfap_y_dc_only_mae"] = _safe_scalar(_mae(y_dc, y_true_flat))
                    diag["lfap_y_amp_only_mae"] = _safe_scalar(_mae(y_amp, y_true_flat))
                    diag["lfap_y_phase_only_mae"] = _safe_scalar(_mae(y_phase, y_true_flat))
                    diag["lfap_y_amp_phase_mae"] = _safe_scalar(_mae(y_amp_phase, y_true_flat))
                    diag["lfap_y_final_mae"] = _safe_scalar(_mae(y_final, y_true_flat))
                    diag["lfap_amp_only_mse_delta"] = diag["lfap_y_amp_only_mse"] - diag["lfap_y_dc_only_mse"]
                    diag["lfap_phase_only_mse_delta"] = diag["lfap_y_phase_only_mse"] - diag["lfap_y_dc_only_mse"]
                    diag["lfap_amp_phase_mse_delta"] = diag["lfap_y_amp_phase_mse"] - diag["lfap_y_dc_only_mse"]
                    diag["lfap_final_mse_delta"] = diag["lfap_y_final_mse"] - diag["lfap_y_dc_only_mse"]
                    self._last_lfap_task_mse_from_cache = diag["lfap_y_final_mse"]
                    self._last_lfap_recon_gain_vs_dc = diag["lfap_y_dc_only_mse"] - diag["lfap_y_final_mse"]
                    self._last_lfap_recon_gain_vs_amp = diag["lfap_y_amp_only_mse"] - diag["lfap_y_final_mse"]

                    for k_idx in range(_LFAP_PER_FREQ_MAX_K):
                        key_id = k_idx + 1
                        if k_idx < self.lfap_k:
                            pred_minus_carrier_k = pred_minus_carrier[:, :, k_idx, :]
                            pred_minus_oracle_k = pred_minus_oracle[:, :, k_idx, :]
                            carrier_minus_oracle_k = carrier_minus_oracle[:, :, k_idx, :]
                            amp_ratio_k = amp_ratio[:, :, k_idx, :]
                            pred_delta_k = pred_delta_phase[:, :, k_idx, :]
                            target_delta_k = target_delta_phase[:, :, k_idx, :]
                            phase_err_k = phase_err[:, :, k_idx, :]
                            carrier_complex_k = self._lfap_carrier_fft_low[:, :, k_idx, :]
                            oracle_complex_k = Y_true_low[:, :, k_idx, :]
                            pred_complex_k = torch.polar(
                                torch.exp(pred_logA[:, :, k_idx, :]),
                                carrier_phase[:, :, k_idx, :] + pred_delta_k,
                            )
                            carrier_complex_err_k = (
                                carrier_complex_k - oracle_complex_k
                            ).abs().pow(2)
                            pred_complex_err_k = (
                                pred_complex_k - oracle_complex_k
                            ).abs().pow(2)
                            carrier_complex_mse_k = carrier_complex_err_k.mean()
                            pred_complex_mse_k = pred_complex_err_k.mean()
                            complex_gain_k = (
                                carrier_complex_mse_k - pred_complex_mse_k
                            )
                            diag[f"lfap_f{key_id}_pred_minus_carrier_logA_rms"] = _safe_scalar(_rms(pred_minus_carrier_k, self.eps))
                            diag[f"lfap_f{key_id}_pred_minus_oracle_logA_rms"] = _safe_scalar(_rms(pred_minus_oracle_k, self.eps))
                            diag[f"lfap_f{key_id}_carrier_minus_oracle_logA_rms"] = _safe_scalar(_rms(carrier_minus_oracle_k, self.eps))
                            diag[f"lfap_f{key_id}_amp_ratio_p95"] = _safe_scalar(_q(amp_ratio_k, 0.95))
                            diag[f"lfap_f{key_id}_delta_phase_rms"] = _safe_scalar(_rms(pred_delta_k, self.eps))
                            diag[f"lfap_f{key_id}_target_delta_phase_rms"] = _safe_scalar(_rms(target_delta_k, self.eps))
                            diag[f"lfap_f{key_id}_phase_err_rms"] = _safe_scalar(_rms(phase_err_k, self.eps))
                            diag[f"lfap_f{key_id}_phase_cos"] = _safe_scalar(torch.cos(phase_err_k))
                            diag[f"lfap_f{key_id}_carrier_complex_mse"] = _safe_scalar(
                                carrier_complex_mse_k
                            )
                            diag[f"lfap_f{key_id}_pred_complex_mse"] = _safe_scalar(
                                pred_complex_mse_k
                            )
                            diag[f"lfap_f{key_id}_complex_gain"] = _safe_scalar(
                                complex_gain_k
                            )
                            diag[f"lfap_f{key_id}_rel_complex_gain"] = _safe_scalar(
                                complex_gain_k / (carrier_complex_mse_k + self.eps)
                            )
                            diag[f"lfap_f{key_id}_pred_better_ratio"] = _safe_scalar(
                                pred_complex_err_k < carrier_complex_err_k
                            )
                            diag[f"lfap_f{key_id}_oracle_energy_share"] = _safe_scalar(
                                oracle_complex_k.abs().pow(2).mean()
                                / (
                                    Y_true_low.abs().pow(2).sum(dim=2).mean()
                                    + self.eps
                                )
                            )
                            diag[f"lfap_f{key_id}_oracle_carrier_amp_ratio_p95"] = _safe_scalar(
                                _q(
                                    torch.exp(
                                        oracle_logA[:, :, k_idx, :]
                                        - carrier_logA[:, :, k_idx, :]
                                    ),
                                    0.95,
                                )
                            )

                    carrier_z = torch.polar(torch.exp(carrier_logA), carrier_phase)
                    pred_z = torch.polar(
                        torch.exp(pred_logA),
                        carrier_phase + pred_delta_phase,
                    )
                    oracle_z = Y_true_low.detach()
                    carrier_err = (carrier_z - oracle_z).abs().pow(2)
                    pred_err = (pred_z - oracle_z).abs().pow(2)
                    oracle_energy = oracle_z.abs().pow(2)
                    total_energy = oracle_energy.mean(dim=(0, 1, 3)).sum().clamp_min(self.eps)

                    for rank in range(1, 5):
                        k_idx = self.lfap_k - rank
                        if k_idx < 0:
                            continue
                        key = f"lfap_tail{rank}"
                        carrier_err_k = carrier_err[:, :, k_idx, :]
                        pred_err_k = pred_err[:, :, k_idx, :]
                        energy_k = oracle_energy[:, :, k_idx, :]
                        phase_err_k = phase_err[:, :, k_idx, :]
                        carrier_mse = carrier_err_k.mean()
                        pred_mse = pred_err_k.mean()
                        gain = carrier_mse - pred_mse
                        diag[f"{key}_bin_idx"] = float(k_idx + 1)
                        diag[f"{key}_energy_share"] = float(
                            (energy_k.mean() / total_energy).detach().item()
                        )
                        diag[f"{key}_carrier_complex_mse"] = float(
                            carrier_mse.detach().item()
                        )
                        diag[f"{key}_pred_complex_mse"] = float(
                            pred_mse.detach().item()
                        )
                        diag[f"{key}_complex_gain"] = float(gain.detach().item())
                        diag[f"{key}_rel_complex_gain"] = float(
                            (gain / (carrier_mse + self.eps)).detach().item()
                        )
                        diag[f"{key}_pred_better_ratio"] = float(
                            (pred_err_k < carrier_err_k).float().mean().detach().item()
                        )
                        diag[f"{key}_phase_err_rms"] = _safe_scalar(
                            _rms(phase_err_k, self.eps)
                        )
                        diag[f"{key}_phase_cos"] = _safe_scalar(
                            torch.cos(phase_err_k)
                        )

                    diag["lfap_nan_ratio_pred_logA"] = float(torch.isnan(pred_logA).float().mean().item())
                    diag["lfap_inf_ratio_pred_logA"] = float(torch.isinf(pred_logA).float().mean().item())
                    diag["lfap_nan_ratio_delta_phase"] = float(torch.isnan(pred_delta_phase).float().mean().item())
                    diag["lfap_inf_ratio_delta_phase"] = float(torch.isinf(pred_delta_phase).float().mean().item())
                    diag["lfap_nan_ratio_y_final"] = float(torch.isnan(y_final).float().mean().item())
                    diag["lfap_inf_ratio_y_final"] = float(torch.isinf(y_final).float().mean().item())
                    diag["lfap_pred_logA_abs_max"] = _safe_scalar(_max_abs_safe(pred_logA))
                    diag["lfap_delta_phase_abs_max"] = _safe_scalar(_max_abs_safe(pred_delta_phase))
                    diag["lfap_y_final_abs_max"] = _safe_scalar(_max_abs_safe(y_final))
                    diag["lfap_grad_sensitive_amp_ratio_gt_10"] = _safe_scalar(_bool_ratio((amp_ratio > 10) & torch.isfinite(amp_ratio)))
                    diag["lfap_grad_sensitive_amp_ratio_gt_100"] = _safe_scalar(_bool_ratio((amp_ratio > 100) & torch.isfinite(amp_ratio)))
        except Exception:
            diag = self._diag_zeros()

        self._last_lfap_diag_stats = diag
        return state_loss

    def compute_total_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        return self.compute_base_aux_loss(y_true)

    def loss(self, y_true: torch.Tensor) -> torch.Tensor:
        return self.compute_total_aux_loss(y_true)

    def get_last_aux_stats(self) -> dict:
        out = {
            "aux_total": self._last_aux_total,
            "base_aux_loss": self._last_base_aux_loss,
            "mu_loss": self._last_mu_loss,
            "std_loss": self._last_std_loss,
            "phase_loss": self._last_phase_loss,
            "route_state_loss": self._last_route_state_loss,
            "lfap_k": self.lfap_k,
            "lfap_mu_loss": self._last_lfap_mu_loss,
            "lfap_amp_loss": self._last_lfap_amp_loss,
            "lfap_phase_loss": self._last_lfap_phase_loss,
            "lfap_state_loss": self._last_lfap_state_loss,
            "lfap_delta_logA_rms": self._last_lfap_delta_loga_rms,
            "lfap_delta_phase_rms": self._last_lfap_delta_phase_rms,
            "lfap_effect_rms": self._last_lfap_effect_rms,
            "lfap_effect_ratio": self._last_lfap_effect_ratio,
            "lfap_pred_residual_rms_before": self._last_lfap_pred_residual_rms_before,
            "lfap_pred_residual_rms_after": self._last_lfap_pred_residual_rms_after,
            "lfap_oracle_residual_rms": self._last_lfap_oracle_residual_rms,
            "lfap_residual_rms_ratio": self._last_lfap_residual_rms_ratio,
            "lfap_task_mse_from_cache": self._last_lfap_task_mse_from_cache,
            "lfap_recon_gain_vs_dc": self._last_lfap_recon_gain_vs_dc,
            "lfap_recon_gain_vs_amp": self._last_lfap_recon_gain_vs_amp,
        }
        out.update(self._last_lfap_diag_stats)
        return out

    def get_route_diagnostics(self) -> dict:
        return {
            "route_path": self.route_path,
            "route_state": self.route_state,
            "san_lowfreq_ap": True,
            "lfap_k": self.lfap_k,
            "lfap_mu_loss": self._last_lfap_mu_loss,
            "lfap_amp_loss": self._last_lfap_amp_loss,
            "lfap_phase_loss": self._last_lfap_phase_loss,
            "lfap_state_loss": self._last_lfap_state_loss,
            "lfap_delta_logA_rms": self._last_lfap_delta_loga_rms,
            "lfap_delta_phase_rms": self._last_lfap_delta_phase_rms,
            "lfap_effect_rms": self._last_lfap_effect_rms,
            "lfap_effect_ratio": self._last_lfap_effect_ratio,
            "lfap_pred_residual_rms_before": self._last_lfap_pred_residual_rms_before,
            "lfap_pred_residual_rms_after": self._last_lfap_pred_residual_rms_after,
            "lfap_oracle_residual_rms": self._last_lfap_oracle_residual_rms,
            "lfap_residual_rms_ratio": self._last_lfap_residual_rms_ratio,
            "lfap_task_mse_from_cache": self._last_lfap_task_mse_from_cache,
            "lfap_recon_gain_vs_dc": self._last_lfap_recon_gain_vs_dc,
            "lfap_recon_gain_vs_amp": self._last_lfap_recon_gain_vs_amp,
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
