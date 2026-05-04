"""SANRouteNorm — exact SAN base + output-side route framework.

Design principles
-----------------
- Exact original SAN is the base family.
- normalize() prepares and caches all state features needed by denormalize() and loss().
- The base predictor (nm.predictor) is always trained in Stage 1.
- For route_path="none" (pure SAN) the output is numerically identical to the
  official SAN implementation.

Pure SAN (route_path="none", route_state="none")
------------------------------------------------
  stride MUST equal period_len (non-overlap patches only).
  normalize():
    x reshaped to (B, P_hist, period_len, C).
    patch mean/std computed per patch.
    z = (x - mean) / (std + eps), reshaped back to (B, T, C).
    Predictor receives raw (un-normalised) flattened windows xbar_raw:
      mean head input: (hist_mean - mean_all) and (raw_x - mean_all)
      std  head input: hist_std               and raw_x
  denormalize():
    y_norm reshaped to (B, P_fut, period_len, C).
    y = y_norm * (pred_std + eps) + pred_mean, reshaped back to (B, pred_len, C).
  Stage 1: train SAN predictor only, supervised on oracle patch mean/std MSE.
  Stage 2: freeze predictor, train fm only, task loss.

Route layering
--------------
  All routes EXCEPT timeapn_correction build on the exact SAN base.

lp_state + lp_state_correction
--------------------------------
  lp slow state: mu_lp = patch_mean(lowpass_time(raw_series))
  denormalize uses lp mean + base std.  Stages: stage1, lp_pretrain, stage2.

timeapn_correction + timeapn_state
----------------------------------------------------------
  Uses OfficialAPN (timeapn_official.py) in place of the SAN base predictor.
  This pair does NOT use the SAN base predictor at all.
  See timeapn_official.py for the module interface.

  normalize():
    Calls apn.normalize(x):
      - norm_x, pred_ms, seq_ms = apn.normalize(x)
      - Caches pred_ms = (pred_m, pred_s) and seq_ms = (seq_m, seq_s).
      - Returns norm_x.

  denormalize(y_norm):
    1. Compute seq_ms_y = (seq_m_y, seq_s_y) from y_norm via apn.norm_sliding
       (needed for phase supervision in loss, NOT used in de_normalize).
    2. Build station_pred_raw  = cat([pred_m, pred_s]).T  — raw predicted station.
       Build station_pred_loss = cat([pred_m, seq_s_y + pred_s]).T  — for phase loss.
    3. Apply apn.de_normalize(y_norm, station_pred_raw) → y_out.
       (de_normalize receives raw pred_s, not combined phase)
    4. Cache y_out, station_pred_raw, station_pred_loss.

  compute_route_state_loss(y_true):
    Implements reconstruction + mean + phase supervision loss:
      _, (true_m, true_phi) = apn.norm_sliding(y_true_BCT)
      station_pred_loss from cache → (pred_mean, seq_s_y + pred_s)
      loss_mean  = MSE(pred_mean, true_m_transposed)
      loss_phase = MSE(combined_phase, true_phi_transposed)
      loss_recon = MSE(y_out, y_true)
      total = loss_recon + loss_mean + loss_phase

  Training stages (enforced by train_state_routes.py):
    timeapn_pretrain: station_optim on APN params only; fm runs forward but not updated.
    stage2:           fm_optim only; APN stays frozen unless timeapn_enable_late_merge=True.

  Other routes (local_transport, residual_content, alignment, gating,
  slope_state_correction): overlap-add generalised implementation unchanged.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .san_route_paths import build_route_path
from .san_route_states import build_route_state
from .timeapn_official import OfficialAPN

# Valid route_path values (including backward-compat alias)
_VALID_PATHS = {
    "none",
    "local_transport",
    "residual_content",
    "alignment",
    "gating",
    "lp_state_correction",
    "local_value_parameter",    # alias → local_transport
    "slope_state_correction",
    "timeapn_correction",
}
_VALID_STATES = {
    "none",
    "nu",
    "dlogsigma",
    "omega_spec",
    "lp_state",
    "slope_state",
    "timeapn_state",
}


def _normalize_patch_state(feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Normalise patch features per-batch, per-channel across the patch dimension.

    feat: (B, P, C)
    Returns (feat - mean) / std where mean and std are computed over dim=1 (patches).
    Applied to nu and dlogsigma by default.  omega_spec and lp_state are exempt.
    """
    mean = feat.mean(dim=1, keepdim=True)
    std  = feat.std(dim=1, keepdim=True).clamp_min(eps)
    return (feat - mean) / std


class _LPStatePredictor(nn.Module):
    """Channel-wise MLP predicting future low-pass patch mean."""

    def __init__(self, hist_stat_len: int, pred_stat_len: int, hidden: int = 128):
        super().__init__()
        self.enc = nn.Linear(hist_stat_len, hidden)
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden, pred_stat_len)

    def forward(self, mu_hist: torch.Tensor) -> torch.Tensor:
        x = mu_hist.permute(0, 2, 1)
        return self.out(self.act(self.enc(x))).permute(0, 2, 1)


class SANRouteNorm(nn.Module):
    """Single-level SAN base for state-path routing experiments.

    Args:
        seq_len:               Length of the historical input window.
        pred_len:              Length of the prediction horizon.
        period_len:            Window (patch/slice) length.
        enc_in:                Number of input channels.
        stride:                Stride between windows. Defaults to period_len.
        sigma_min:             Minimum allowed std (clamp floor).
        w_mu:                  Weight for mu component of base aux loss.
        w_std:                 Weight for std component of base aux loss.
        route_path:            One of _VALID_PATHS.
        route_state:           One of _VALID_STATES.
        route_state_loss_scale: Weight for route state loss.
        san_ablation_mode:     "none", "seq_std", or "base_mean_only".

        # TimeAPN parameters (only used when route_path="timeapn_correction")
        timeapn_j, timeapn_learnable, timeapn_wavelet, timeapn_dr,
        timeapn_kernel_len, timeapn_hkernel_len,
        timeapn_pd_model, timeapn_pd_ff, timeapn_pe_layers, timeapn_data_path
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
        route_path: str = "none",
        route_state: str = "none",
        route_state_loss_scale: float = 0.1,
        san_ablation_mode: str = "none",
        # TimeAPN-specific (official APN defaults)
        timeapn_j: int = 1,
        timeapn_learnable: bool = True,
        timeapn_wavelet: str = "bior3.5",
        timeapn_dr: float = 0.05,
        timeapn_kernel_len: int = 7,
        timeapn_hkernel_len: int = 5,
        timeapn_pd_model: int = 128,
        timeapn_pd_ff: int = 128,
        timeapn_pe_layers: int = 2,
        timeapn_data_path: Optional[str] = None,
        # B2SC inference-time update parameters
        b2sc_enable: bool = False,
        b2sc_recent_weight: float = 0.75,
        b2sc_prev_weight: float = 0.25,
        b2sc_second_slice_scale: float = 0.5,
        # SSC (streaming state calibration) parameters
        ssc_enable: bool = False,
        ssc_lambda: float = 0.9,
        ssc_decay_rho: float = 0.5,
        ssc_scale_beta: float = 0.5,
        ssc_trend_scale: float = 0.1,
        # Phase correction (pure baseline only: route_path="none", route_state="none")
        use_phase: bool = False,
        phase_k: int = 8,
        phase_zero_init: bool = True,
        phase_loss_weight: float = 1.0,
        phase_energy_gate: bool = False,
        phase_energy_gate_center: float = 0.25,
        phase_energy_gate_temp: float = 0.05,
        phase_energy_gate_strength: float = 0.3,
        phase_energy_gate_min: float = 0.7,
        phase_energy_gate_eps: float = 1e-8,
        phase_energy_mean_gate: bool = False,
        phase_energy_mean_strength: float = 0.3,
        phase_output_zero_mean: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.seq_len   = int(seq_len)
        self.pred_len  = int(pred_len)
        self.window_len = int(period_len)
        self.stride    = int(stride) if int(stride) > 0 else self.window_len
        self.enc_in    = enc_in
        self.channels  = enc_in
        self.sigma_min = float(sigma_min)
        self.epsilon   = 1e-5
        self.w_mu      = float(w_mu)
        self.w_std     = float(w_std)
        self.route_path  = route_path
        self.route_state = route_state
        self.route_state_loss_scale = float(route_state_loss_scale)
        self.san_ablation_mode = str(san_ablation_mode)
        if self.san_ablation_mode not in ("none", "seq_std", "base_mean_only"):
            raise ValueError(
                "san_ablation_mode must be one of "
                "('none', 'seq_std', 'base_mean_only'), "
                f"got '{self.san_ablation_mode}'."
            )
        if self.san_ablation_mode != "none":
            if route_path != "none" or route_state != "none":
                raise ValueError(
                    "san_ablation_mode requires route_path='none' and "
                    "route_state='none'."
                )
            use_phase = False
            phase_energy_gate = False
            phase_energy_mean_gate = False
            phase_output_zero_mean = False
            phase_loss_weight = 0.0

        # ------------------------------------------------------------------
        # B2SC inference-time update
        # ------------------------------------------------------------------
        if b2sc_recent_weight < 0 or b2sc_prev_weight < 0 or b2sc_second_slice_scale < 0:
            raise ValueError(
                "b2sc_recent_weight, b2sc_prev_weight, and b2sc_second_slice_scale "
                "must each be >= 0."
            )
        if b2sc_recent_weight + b2sc_prev_weight <= 0:
            raise ValueError(
                "b2sc_recent_weight + b2sc_prev_weight must be > 0."
            )
        self.b2sc_enable             = bool(b2sc_enable)
        self.b2sc_recent_weight      = float(b2sc_recent_weight)
        self.b2sc_prev_weight        = float(b2sc_prev_weight)
        self.b2sc_second_slice_scale = float(b2sc_second_slice_scale)
        self._b2sc_mode              = False

        # ------------------------------------------------------------------
        # SSC (streaming state calibration)
        # ------------------------------------------------------------------
        self.ssc_enable = bool(ssc_enable)
        self.ssc_lambda = float(ssc_lambda)
        if self.ssc_lambda < 0.0 or self.ssc_lambda >= 1.0:
            raise ValueError("ssc_lambda must satisfy 0 <= ssc_lambda < 1.")
        self.ssc_decay_rho = float(ssc_decay_rho)
        if self.ssc_decay_rho <= 0.0 or self.ssc_decay_rho > 1.0:
            raise ValueError("ssc_decay_rho must satisfy 0 < ssc_decay_rho <= 1.")
        self.ssc_scale_beta = float(ssc_scale_beta)
        if self.ssc_scale_beta < 0.0:
            raise ValueError("ssc_scale_beta must satisfy ssc_scale_beta >= 0.")
        self.ssc_trend_scale = float(ssc_trend_scale)
        if self.ssc_enable and self.route_path != "none":
            raise ValueError("SSC supports pure SAN only.")

        # Phase correction: only active for pure baseline (route_path/state both "none").
        # Silently disabled for any other route combination.
        self.use_phase = bool(use_phase) and (route_path == "none") and (route_state == "none")
        self._phase_k  = int(phase_k)
        self._phase_loss_weight = float(phase_loss_weight)
        self.phase_energy_gate          = bool(phase_energy_gate)
        self.phase_energy_gate_center   = float(phase_energy_gate_center)
        self.phase_energy_gate_temp     = float(phase_energy_gate_temp)
        self.phase_energy_gate_strength = float(phase_energy_gate_strength)
        self.phase_energy_gate_min      = float(phase_energy_gate_min)
        self.phase_energy_gate_eps      = float(phase_energy_gate_eps)
        self.phase_energy_mean_gate     = bool(phase_energy_mean_gate)
        self.phase_energy_mean_strength = float(phase_energy_mean_strength)
        self.phase_output_zero_mean     = bool(phase_output_zero_mean)

        self._validate_config()

        self.hist_stat_len = self._compute_n_windows(self.seq_len)
        self.pred_stat_len = self._compute_n_windows(self.pred_len)
        raw_hist_len = self.hist_stat_len * self.window_len

        # ------------------------------------------------------------------
        # SAN base predictor:
        #   route_path="none" + use_phase=True  → _PureSANSeqPredictor (mean+phase)
        #   route_path="none" + use_phase=False → _SANBasePredictor
        #   other routes (except timeapn pair)  → _SANBasePredictor
        # ------------------------------------------------------------------
        self.predictor: Optional[nn.Module] = None
        if route_path == "none" and self.san_ablation_mode == "seq_std":
            self.predictor = _PureSANSeqPredictor(
                hist_stat_len=self.hist_stat_len,
                pred_stat_len=self.pred_stat_len,
                window_len=self.window_len,
                enc_in=enc_in,
                sigma_min=sigma_min,
                use_phase=False,
                phase_k=phase_k,
                phase_zero_init=phase_zero_init,
                state_residual_scale_init=0.1,
            )
        elif route_path == "none" and self.san_ablation_mode == "base_mean_only":
            self.predictor = _SANBasePredictor(
                hist_stat_len=self.hist_stat_len,
                raw_hist_len=raw_hist_len,
                pred_stat_len=self.pred_stat_len,
                enc_in=enc_in,
                sigma_min=sigma_min,
            )
        elif route_path == "none" and self.use_phase:
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
            )
        elif route_path != "timeapn_correction":
            self.predictor = _SANBasePredictor(
                hist_stat_len=self.hist_stat_len,
                raw_hist_len=raw_hist_len,
                pred_stat_len=self.pred_stat_len,
                enc_in=enc_in,
                sigma_min=sigma_min,
            )

        # ------------------------------------------------------------------
        # Route-specific modules
        # ------------------------------------------------------------------
        self.lp_state_predictor: Optional[nn.Module]  = None
        self.route_state_predictor: Optional[nn.Module] = None
        self.route_path_impl: Optional[nn.Module]     = None
        self.route_state_impl                          = None
        self.apn_module: Optional[OfficialAPN]         = None

        if route_path == "lp_state_correction":
            self.lp_state_predictor = _LPStatePredictor(
                hist_stat_len=self.hist_stat_len,
                pred_stat_len=self.pred_stat_len,
            )

        elif route_path == "timeapn_correction":
            self.apn_module = OfficialAPN(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=enc_in,
                kernel_len=timeapn_kernel_len,
                hkernel_len=timeapn_hkernel_len,
                j=timeapn_j,
                learnable=timeapn_learnable,
                wavelet=timeapn_wavelet,
                dr=timeapn_dr,
                pd_model=timeapn_pd_model,
                pd_ff=timeapn_pd_ff,
                pe_layers=timeapn_pe_layers,
                data_path=timeapn_data_path,
            )

        elif route_path != "none":
            self.route_state_predictor = RouteStatePredictor(
                hist_stat_len=self.hist_stat_len,
                raw_hist_len=raw_hist_len,
                pred_stat_len=self.pred_stat_len,
                enc_in=enc_in,
            )
            self.route_path_impl = build_route_path(
                route_path, pred_len=self.pred_len, enc_in=enc_in, sigma_min=sigma_min
            )
            self.route_state_impl = build_route_state(route_state)

        self._reset_cache()
        # SSC persistent state — must live outside _reset_cache so it
        # survives per-sample normalize() calls during sequential eval.
        self._ssc_mode:            bool                   = False
        self._ssc_trend_mem:       Optional[torch.Tensor] = None
        self._ssc_prev_obs_point:  Optional[torch.Tensor] = None
        self._ssc_prev_mu_pred:    Optional[torch.Tensor] = None
        self._ssc_prev_std_pred:   Optional[torch.Tensor] = None
        self._ssc_pred_queue:      deque                  = deque()

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if self.route_path not in _VALID_PATHS:
            raise ValueError(
                f"route_path must be one of {_VALID_PATHS}, got '{self.route_path}'."
            )
        if self.route_state not in _VALID_STATES:
            raise ValueError(
                f"route_state must be one of {_VALID_STATES}, got '{self.route_state}'."
            )
        if self.route_path == "none" and self.route_state != "none":
            raise ValueError("When route_path='none', route_state must also be 'none'.")
        if self.route_path != "none" and self.route_state == "none":
            raise ValueError(
                f"When route_path='{self.route_path}', route_state must not be 'none'."
            )
        # lp pair
        if self.route_path == "lp_state_correction" and self.route_state != "lp_state":
            raise ValueError(
                "route_path='lp_state_correction' requires route_state='lp_state'."
            )
        if self.route_state == "lp_state" and self.route_path != "lp_state_correction":
            raise ValueError(
                f"route_state='lp_state' requires route_path='lp_state_correction',"
                f" got '{self.route_path}'."
            )
        # timeapn pair
        if self.route_path == "timeapn_correction" and self.route_state != "timeapn_state":
            raise ValueError(
                "route_path='timeapn_correction' requires route_state='timeapn_state'."
            )
        if self.route_state == "timeapn_state" and self.route_path != "timeapn_correction":
            raise ValueError(
                f"route_state='timeapn_state' requires route_path='timeapn_correction',"
                f" got '{self.route_path}'."
            )
        if self.seq_len < self.window_len:
            raise ValueError(f"seq_len={self.seq_len} < window_len={self.window_len}.")
        if self.pred_len < self.window_len:
            raise ValueError(f"pred_len={self.pred_len} < window_len={self.window_len}.")
        if self.route_path != "timeapn_correction":
            # stride/window checks only for SAN-based routes
            if (self.seq_len - self.window_len) % self.stride != 0:
                raise ValueError(
                    f"(seq_len - window_len) % stride != 0: "
                    f"seq_len={self.seq_len}, window_len={self.window_len}, "
                    f"stride={self.stride}."
                )
            if (self.pred_len - self.window_len) % self.stride != 0:
                raise ValueError(
                    f"(pred_len - window_len) % stride != 0: "
                    f"pred_len={self.pred_len}, window_len={self.window_len}, "
                    f"stride={self.stride}."
                )

    # ------------------------------------------------------------------
    # Window utilities (used by SAN-based routes)
    # ------------------------------------------------------------------

    def _compute_n_windows(self, length: int) -> int:
        return (length - self.window_len) // self.stride + 1

    def _extract_windows(self, x: torch.Tensor) -> torch.Tensor:
        windows = x.unfold(dimension=1, size=self.window_len, step=self.stride)
        return windows.permute(0, 1, 3, 2).contiguous()

    def _compute_window_stats(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = windows.mean(dim=2)
        std  = windows.std(dim=2).clamp(min=self.sigma_min)
        return mean, std

    def _window_stats_to_time_stats(
        self, mean: torch.Tensor, std: torch.Tensor, total_length: int
    ) -> torch.Tensor:
        B, N, C = mean.shape
        mu_t    = torch.zeros(B, total_length, C, device=mean.device, dtype=mean.dtype)
        e2_t    = torch.zeros_like(mu_t)
        counts  = torch.zeros_like(mu_t)
        e2 = std.pow(2) + mean.pow(2)
        for i in range(N):
            s = i * self.stride
            e = s + self.window_len
            mu_t[:, s:e, :]   += mean[:, i:i+1, :]
            e2_t[:, s:e, :]   += e2[:, i:i+1, :]
            counts[:, s:e, :] += 1.0
        mu_t  = mu_t  / counts.clamp_min(1.0)
        e2_t  = e2_t  / counts.clamp_min(1.0)
        sigma_t = torch.sqrt(
            torch.clamp(e2_t - mu_t.pow(2), min=self.sigma_min ** 2)
        )
        return torch.cat([mu_t, sigma_t], dim=-1)

    def _patch_features_to_time(self, feat: torch.Tensor, total_length: int) -> torch.Tensor:
        B, P, C = feat.shape
        out    = torch.zeros(B, total_length, C, device=feat.device, dtype=feat.dtype)
        counts = torch.zeros_like(out)
        for i in range(P):
            s = i * self.stride
            e = s + self.window_len
            out[:, s:e, :]    += feat[:, i:i+1, :]
            counts[:, s:e, :] += 1.0
        return out / counts.clamp_min(1.0)

    def _split_stats(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return stats[:, :, :self.channels], stats[:, :, self.channels:]

    def _time_to_patch_mean(self, x_time: torch.Tensor) -> torch.Tensor:
        wins = self._extract_windows(x_time)
        mu, _ = self._compute_window_stats(wins)
        return mu

    def _lowpass_time_series(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T == 1:
            return x
        k = x.new_tensor([1.0, 2.0, 1.0]) / 4.0
        xt = x.permute(0, 2, 1).reshape(B * C, 1, T)
        xt_padded = F.pad(xt, (1, 1), mode="reflect")
        w = k.view(1, 1, 3)
        out = F.conv1d(xt_padded, w, padding=0)
        return out.reshape(B, C, T).permute(0, 2, 1)

    def _lowpass_time_to_patch_mean(self, x: torch.Tensor) -> torch.Tensor:
        lp = self._lowpass_time_series(x)
        wins = self._extract_windows(lp)
        mu, _ = self._compute_window_stats(wins)
        return mu

    def _lowpass_patch_mean(self, feat: torch.Tensor) -> torch.Tensor:
        B, P, C = feat.shape
        if P == 1:
            return feat
        k = feat.new_tensor([1.0, 2.0, 1.0]) / 4.0
        xp = feat.permute(0, 2, 1).reshape(B * C, 1, P)
        xp_padded = F.pad(xp, (1, 1), mode="reflect")
        w = k.view(1, 1, 3)
        out = F.conv1d(xp_padded, w, padding=0)
        return out.reshape(B, C, P).permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _reset_cache(self) -> None:
        # SAN base
        self._mu_hist: Optional[torch.Tensor]        = None
        self._std_hist: Optional[torch.Tensor]       = None
        self._base_mu_fut_hat: Optional[torch.Tensor]  = None
        self._base_std_fut_hat: Optional[torch.Tensor] = None
        self._pred_time_stats: Optional[torch.Tensor]  = None
        self._base_time_mean: Optional[torch.Tensor]   = None
        self._base_time_std: Optional[torch.Tensor]    = None
        # Generic route
        self._route_hist_state: Optional[torch.Tensor]            = None
        self._route_future_state_hat: Optional[torch.Tensor]      = None
        self._route_future_state_time: Optional[torch.Tensor]     = None
        self._route_future_oracle_state: Optional[torch.Tensor]   = None
        self._route_hist_state_raw: Optional[torch.Tensor]        = None
        self._route_future_state_hat_raw: Optional[torch.Tensor]  = None
        # lp_state
        self._lp_mu_hist: Optional[torch.Tensor]     = None
        self._lp_mu_fut_hat: Optional[torch.Tensor]  = None
        self._oracle_lp_mu_fut: Optional[torch.Tensor] = None
        # TimeAPN (official APN caches)
        self._apn_pred_ms: Optional[tuple]            = None   # (pred_m, pred_s)
        self._apn_seq_ms: Optional[tuple]             = None   # (seq_m, seq_s)
        self._apn_station_pred_raw:  Optional[torch.Tensor] = None  # (B, pred_len, 2C) raw [pred_m, pred_s]
        self._apn_station_pred_loss: Optional[torch.Tensor] = None  # (B, pred_len, 2C) [pred_m, seq_s_y+pred_s]
        self._apn_y_out: Optional[torch.Tensor]             = None  # final denorm output
        # Diagnostics
        self._route_path_name:  str = self.route_path
        self._route_state_name: str = self.route_state
        self._last_mu_loss:           float = 0.0
        self._last_std_loss:          float = 0.0
        self._last_base_aux_loss:     float = 0.0
        self._last_phase_loss:        float = 0.0
        self._last_route_state_loss:  float = 0.0
        self._last_aux_total:         float = 0.0
        # phase aux cache (pure baseline + phase mode)
        self._hist_last_patch_norm: Optional[torch.Tensor] = None  # (B, W, C) mean-normalized
        self._phase_rot_fut_hat: Optional[torch.Tensor]    = None  # (B, P_fut, K, C, 2) residual rotation
        self._phase_energy_gate:   Optional[torch.Tensor] = None  # (B, C)
        self._phase_energy_r_high: Optional[torch.Tensor] = None  # (B, C)
        self._phase_energy_risk:   Optional[torch.Tensor] = None  # (B, C)
        self._last_phase_output_mean_abs:          float = 0.0
        self._last_phase_output_zero_mean_applied: bool  = False
        self._last_input_residual_rms: float = 0.0
        # output_patch_mean_abs_before_gauge: gauge 删除了多少 patch DC。
        self._last_output_patch_mean_abs_before_gauge: float = 0.0
        self._last_output_rms_before_gauge: float = 0.0
        self._last_output_rms_after_gauge: float = 0.0
        self._last_phase_angle_abs_mean: float = 0.0
        # phase_effect_rms: phase rotation 实际改变 backbone 输出的幅度。
        self._last_phase_effect_rms: float = 0.0
        self._last_phase_effect_ratio: float = 0.0
        self._last_dc_gauge_ratio: float = 0.0
        self._last_pred_residual_rms_after_phase: float = 0.0
        self._last_oracle_residual_rms: float = 0.0
        # residual_rms_ratio: 预测 residual 能量相对 oracle residual 能量是否过大或过小。
        self._last_residual_rms_ratio: float = 0.0
        self._last_apn_loss_mean:     float = 0.0
        self._last_apn_loss_phase:    float = 0.0
        self._last_apn_loss_recon:    float = 0.0
        self._last_apn_conv_effect_rms: float = 0.0
        self._last_apn_conv_effect_ratio: float = 0.0
        self._last_apn_pred_residual_rms: float = 0.0
        self._last_apn_oracle_residual_rms: float = 0.0
        self._last_apn_residual_rms_ratio: float = 0.0
        self._last_apn_kernel_l1: float = 0.0
        self._last_apn_kernel_l2: float = 0.0
        self._last_apn_kernel_max_ratio: float = 0.0
        self._last_apn_kernel_center_ratio: float = 0.0
        self._last_apn_kernel_entropy: float = 0.0
        self._last_apn_phase_abs_err: float = 0.0
        self._last_apn_phase_cos: float = 0.0
        # B2SC cache
        self._b2sc_input_raw:             Optional[torch.Tensor] = None
        self._last_b2sc_applied:          bool                   = False
        self._last_b2sc_delta_mu:         Optional[torch.Tensor] = None
        self._last_b2sc_delta_logsigma:   Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Parameter group accessors
    # ------------------------------------------------------------------

    def parameters_base_predictor(self) -> list:
        """Stage 1 optimizer target — SAN base predictor only."""
        if self.predictor is not None:
            return list(self.predictor.parameters())
        return []

    def parameters_route_modules(self) -> list:
        """Route-module params (lp_pretrain / timeapn_pretrain / stage3)."""
        params: list = []
        if self.lp_state_predictor is not None:
            params.extend(self.lp_state_predictor.parameters())
        if self.route_state_predictor is not None:
            params.extend(self.route_state_predictor.parameters())
        if self.route_path_impl is not None:
            params.extend(self.route_path_impl.parameters())
        if self.apn_module is not None:
            params.extend(self.apn_module.parameters())
        return params

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_base_predictor(self) -> None:
        if self.predictor is not None:
            for p in self.predictor.parameters():
                p.requires_grad_(False)

    def unfreeze_base_predictor(self) -> None:
        if self.predictor is not None:
            for p in self.predictor.parameters():
                p.requires_grad_(True)

    def freeze_route_modules(self) -> None:
        for m in [
            self.lp_state_predictor,
            self.route_state_predictor,
            self.route_path_impl,
            self.apn_module,
        ]:
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(False)

    def unfreeze_route_modules(self) -> None:
        for m in [
            self.lp_state_predictor,
            self.route_state_predictor,
            self.route_path_impl,
            self.apn_module,
        ]:
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(True)

    def _should_normalize_route_state(self) -> bool:
        return self.route_state not in ("omega_spec", "lp_state", "timeapn_state")

    # ------------------------------------------------------------------
    # SSC: streaming state calibration
    # ------------------------------------------------------------------

    def set_ssc_mode(self, enabled: bool) -> None:
        self._ssc_mode = bool(enabled)

    def reset_ssc_state(self) -> None:
        self._ssc_trend_mem      = None
        self._ssc_prev_obs_point = None
        self._ssc_prev_mu_pred   = None
        self._ssc_prev_std_pred  = None
        self._ssc_pred_queue     = deque()

    def ssc_update_from_current_lookback(
        self, x: torch.Tensor, sample_start: int
    ) -> None:
        """Update SSC bias from any queued prediction whose target point is now
        the boundary of the current lookback window.

        ``sample_start`` is the absolute time index of the first point of the
        current lookback ``x`` (shape ``(1, seq_len, C)``).  The boundary point
        of the current lookback is ``sample_start + seq_len - 1``.  A queued
        entry matches when its ``target_point`` equals that boundary point.
        Past entries (target_point < boundary) are discarded; future entries
        stop the scan.
        """
        if not self.ssc_enable or not self._ssc_mode:
            return
        if x.shape[0] != 1:
            raise ValueError("SSC expects single-sample sequential eval.")
        current_boundary_point = int(sample_start) + self.seq_len - 1

        while self._ssc_pred_queue:
            target_point, mu_pred, std_pred = self._ssc_pred_queue[0]

            if target_point < current_boundary_point:
                # Already passed — discard without updating
                self._ssc_pred_queue.popleft()
                continue

            if target_point > current_boundary_point:
                # Not yet reached — stop
                break

            # Exact match: target_point == current_boundary_point
            self._ssc_pred_queue.popleft()
            obs_point = x[:, -1, :].detach()   # (1, C) — last point of the lookback

            if self._ssc_prev_obs_point is None:
                # First observation — cache and wait for the next pair
                self._ssc_prev_obs_point = obs_point
                self._ssc_prev_mu_pred   = mu_pred
                self._ssc_prev_std_pred  = std_pred
                break

            # Compute trend observation: normalised (real change - predicted change)
            d_obs  = obs_point - self._ssc_prev_obs_point
            d_pred = mu_pred   - self._ssc_prev_mu_pred
            denom  = 0.5 * (std_pred + self._ssc_prev_std_pred) + self.epsilon
            trend_obs = (d_obs - d_pred) / denom

            if self._ssc_trend_mem is None:
                self._ssc_trend_mem = trend_obs
            else:
                lam = self.ssc_lambda
                self._ssc_trend_mem = lam * self._ssc_trend_mem + (1.0 - lam) * trend_obs

            self._ssc_prev_obs_point = obs_point
            self._ssc_prev_mu_pred   = mu_pred
            self._ssc_prev_std_pred  = std_pred
            break

    def ssc_cache_current_base_prediction(
        self, first_future_start: int
    ) -> None:
        """Push the boundary-point prediction keyed by its global time index.

        ``first_future_start`` is the absolute time index of the first future
        point (i.e. one step beyond the current lookback).  This is the
        boundary point that will be observed when the window advances by one
        step.
        """
        if not self.ssc_enable or not self._ssc_mode:
            return
        if self._base_mu_fut_hat is None or self._base_std_fut_hat is None:
            return
        mu_pred  = self._base_mu_fut_hat[:, 0, :].detach()
        std_pred = self._base_std_fut_hat[:, 0, :].detach()
        self._ssc_pred_queue.append((int(first_future_start), mu_pred, std_pred))

    def _apply_ssc_to_patch_stats(
        self,
        mu_fut:  torch.Tensor,
        std_fut: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply trend SSC: accumulated correction grows with patch distance."""
        if not self.ssc_enable or not self._ssc_mode:
            return mu_fut, std_fut
        if self._ssc_trend_mem is None:
            return mu_fut, std_fut
        mu_cal = mu_fut.clone()
        for j in range(mu_cal.shape[1]):
            time_scale = (j + 0.5) * self.window_len
            weight = time_scale * (self.ssc_decay_rho ** j)
            mu_cal[:, j, :] = mu_cal[:, j, :] + self.ssc_trend_scale * weight * std_fut[:, j, :] * self._ssc_trend_mem
        return mu_cal, std_fut

    # ------------------------------------------------------------------
    # B2SC: inference-time update
    # ------------------------------------------------------------------

    def set_b2sc_mode(self, enabled: bool) -> None:
        """Enable or disable B2SC correction.  Called by the trainer before eval."""
        self._b2sc_mode = bool(enabled)

    def _b2sc_block_state(
        self, block_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(mu, logstd) summary for a block of shape (B, h, C)."""
        mu_block     = block_x.mean(dim=1)
        logstd_block = torch.log(block_x.std(dim=1).clamp(min=self.sigma_min))
        return mu_block, logstd_block

    def _compute_b2sc_delta(
        self,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Compute (delta_mu, delta_logsigma) from observed block drifts.

        Compares the locally-observed recent drift (between the last two or three
        input blocks) with the base predictor's implicit first-step drift at the
        boundary.  Returns None when B2SC cannot fire.
        """
        if (
            not self.b2sc_enable
            or not self._b2sc_mode
            or self._b2sc_input_raw is None
            or self.predictor is None
            or self.route_path == "timeapn_correction"
        ):
            return None
        if self._base_mu_fut_hat is None or self._base_std_fut_hat is None:
            return None

        x = self._b2sc_input_raw
        h = self.window_len
        L = x.shape[1]
        if L < 2 * h:
            return None

        # --- observed blocks ---
        B1 = x[:, L - h:L,     :]   # most recent
        B2 = x[:, L - 2*h:L-h, :]   # second most recent
        mu1, logstd1 = self._b2sc_block_state(B1)
        mu2, logstd2 = self._b2sc_block_state(B2)

        # --- observed drift (most recent step) ---
        d_recent_mu       = mu1     - mu2
        d_recent_logsigma = logstd1 - logstd2

        if L >= 3 * h:
            B3 = x[:, L - 3*h:L-2*h, :]
            mu3, logstd3 = self._b2sc_block_state(B3)
            d_prev_mu       = mu2     - mu3
            d_prev_logsigma = logstd2 - logstd3

            # Consistency gate: both drifts must agree in sign
            g_mu       = (d_recent_mu       * d_prev_mu       > 0).to(d_recent_mu.dtype)
            g_logsigma = (d_recent_logsigma * d_prev_logsigma > 0).to(d_recent_logsigma.dtype)

            obs_step_mu = g_mu * (
                self.b2sc_recent_weight * d_recent_mu + self.b2sc_prev_weight * d_prev_mu
            )
            obs_step_logsigma = g_logsigma * (
                self.b2sc_recent_weight * d_recent_logsigma
                + self.b2sc_prev_weight * d_prev_logsigma
            )
        else:
            # Only two blocks available: use recent drift directly
            obs_step_mu       = d_recent_mu
            obs_step_logsigma = d_recent_logsigma

        # --- base predictor's implied first-step drift at the boundary ---
        base_mu0    = self._base_mu_fut_hat[:, 0, :]
        base_logstd0 = torch.log(
            self._base_std_fut_hat[:, 0, :].clamp(min=self.sigma_min)
        )
        base_step_mu       = base_mu0    - mu1
        base_step_logsigma = base_logstd0 - logstd1

        # --- correction: observed drift minus base drift ---
        delta_mu       = obs_step_mu       - base_step_mu
        delta_logsigma = obs_step_logsigma - base_step_logsigma

        self._last_b2sc_applied        = True
        self._last_b2sc_delta_mu       = delta_mu.detach()
        self._last_b2sc_delta_logsigma = delta_logsigma.detach()
        return delta_mu, delta_logsigma

    def _apply_b2sc_to_patch_stats(
        self,
        mu_fut:  torch.Tensor,
        std_fut: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Correct patch-level future (mu, std) with B2SC delta.

        Returns the originals unchanged when B2SC is inactive or delta is None.
        Corrections are always in log-sigma space and clipped back to std space.
        """
        result = self._compute_b2sc_delta()
        if result is None:
            return mu_fut, std_fut

        delta_mu, delta_logsigma = result

        logstd_fut = torch.log(std_fut.clamp(min=self.sigma_min))
        mu_cal     = mu_fut.clone()
        logstd_cal = logstd_fut.clone()

        # First future slice
        mu_cal[:, 0, :]     += delta_mu
        logstd_cal[:, 0, :] += delta_logsigma

        # Second future slice (attenuated)
        if mu_fut.shape[1] >= 2:
            mu_cal[:, 1, :]     += self.b2sc_second_slice_scale * delta_mu
            logstd_cal[:, 1, :] += self.b2sc_second_slice_scale * delta_logsigma

        std_cal = logstd_cal.exp().clamp(min=self.sigma_min)
        return mu_cal, std_cal

    # ------------------------------------------------------------------
    # Main interface: normalize
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input.

        timeapn pair:
            Calls OfficialAPN.normalize(x); caches pred_ms / seq_ms.
            Returns normalized x.
        pure SAN / other routes:
            Exact SAN (reshape-based) or overlap-add generalised implementation.
        """
        self._reset_cache()
        self._b2sc_input_raw = x.detach()
        B, T, C = x.shape
        if T != self.seq_len and self.route_path != "timeapn_correction":
            raise ValueError(
                f"SANRouteNorm expected seq_len={self.seq_len}, got {T}."
            )

        is_pure_san = (self.route_path == "none")
        is_timeapn  = (self.route_path == "timeapn_correction")

        # ==============================================================
        # Official TimeAPN path
        # ==============================================================
        if is_timeapn:
            if self.apn_module is None:
                return x
            norm_x, pred_ms, seq_ms = self.apn_module.normalize(x)
            self._apn_pred_ms = pred_ms   # (pred_m, pred_s) each (B, C, pred_len)
            self._apn_seq_ms  = seq_ms    # (seq_m, seq_s)  each (B, C, seq_len)
            return norm_x

        # ==============================================================
        # Exact SAN reshape-based path (pure SAN)
        # ==============================================================
        if is_pure_san:
            P_hist    = self.hist_stat_len
            x_patches = x.reshape(B, P_hist, self.window_len, C)
            mu_hist   = x_patches.mean(dim=2)
            std_hist  = x_patches.std(dim=2).clamp(min=self.sigma_min)
            self._mu_hist  = mu_hist
            self._std_hist = std_hist
            anchor = mu_hist.mean(dim=1, keepdim=True)

            # ----------------------------------------------------------
            # Ablation: PureSANSeqPredictor architecture with std
            # normalize/denorm restored.
            # ----------------------------------------------------------
            if self.san_ablation_mode == "seq_std":
                if not isinstance(self.predictor, _PureSANSeqPredictor):
                    raise RuntimeError(
                        "seq_std ablation must use _PureSANSeqPredictor"
                    )
                z_patches = (
                    (x_patches - mu_hist.unsqueeze(2))
                    / (std_hist.unsqueeze(2) + self.epsilon)
                )
                z_out = z_patches.reshape(B, T, C)
                self._hist_last_patch_norm = None

                mu_fut_hat, std_fut_hat = self.predictor.predict_mean_std_seq(
                    mu_hist=mu_hist,
                    std_hist=std_hist,
                    anchor=anchor,
                    raw_patches=x_patches.detach(),
                    z_patches=z_patches.detach(),
                    eps=self.epsilon,
                )
                self._base_mu_fut_hat  = mu_fut_hat
                self._base_std_fut_hat = std_fut_hat
                self._pred_time_stats  = self._window_stats_to_time_stats(
                    mu_fut_hat, std_fut_hat, self.pred_len
                )
                bts_m, bts_s = self._split_stats(self._pred_time_stats)
                self._base_time_mean = bts_m
                self._base_time_std  = bts_s
                return z_out

            # ----------------------------------------------------------
            # Ablation: original SAN predictor architecture, mean-only
            # normalize/denorm.  std is computed for predictor input but
            # ignored by recovery and aux loss.
            # ----------------------------------------------------------
            if self.san_ablation_mode == "base_mean_only":
                if not isinstance(self.predictor, _SANBasePredictor):
                    raise RuntimeError(
                        "base_mean_only ablation must use _SANBasePredictor"
                    )
                z_patches = x_patches - mu_hist.unsqueeze(2)
                z_out = z_patches.reshape(B, T, C)
                self._hist_last_patch_norm = None

                xbar_raw = x_patches.reshape(B, -1, C).detach()
                mu_fut_hat, _std_unused = self.predictor.predict(
                    mu_hist=mu_hist,
                    std_hist=std_hist,
                    xbar=None,
                    anchor=anchor,
                    xbar_raw=xbar_raw,
                )
                self._base_mu_fut_hat  = mu_fut_hat
                self._base_std_fut_hat = None
                std_compat = std_hist.mean(dim=1, keepdim=True).expand(
                    -1, self.pred_stat_len, -1
                ).contiguous()
                self._pred_time_stats  = self._window_stats_to_time_stats(
                    mu_fut_hat, std_compat, self.pred_len
                )
                bts_m, bts_s = self._split_stats(self._pred_time_stats)
                self._base_time_mean = bts_m
                self._base_time_std  = bts_s
                return z_out

            # ----------------------------------------------------------
            # Pure baseline + phase: mean-normalization only
            # Backbone input space is mean-normalized, NOT std-normalized.
            # ----------------------------------------------------------
            if self.use_phase:
                if self.phase_energy_gate or self.phase_energy_mean_gate:
                    period_len = self.window_len
                    x_ref = x[:, -period_len:, :]                    # (B, W, C)
                    z_ref = x_ref - x_ref.mean(dim=1, keepdim=True)
                    X_ref = torch.fft.rfft(z_ref, dim=1)             # (B, W//2+1, C)
                    max_bin = period_len // 2
                    low_k   = self._phase_k
                    if low_k + 1 <= max_bin:
                        E_high  = (X_ref[:, low_k + 1:max_bin + 1, :].abs() ** 2).sum(dim=1)
                        E_total = (X_ref[:, 1:max_bin + 1, :].abs() ** 2).sum(dim=1) + self.phase_energy_gate_eps
                        r_high  = E_high / E_total                   # (B, C)
                        s0_val  = -self.phase_energy_gate_center / self.phase_energy_gate_temp
                        s0 = torch.sigmoid(
                            torch.tensor(s0_val, device=x.device, dtype=x.dtype)
                        )
                        s    = torch.sigmoid((r_high - self.phase_energy_gate_center) / self.phase_energy_gate_temp)
                        risk = (s - s0) / (1.0 - s0 + self.phase_energy_gate_eps)
                        risk = risk.clamp(0.0, 1.0)
                        gate = (1.0 - self.phase_energy_gate_strength * risk).clamp(
                            self.phase_energy_gate_min, 1.0
                        )
                    else:
                        r_high = torch.zeros(B, C, device=x.device, dtype=x.dtype)
                        gate   = torch.ones(B, C, device=x.device, dtype=x.dtype)
                        risk   = torch.zeros(B, C, device=x.device, dtype=x.dtype)
                    self._phase_energy_gate   = gate
                    self._phase_energy_r_high = r_high
                    self._phase_energy_risk   = risk

                z_patches = x_patches - mu_hist.unsqueeze(2)          # (B, P_hist, W, C)
                self._last_input_residual_rms = float(
                    torch.sqrt(z_patches.detach().pow(2).mean() + self.epsilon).item()
                )
                z_out     = z_patches.reshape(B, T, C)
                # Cache last mean-normalized patch for phase loss supervision
                self._hist_last_patch_norm = z_patches[:, -1, :, :].detach()   # (B, W, C)

                mu_fut_hat, phase_rot_fut_hat = self.predictor.predict_mean_phase(
                    mu_hist=mu_hist,
                    anchor=anchor,
                    raw_patches=x_patches.detach(),
                    z_patches=z_patches.detach(),
                    eps=self.epsilon,
                )
                self._base_mu_fut_hat   = mu_fut_hat
                self._phase_rot_fut_hat = phase_rot_fut_hat
                # compat only — never used for denorm/loss in pure phase mode
                std_compat = std_hist.mean(dim=1, keepdim=True).expand(
                    -1, self.pred_stat_len, -1
                ).contiguous()
                self._base_std_fut_hat = std_compat
                self._pred_time_stats  = self._window_stats_to_time_stats(
                    mu_fut_hat, std_compat, self.pred_len
                )
                bts_m, bts_s = self._split_stats(self._pred_time_stats)
                self._base_time_mean = bts_m
                self._base_time_std  = bts_s
                return z_out

            # ----------------------------------------------------------
            # Pure SAN (use_phase=False): exact original z = (x-mu)/std
            # ----------------------------------------------------------
            if not isinstance(self.predictor, _SANBasePredictor):
                raise RuntimeError(
                    "pure SAN without phase must use _SANBasePredictor"
                )
            z_patches = (
                (x_patches - mu_hist.unsqueeze(2))
                / (std_hist.unsqueeze(2) + self.epsilon)
            )  # (B, P_hist, W, C)
            z_san = z_patches.reshape(B, T, C)
            self._hist_last_patch_norm = None

            xbar_raw = x_patches.reshape(B, -1, C).detach()
            mu_base_fut, std_base_fut = self.predictor.predict(
                mu_hist=mu_hist,
                std_hist=std_hist,
                xbar=None,
                anchor=anchor,
                xbar_raw=xbar_raw,
            )
            self._base_mu_fut_hat  = mu_base_fut
            self._base_std_fut_hat = std_base_fut
            self._pred_time_stats  = self._window_stats_to_time_stats(
                mu_base_fut, std_base_fut, self.pred_len
            )
            bts_m, bts_s = self._split_stats(self._pred_time_stats)
            self._base_time_mean = bts_m
            self._base_time_std  = bts_s
            return z_san

        # ==============================================================
        # Generalised overlap-add path (lp_state + generic routes)
        # ==============================================================
        hist_windows = self._extract_windows(x.detach())
        mu_hist, std_hist = self._compute_window_stats(hist_windows)
        self._mu_hist  = mu_hist
        self._std_hist = std_hist

        hist_time_stats     = self._window_stats_to_time_stats(mu_hist, std_hist, T)
        hist_time_mean, hist_time_std = self._split_stats(hist_time_stats)
        z = (x - hist_time_mean) / (hist_time_std + self.epsilon)

        norm_windows = self._extract_windows(z.detach())
        xbar = norm_windows.reshape(B, -1, C)

        anchor = mu_hist.mean(dim=1, keepdim=True)
        mu_base_fut, std_base_fut = self.predictor.predict(
            mu_hist, std_hist, xbar, anchor
        )
        self._base_mu_fut_hat  = mu_base_fut
        self._base_std_fut_hat = std_base_fut
        self._pred_time_stats  = self._window_stats_to_time_stats(
            mu_base_fut, std_base_fut, self.pred_len
        )
        bts_m, bts_s = self._split_stats(self._pred_time_stats)
        self._base_time_mean = bts_m
        self._base_time_std  = bts_s

        if self.route_path == "lp_state_correction":
            mu_hist_lp = self._lowpass_time_to_patch_mean(x.detach())
            self._lp_mu_hist    = mu_hist_lp
            self._lp_mu_fut_hat = self.lp_state_predictor(mu_hist_lp)

        elif self.route_path != "none":
            should_normalize = self._should_normalize_route_state()
            hist_state_raw = self.route_state_impl.extract_hist_state(
                x.detach(), hist_windows, mu_hist, std_hist, self.sigma_min
            )
            self._route_hist_state_raw = hist_state_raw
            hist_state = (
                _normalize_patch_state(hist_state_raw) if should_normalize
                else hist_state_raw
            )
            self._route_hist_state = hist_state
            fut_hat_raw = self.route_state_predictor(hist_state, xbar)
            fut_hat_adapted = self.route_state_impl.adapt_future_state(fut_hat_raw)
            self._route_future_state_hat_raw = fut_hat_adapted
            future_state_hat = (
                _normalize_patch_state(fut_hat_adapted) if should_normalize
                else fut_hat_adapted
            )
            self._route_future_state_hat  = future_state_hat
            self._route_future_state_time = self._patch_features_to_time(
                future_state_hat, self.pred_len
            )

        return z

    # ------------------------------------------------------------------
    # Main interface: denormalize
    # ------------------------------------------------------------------

    def denormalize(
        self,
        y_norm: torch.Tensor,
        y_true_oracle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Restore y_norm to original scale.

        timeapn pair:
            1. Compute seq_ms_y from y_norm via norm_sliding (needed for phase loss).
            2. Build station_pred_raw  = cat([pred_m, pred_s], dim=-1)  → raw predicted station.
               Build station_pred_loss = cat([pred_m, seq_s_y + pred_s], dim=-1) → phase-supervised.
            3. de_normalize uses station_pred_raw (official semantics: raw pred_s for phase compensation).
            4. Cache y_out, station_pred_raw, station_pred_loss for loss.
        pure SAN:
            Exact reshape-based denorm.
        Others:
            Overlap-add denorm + optional route_path_impl.
        """
        T = y_norm.shape[1]

        # ==============================================================
        # Official TimeAPN denorm
        # ==============================================================
        if self.route_path == "timeapn_correction":
            if self.apn_module is None or self._apn_pred_ms is None:
                return y_norm
            pred_m, pred_s = self._apn_pred_ms  # (B, C, pred_len) each

            # seq_ms of model output — needed for phase supervision in loss
            y_BCT = y_norm.transpose(-1, -2)  # (B, C, pred_len)
            _, (seq_m_y, seq_s_y) = self.apn_module.norm_sliding(y_BCT)
            # seq_s_y: (B, C, pred_len) — phase of model output

            # Raw predicted station tensor: only pred_m and pred_s
            # This is the official APN semantics: pred_s is used for phase compensation
            station_pred_raw = self.apn_module.build_station_pred_tensor(
                pred_m, pred_s
            )  # (B, pred_len, 2C)

            # Loss station tensor: combined phase = seq_s_y + pred_s for phase supervision
            station_pred_loss = torch.cat(
                [pred_m, seq_s_y + pred_s], dim=1
            ).transpose(-1, -2)   # (B, pred_len, 2C)

            # de_normalize with raw predicted station (official semantics)
            y_out = self.apn_module.de_normalize(y_norm, station_pred_raw)
            mean_time = pred_m.transpose(-1, -2)
            y_conv_residual = y_out - mean_time
            y_norm_rms = torch.sqrt(y_norm.detach().pow(2).mean() + self.epsilon)
            apn_conv_effect = torch.sqrt(
                (y_conv_residual.detach() - y_norm.detach()).pow(2).mean()
                + self.epsilon
            )
            self._last_apn_conv_effect_rms = float(apn_conv_effect.detach().item())
            self._last_apn_conv_effect_ratio = float(
                (apn_conv_effect / (y_norm_rms + self.epsilon)).detach().item()
            )
            self._last_apn_pred_residual_rms = float(
                torch.sqrt(y_conv_residual.detach().pow(2).mean() + self.epsilon).item()
            )

            half = station_pred_raw.shape[-1] // 2
            pred_phase = station_pred_raw[..., half:]  # (B, pred_len, C)
            kernel_raw = torch.fft.ifft(
                torch.exp(1j * pred_phase.transpose(-1, -2)), dim=-1
            ).transpose(-1, -2)
            kernel = torch.abs(kernel_raw)
            kernel_sum = kernel.sum(dim=1)
            kernel_l2 = torch.sqrt(kernel.pow(2).sum(dim=1))
            kernel_max = kernel.max(dim=1).values
            center = kernel.shape[1] // 2
            kernel_center = kernel[:, center, :]
            p_kernel = kernel / (kernel_sum.unsqueeze(1) + self.epsilon)
            kernel_entropy = -(p_kernel * torch.log(p_kernel + self.epsilon)).sum(dim=1)
            self._last_apn_kernel_l1 = float(kernel_sum.detach().mean().item())
            self._last_apn_kernel_l2 = float(kernel_l2.detach().mean().item())
            self._last_apn_kernel_max_ratio = float(
                (kernel_max / (kernel_sum + self.epsilon)).detach().mean().item()
            )
            self._last_apn_kernel_center_ratio = float(
                (kernel_center / (kernel_sum + self.epsilon)).detach().mean().item()
            )
            self._last_apn_kernel_entropy = float(kernel_entropy.detach().mean().item())

            # Cache for loss
            self._apn_station_pred_raw  = station_pred_raw
            self._apn_station_pred_loss = station_pred_loss
            self._apn_y_out             = y_out
            return y_out

        if self._pred_time_stats is None:
            return y_norm

        # ==============================================================
        # Pure SAN exact reshape-based denorm
        # ==============================================================
        if self.route_path == "none":
            if self.san_ablation_mode == "seq_std":
                B, _, C = y_norm.shape
                P_fut = self.pred_stat_len
                W = self.window_len
                mu_fut = self._base_mu_fut_hat
                std_fut = self._base_std_fut_hat
                if mu_fut is None or std_fut is None:
                    return y_norm
                if T != self.pred_len:
                    ts = self._window_stats_to_time_stats(mu_fut, std_fut, T)
                    mu, sig = self._split_stats(ts)
                    return (y_norm * (sig + self.epsilon) + mu).contiguous()
                y_patches = y_norm.reshape(B, P_fut, W, C)
                y_residual = y_patches * (std_fut.unsqueeze(2) + self.epsilon)
                self._last_pred_residual_rms_after_phase = float(
                    torch.sqrt(y_residual.detach().pow(2).mean() + self.epsilon).item()
                )
                y_out = y_residual + mu_fut.unsqueeze(2)
                return y_out.reshape(B, self.pred_len, C).contiguous()

            if self.san_ablation_mode == "base_mean_only":
                B, _, C = y_norm.shape
                P_fut = self.pred_stat_len
                W = self.window_len
                mu_fut = self._base_mu_fut_hat
                if mu_fut is None:
                    return y_norm
                if T != self.pred_len:
                    mu_time = self._patch_features_to_time(mu_fut, T)
                    return (y_norm + mu_time).contiguous()
                y_patches = y_norm.reshape(B, P_fut, W, C)
                self._last_pred_residual_rms_after_phase = float(
                    torch.sqrt(y_patches.detach().pow(2).mean() + self.epsilon).item()
                )
                y_out = y_patches + mu_fut.unsqueeze(2)
                return y_out.reshape(B, self.pred_len, C).contiguous()

            # ----------------------------------------------------------
            # Pure baseline + phase: mean + phase main chain.
            # pred_std is NOT used for output recovery here.
            # ----------------------------------------------------------
            if self.use_phase and self._phase_rot_fut_hat is not None:
                B, _, C = y_norm.shape
                P_fut   = self.pred_stat_len
                W       = self.window_len
                K       = self._phase_k
                mu_fut  = self._base_mu_fut_hat          # (B, P_fut, C)

                if self.phase_energy_mean_gate and self._phase_energy_risk is not None:
                    risk      = self._phase_energy_risk              # (B, C)
                    w         = (self.phase_energy_mean_strength * risk[:, None, :]).clamp(0.0, 1.0)
                    mu_anchor = mu_fut.mean(dim=1, keepdim=True)     # (B, 1, C)
                    mu_fut    = (1.0 - w) * mu_fut + w * mu_anchor

                if T != self.pred_len:
                    # Fallback for non-standard T (e.g. validation with different length):
                    # broadcast mean over time and add to y_norm (std=1 in mean-norm space).
                    mu_time = self._patch_features_to_time(mu_fut, T)  # (B, T, C)
                    return (mu_time + y_norm).contiguous()

                y_patches = y_norm.reshape(B, P_fut, W, C)            # (B, P_fut, W, C)

                patch_mean = y_patches.mean(dim=2, keepdim=True)       # (B, P_fut, 1, C)
                self._last_phase_output_mean_abs = float(
                    patch_mean.detach().abs().mean().item()
                )
                self._last_output_patch_mean_abs_before_gauge = float(
                    patch_mean.detach().abs().mean().item()
                )
                self._last_output_rms_before_gauge = float(
                    torch.sqrt(y_patches.detach().pow(2).mean() + self.epsilon).item()
                )
                self._last_dc_gauge_ratio = (
                    self._last_output_patch_mean_abs_before_gauge
                    / (self._last_output_rms_before_gauge + self.epsilon)
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

                # Apply predicted phase rotation in frequency domain
                BP     = B * P_fut
                y_flat = y_patches.reshape(BP, W, C)
                Y_flat = torch.fft.rfft(y_flat, dim=1)                # (BP, W//2+1, C)

                K_eff     = min(K, Y_flat.shape[1] - 1)               # skip DC bin 0
                phase_bp  = self._phase_rot_fut_hat.reshape(BP, K, C, 2)
                a = phase_bp[:, :K_eff, :, 0]                         # (BP, K_eff, C)
                b = phase_bp[:, :K_eff, :, 1]
                r_re   = 1.0 + a
                r_im   = b
                norm_r = torch.sqrt(r_re.pow(2) + r_im.pow(2) + self.epsilon)
                r      = torch.complex(r_re / norm_r, r_im / norm_r)  # (BP, K_eff, C)

                Y_tilde = Y_flat.clone()
                if self.phase_energy_gate and self._phase_energy_gate is not None:
                    gate      = self._phase_energy_gate                         # (B, C)
                    angle_hat = torch.atan2(r.imag, r.real).reshape(B, P_fut, K_eff, C)
                    angle_hat = angle_hat * gate[:, None, None, :]
                    self._last_phase_angle_abs_mean = (
                        float(angle_hat.detach().abs().mean().item())
                        if K_eff > 0 else 0.0
                    )
                    r_gated   = torch.polar(
                        torch.ones_like(angle_hat), angle_hat
                    ).reshape(BP, K_eff, C)
                    Y_tilde[:, 1:K_eff + 1, :] = Y_flat[:, 1:K_eff + 1, :] * r_gated
                else:
                    angle = torch.atan2(r.imag, r.real)
                    self._last_phase_angle_abs_mean = (
                        float(angle.detach().abs().mean().item())
                        if K_eff > 0 else 0.0
                    )
                    Y_tilde[:, 1:K_eff + 1, :] = Y_flat[:, 1:K_eff + 1, :] * r
                y_phase = torch.fft.irfft(Y_tilde, n=W, dim=1)        # (BP, W, C)
                y_phase = y_phase.reshape(B, P_fut, W, C)
                self._last_phase_effect_rms = float(
                    torch.sqrt(
                        (y_phase.detach() - y_before_phase).pow(2).mean()
                        + self.epsilon
                    ).item()
                )
                self._last_phase_effect_ratio = (
                    self._last_phase_effect_rms
                    / (self._last_output_rms_before_gauge + self.epsilon)
                )
                self._last_pred_residual_rms_after_phase = float(
                    torch.sqrt(y_phase.detach().pow(2).mean() + self.epsilon).item()
                )

                # Restore mean
                y_out = (y_phase + mu_fut.unsqueeze(2)).reshape(B, self.pred_len, C)
                return y_out.contiguous()

            # ----------------------------------------------------------
            # Pure SAN (use_phase=False): original std-based denorm.
            # SSC / B2SC only apply to the std-based path.
            # ----------------------------------------------------------
            mu_fut_cal, std_fut_cal = self._apply_ssc_to_patch_stats(
                self._base_mu_fut_hat, self._base_std_fut_hat
            )
            mu_fut_cal, std_fut_cal = self._apply_b2sc_to_patch_stats(
                mu_fut_cal, std_fut_cal
            )
            if T != self.pred_len:
                ts = self._window_stats_to_time_stats(mu_fut_cal, std_fut_cal, T)
                mu, sig = self._split_stats(ts)
                return (mu + (sig + self.epsilon) * y_norm).contiguous()

            B, _, C = y_norm.shape
            P_fut     = self.pred_stat_len
            y_patches = y_norm.reshape(B, P_fut, self.window_len, C)
            y_san = (
                y_patches * (std_fut_cal.unsqueeze(2) + self.epsilon)
                + mu_fut_cal.unsqueeze(2)
            ).reshape(B, self.pred_len, C)
            return y_san.contiguous()

        # ==============================================================
        # lp_state_correction
        # ==============================================================
        if (
            self.route_path == "lp_state_correction"
            and self._lp_mu_fut_hat is not None
            and self._base_std_fut_hat is not None
            and T == self.pred_len
        ):
            _, std_fut_cal = self._apply_b2sc_to_patch_stats(
                self._base_mu_fut_hat, self._base_std_fut_hat
            )
            lp_mu_cal = self._lp_mu_fut_hat.clone()
            if self._last_b2sc_applied and self._last_b2sc_delta_mu is not None:
                lp_mu_cal[:, 0, :] += self._last_b2sc_delta_mu
                if lp_mu_cal.shape[1] >= 2:
                    lp_mu_cal[:, 1, :] += (
                        self.b2sc_second_slice_scale * self._last_b2sc_delta_mu
                    )
            lp_ts = self._window_stats_to_time_stats(lp_mu_cal, std_fut_cal, T)
            mu_lp, sig_base = self._split_stats(lp_ts)
            return (mu_lp + (sig_base + self.epsilon) * y_norm).contiguous()

        # ==============================================================
        # Generic overlap-add + route_path_impl
        # ==============================================================
        mu_fut_cal, std_fut_cal = self._apply_b2sc_to_patch_stats(
            self._base_mu_fut_hat, self._base_std_fut_hat
        )
        time_stats_cal = self._window_stats_to_time_stats(mu_fut_cal, std_fut_cal, T)
        base_time_mean_cal, base_time_std_cal = self._split_stats(time_stats_cal)
        y_base = base_time_mean_cal + (base_time_std_cal + self.epsilon) * y_norm

        if (
            self.route_path not in ("none", "lp_state_correction")
            and self._route_future_state_hat is not None
            and self._route_future_state_time is not None
            and self._base_time_mean is not None
            and T == self.pred_len
        ):
            y_base = self.route_path_impl(
                y_norm=y_norm,
                y_base=y_base,
                future_state_hat=self._route_future_state_hat,
                future_state_time=self._route_future_state_time,
                mu_base_fut=mu_fut_cal,
                std_base_fut=std_fut_cal,
                base_time_mean=base_time_mean_cal,
                base_time_std=base_time_std_cal,
            )
        return y_base

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_oracle_stats(
        self, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        true_windows = self._extract_windows(y_true)
        oracle_mu, oracle_std = self._compute_window_stats(true_windows)
        return true_windows, oracle_mu, oracle_std

    # ------------------------------------------------------------------
    # Phase correction (pure baseline only)
    # ------------------------------------------------------------------

    def _apply_phase_correction(
        self, y_patches: torch.Tensor
    ) -> torch.Tensor:
        """Apply cached phase rotation to future patches.

        Phase prediction is completed in normalize() and stored in
        ``self._phase_rot_fut_hat``.  This method only reads from that cache
        and applies the rotation — no re-forward of the phase predictor.

        Args:
            y_patches: (B, P_fut, W, C) — mean-normalized backbone output
                       reshaped into future patches.
        Returns:
            y_patches_phase: same shape (B, P_fut, W, C).
        """
        if self._phase_rot_fut_hat is None:
            return y_patches

        B, P_fut, W, C = y_patches.shape
        K      = self._phase_k
        BP     = B * P_fut
        y_flat = y_patches.reshape(BP, W, C)
        Y_flat = torch.fft.rfft(y_flat, dim=1)          # (BP, W//2+1, C)

        K_eff    = min(K, Y_flat.shape[1] - 1)
        phase_bp = self._phase_rot_fut_hat.reshape(BP, K, C, 2)
        a = phase_bp[:, :K_eff, :, 0]
        b = phase_bp[:, :K_eff, :, 1]
        r_re   = 1.0 + a
        r_im   = b
        norm_r = torch.sqrt(r_re.pow(2) + r_im.pow(2) + self.epsilon)
        r      = torch.complex(r_re / norm_r, r_im / norm_r)   # (BP, K_eff, C)

        Y_tilde = Y_flat.clone()
        Y_tilde[:, 1:K_eff + 1, :] = Y_flat[:, 1:K_eff + 1, :] * r
        y_tilde = torch.fft.irfft(Y_tilde, n=W, dim=1)          # (BP, W, C)
        return y_tilde.reshape(B, P_fut, W, C)

    def _compute_phase_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Pure-baseline phase supervision loss (mean + phase main chain).

        Target: oracle phase drift from the last historical mean-normalized
        patch to each oracle future mean-normalized patch, measured on
        low-frequency bins 1..K.

        Supervision is applied to the cached ``_phase_rot_fut_hat`` that was
        already computed in ``normalize()``.  No re-forward of the phase
        predictor here.

        Returns 0 if any required precondition is unmet.
        """
        dev  = y_true.device
        zero = torch.tensor(0.0, device=dev)
        if (
            not self.use_phase
            or self.route_path != "none"
            or self.route_state != "none"
            or self._hist_last_patch_norm is None
            or self._phase_rot_fut_hat is None
        ):
            return zero

        B     = y_true.shape[0]
        P_fut = self.pred_stat_len
        W     = self.window_len
        K     = self._phase_k
        C     = y_true.shape[-1]

        # --- oracle future mean-normalized patches (no std division) ---
        y_patches  = y_true.reshape(B, P_fut, W, C)               # (B, P_fut, W, C)
        oracle_mu  = y_patches.mean(dim=2)                         # (B, P_fut, C)
        y_oracle_mn = y_patches - oracle_mu.unsqueeze(2)           # (B, P_fut, W, C)

        # --- rfft of hist last mean-normalized patch and oracle future patches ---
        Y_hist   = torch.fft.rfft(self._hist_last_patch_norm, dim=1)  # (B, W//2+1, C)
        BP       = B * P_fut
        Y_oracle = torch.fft.rfft(
            y_oracle_mn.reshape(BP, W, C), dim=1
        )  # (BP, W//2+1, C)

        n_bins = Y_hist.shape[1]
        K_eff  = min(K, n_bins - 1)   # exclude DC bin 0

        Y_h = Y_hist[:, 1:K_eff + 1, :]                           # (B,  K_eff, C)
        Y_o = Y_oracle[:, 1:K_eff + 1, :]                         # (BP, K_eff, C)

        # Oracle phase drift = Y_oracle * conj(Y_hist), normalised to unit complex
        Y_h_rep   = Y_h.unsqueeze(1).expand(-1, P_fut, -1, -1).reshape(BP, K_eff, C)
        drift_raw = Y_o * Y_h_rep.conj()                           # (BP, K_eff, C)
        drift_abs = drift_raw.abs().clamp(min=self.epsilon)
        R_target  = drift_raw / drift_abs                          # unit complex

        # --- predicted rotation: read from cache, do NOT re-run predictor ---
        phase_bp = self._phase_rot_fut_hat.reshape(BP, K, C, 2)
        a = phase_bp[:, :K_eff, :, 0]                             # (BP, K_eff, C)
        b = phase_bp[:, :K_eff, :, 1]
        r_re   = 1.0 + a
        r_im   = b
        norm_r = torch.sqrt(r_re.pow(2) + r_im.pow(2) + self.epsilon)
        R_pred = torch.complex(r_re / norm_r, r_im / norm_r)      # (BP, K_eff, C)

        # MSE on unit-complex plane (re and im separately)
        loss_re = F.mse_loss(R_pred.real, R_target.real.detach())
        loss_im = F.mse_loss(R_pred.imag, R_target.imag.detach())
        return loss_re + loss_im

    def compute_base_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Base SAN stats loss (Stage 1).  Not used for timeapn pair."""
        if self.route_path == "timeapn_correction":
            return torch.tensor(0.0, device=y_true.device)

        if self._base_mu_fut_hat is None:
            return torch.tensor(0.0, device=y_true.device)
        if self.san_ablation_mode != "base_mean_only" and self._base_std_fut_hat is None:
            return torch.tensor(0.0, device=y_true.device)

        true_windows, oracle_mu, oracle_std = self._compute_oracle_stats(y_true)
        if self.san_ablation_mode in ("seq_std", "base_mean_only"):
            B, _, C = y_true.shape
            true_windows = y_true.reshape(B, self.pred_stat_len, self.window_len, C)
            oracle_mu = true_windows.mean(dim=2)
            oracle_std = true_windows.std(dim=2).clamp_min(self.sigma_min)
            oracle_residual = true_windows - oracle_mu.unsqueeze(2)
            oracle_rms = torch.sqrt(oracle_residual.pow(2).mean() + self.epsilon)
            self._last_oracle_residual_rms = float(oracle_rms.detach().item())
            if self._last_pred_residual_rms_after_phase > 0:
                self._last_residual_rms_ratio = (
                    self._last_pred_residual_rms_after_phase
                    / (self._last_oracle_residual_rms + self.epsilon)
                )

        if self.use_phase and self.route_path == "none":
            oracle_residual = true_windows - oracle_mu.unsqueeze(2)
            oracle_rms = torch.sqrt(oracle_residual.pow(2).mean() + self.epsilon)
            self._last_oracle_residual_rms = float(oracle_rms.detach().item())
            if self._last_pred_residual_rms_after_phase > 0:
                self._last_residual_rms_ratio = (
                    self._last_pred_residual_rms_after_phase
                    / (self._last_oracle_residual_rms + self.epsilon)
                )

        if self.san_ablation_mode == "seq_std":
            mu_loss  = F.mse_loss(self._base_mu_fut_hat, oracle_mu)
            std_loss = F.mse_loss(self._base_std_fut_hat, oracle_std)
            base_aux = self.w_mu * mu_loss + self.w_std * std_loss
        elif self.san_ablation_mode == "base_mean_only":
            mu_loss  = F.mse_loss(self._base_mu_fut_hat, oracle_mu)
            std_loss = torch.tensor(0.0, device=y_true.device)
            base_aux = self.w_mu * mu_loss
        elif self.route_path == "lp_state_correction":
            mu_loss  = torch.tensor(0.0, device=y_true.device)
            std_loss = F.mse_loss(self._base_std_fut_hat, oracle_std)
            base_aux = self.w_std * std_loss
        elif self.use_phase:
            # Pure baseline + phase: main chain is mean + phase.
            # std is not used for output recovery, so its loss is excluded
            # from base_aux.  phase_loss is added separately below.
            mu_loss  = F.mse_loss(self._base_mu_fut_hat, oracle_mu)
            std_loss = torch.tensor(0.0, device=y_true.device)
            base_aux = self.w_mu * mu_loss
        else:
            mu_loss  = F.mse_loss(self._base_mu_fut_hat, oracle_mu)
            std_loss = F.mse_loss(self._base_std_fut_hat, oracle_std)
            base_aux = self.w_mu * mu_loss + self.w_std * std_loss

        self._last_mu_loss       = float(mu_loss.detach().item())
        self._last_std_loss      = float(std_loss.detach().item())
        self._last_base_aux_loss = float(base_aux.detach().item())

        # Pure baseline phase aux (returns 0 when use_phase=False)
        if self.san_ablation_mode != "none":
            phase_loss = torch.tensor(0.0, device=y_true.device)
        else:
            phase_loss = self._compute_phase_aux_loss(y_true)
        self._last_phase_loss = float(phase_loss.detach().item())
        total = base_aux + self._phase_loss_weight * phase_loss

        self._last_aux_total = float(total.detach().item())
        return total

    def compute_route_state_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Route state loss.

        timeapn pair: implements official sliding_loss_P.
        lp pair: patch MSE.
        Others: generic route state MSE.
        Returns 0 when route_path='none'.
        """
        if self.route_path == "none":
            return torch.tensor(0.0, device=y_true.device)

        # ------------------------------------------------------------------
        # Official TimeAPN sliding_loss_P
        # ------------------------------------------------------------------
        if self.route_path == "timeapn_correction":
            if self.apn_module is None or self._apn_pred_ms is None:
                return torch.tensor(0.0, device=y_true.device)

            pred_m, pred_s = self._apn_pred_ms  # (B, C, pred_len)

            # Ground-truth mean and phase from y_true
            y_true_BCT = y_true.transpose(-1, -2)  # (B, C, pred_len)
            _, (true_m, true_phi) = self.apn_module.norm_sliding(y_true_BCT)
            # true_m, true_phi: (B, C, pred_len)

            # station_pred1 was built in denormalize as (B, pred_len, 2C)
            if self._apn_station_pred_loss is None:
                return torch.tensor(0.0, device=y_true.device)

            half = self._apn_station_pred_loss.shape[-1] // 2
            pred_mean_flat = self._apn_station_pred_loss[..., :half]  # (B, pred_len, C)
            combined_phase = self._apn_station_pred_loss[..., half:]  # (B, pred_len, C)  seq_s_y + pred_s

            # station_ture for y_true: cat([true_m, true_phi]) → (B, pred_len, 2C)
            true_mean_flat  = true_m.transpose(-1, -2)   # (B, pred_len, C)
            true_phase_flat = true_phi.transpose(-1, -2)  # (B, pred_len, C)
            oracle_residual = y_true - true_mean_flat
            oracle_rms = torch.sqrt(oracle_residual.detach().pow(2).mean() + self.epsilon)
            self._last_apn_oracle_residual_rms = float(oracle_rms.detach().item())
            if self._last_apn_pred_residual_rms > 0:
                self._last_apn_residual_rms_ratio = (
                    self._last_apn_pred_residual_rms
                    / (self._last_apn_oracle_residual_rms + self.epsilon)
                )

            loss_mean  = F.mse_loss(pred_mean_flat,  true_mean_flat)
            loss_phase = F.mse_loss(combined_phase,  true_phase_flat)
            phase_delta = torch.atan2(
                torch.sin(combined_phase.detach() - true_phase_flat.detach()),
                torch.cos(combined_phase.detach() - true_phase_flat.detach()),
            )
            self._last_apn_phase_abs_err = float(phase_delta.abs().mean().item())
            self._last_apn_phase_cos = float(torch.cos(phase_delta).mean().item())

            # Reconstruction loss: y_out vs y_true
            if self._apn_y_out is not None:
                loss_recon = F.mse_loss(self._apn_y_out, y_true)
            else:
                loss_recon = torch.tensor(0.0, device=y_true.device)

            self._last_apn_loss_mean  = float(loss_mean.detach().item())
            self._last_apn_loss_phase = float(loss_phase.detach().item())
            self._last_apn_loss_recon = float(loss_recon.detach().item())

            total = loss_recon + loss_mean + loss_phase
            self._last_route_state_loss = float(total.detach().item())
            return total

        # ------------------------------------------------------------------
        # lp_state
        # ------------------------------------------------------------------
        if self.route_state == "lp_state":
            if self._lp_mu_fut_hat is None:
                return torch.tensor(0.0, device=y_true.device)
            oracle_lp_mu_fut = self._lowpass_time_to_patch_mean(y_true)
            self._oracle_lp_mu_fut = oracle_lp_mu_fut
            lp_loss = F.mse_loss(self._lp_mu_fut_hat, oracle_lp_mu_fut)
            self._last_route_state_loss = float(lp_loss.detach().item())
            return lp_loss

        # ------------------------------------------------------------------
        # Generic route state
        # ------------------------------------------------------------------
        if self._route_future_state_hat is None:
            return torch.tensor(0.0, device=y_true.device)

        true_windows, oracle_mu, oracle_std = self._compute_oracle_stats(y_true)
        oracle_raw = self.route_state_impl.build_future_oracle_state(
            y_true, true_windows, oracle_mu, oracle_std, self.sigma_min
        )
        should_normalize = self._should_normalize_route_state()
        oracle_state = (
            _normalize_patch_state(oracle_raw) if should_normalize else oracle_raw
        )
        self._route_future_oracle_state = oracle_state
        route_state_loss = F.mse_loss(self._route_future_state_hat, oracle_state)
        self._last_route_state_loss = float(route_state_loss.detach().item())
        return route_state_loss

    def compute_total_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Combined base aux + weighted route state loss."""
        if self.route_path == "lp_state_correction":
            route_loss = self.compute_route_state_loss(y_true)
            base_aux   = self.compute_base_aux_loss(y_true)
        else:
            base_aux   = self.compute_base_aux_loss(y_true)
            route_loss = self.compute_route_state_loss(y_true)
        total = base_aux + self.route_state_loss_scale * route_loss
        self._last_aux_total = float(total.detach().item())
        return total

    def loss(self, y_true: torch.Tensor) -> torch.Tensor:
        return self.compute_total_aux_loss(y_true)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_last_aux_stats(self) -> dict:
        return {
            "san_ablation_mode": self.san_ablation_mode,
            "aux_total":         self._last_aux_total,
            "base_aux_loss":     self._last_base_aux_loss,
            "mu_loss":           self._last_mu_loss,
            "std_loss":          self._last_std_loss,
            "phase_loss":        self._last_phase_loss,
            "route_state_loss":  self._last_route_state_loss,
            # TimeAPN decomposition
            "apn_loss_mean":     self._last_apn_loss_mean,
            "apn_loss_phase":    self._last_apn_loss_phase,
            "apn_loss_recon":    self._last_apn_loss_recon,
            # SAN stats
            "mu_hist_mean": (
                float(self._mu_hist.mean().item()) if self._mu_hist is not None else 0.0
            ),
            "std_hist_mean": (
                float(self._std_hist.mean().item()) if self._std_hist is not None else 0.0
            ),
            "base_mu_fut_mean": (
                float(self._base_mu_fut_hat.mean().item())
                if self._base_mu_fut_hat is not None else 0.0
            ),
            "base_std_fut_mean": (
                float(self._base_std_fut_hat.mean().item())
                if self._base_std_fut_hat is not None else 0.0
            ),
            # lp fields
            "lp_mu_hist_mean": (
                float(self._lp_mu_hist.mean().item())
                if self._lp_mu_hist is not None else 0.0
            ),
            "lp_mu_fut_hat_mean": (
                float(self._lp_mu_fut_hat.mean().item())
                if self._lp_mu_fut_hat is not None else 0.0
            ),
            "oracle_lp_mu_fut_mean": (
                float(self._oracle_lp_mu_fut.mean().item())
                if self._oracle_lp_mu_fut is not None else 0.0
            ),
            "lp_mu_abs_error": (
                float((self._lp_mu_fut_hat - self._oracle_lp_mu_fut).abs().mean().item())
                if self._lp_mu_fut_hat is not None and self._oracle_lp_mu_fut is not None
                else 0.0
            ),
            # Phase energy gate stats (only meaningful when phase_energy_gate=True)
            "phase_energy_gate_mean": (
                float(self._phase_energy_gate.mean().item())
                if self._phase_energy_gate is not None else 1.0
            ),
            "phase_energy_gate_min": (
                float(self._phase_energy_gate.min().item())
                if self._phase_energy_gate is not None else 1.0
            ),
            "phase_energy_gate_max": (
                float(self._phase_energy_gate.max().item())
                if self._phase_energy_gate is not None else 1.0
            ),
            "phase_energy_r_high_mean": (
                float(self._phase_energy_r_high.mean().item())
                if self._phase_energy_r_high is not None else 0.0
            ),
            "phase_energy_r_high_q75": (
                float(torch.quantile(self._phase_energy_r_high.float(), 0.75).item())
                if self._phase_energy_r_high is not None else 0.0
            ),
            "phase_energy_risk_mean": (
                float(self._phase_energy_risk.mean().item())
                if self._phase_energy_risk is not None else 0.0
            ),
            "phase_energy_active_ratio": (
                float((self._phase_energy_risk > 0.05).float().mean().item())
                if self._phase_energy_risk is not None else 0.0
            ),
            # Mean gate stats
            "phase_energy_mean_gate_enabled": 1.0 if self.phase_energy_mean_gate else 0.0,
            "phase_energy_mean_strength": self.phase_energy_mean_strength,
            "phase_energy_mean_mix_mean": (
                float((self.phase_energy_mean_strength * self._phase_energy_risk).mean().item())
                if self.phase_energy_mean_gate and self._phase_energy_risk is not None else 0.0
            ),
            "phase_output_mean_abs":           self._last_phase_output_mean_abs,
            "phase_output_zero_mean_applied":  1.0 if self._last_phase_output_zero_mean_applied else 0.0,
            "input_residual_rms": self._last_input_residual_rms,
            "val_pred_rms": self._last_pred_residual_rms_after_phase,
            "val_oracle_rms": self._last_oracle_residual_rms,
            "val_rms_ratio": self._last_residual_rms_ratio,
            "output_patch_mean_abs_before_gauge": self._last_output_patch_mean_abs_before_gauge,
            "output_rms_before_gauge": self._last_output_rms_before_gauge,
            "output_rms_after_gauge": self._last_output_rms_after_gauge,
            "phase_angle_abs_mean": self._last_phase_angle_abs_mean,
            "phase_effect_rms": self._last_phase_effect_rms,
            "phase_effect_ratio": self._last_phase_effect_ratio,
            "dc_gauge_ratio": self._last_dc_gauge_ratio,
            "pred_residual_rms_after_phase": self._last_pred_residual_rms_after_phase,
            "oracle_residual_rms": self._last_oracle_residual_rms,
            "residual_rms_ratio": self._last_residual_rms_ratio,
            "apn_conv_effect_rms": self._last_apn_conv_effect_rms,
            "apn_conv_effect_ratio": self._last_apn_conv_effect_ratio,
            "apn_pred_residual_rms": self._last_apn_pred_residual_rms,
            "apn_oracle_residual_rms": self._last_apn_oracle_residual_rms,
            "apn_residual_rms_ratio": self._last_apn_residual_rms_ratio,
            "apn_kernel_l1": self._last_apn_kernel_l1,
            "apn_kernel_l2": self._last_apn_kernel_l2,
            "apn_kernel_max_ratio": self._last_apn_kernel_max_ratio,
            "apn_kernel_center_ratio": self._last_apn_kernel_center_ratio,
            "apn_kernel_entropy": self._last_apn_kernel_entropy,
            "apn_phase_abs_err": self._last_apn_phase_abs_err,
            "apn_phase_cos": self._last_apn_phase_cos,
        }

    def get_route_diagnostics(self) -> dict:
        _b2sc_fields: dict = {
            "b2sc_enable":  self.b2sc_enable,
            "b2sc_mode":    self._b2sc_mode,
            "b2sc_applied": self._last_b2sc_applied,
            "b2sc_delta_mu_abs_mean": (
                float(self._last_b2sc_delta_mu.abs().mean().item())
                if self._last_b2sc_delta_mu is not None else 0.0
            ),
            "b2sc_delta_logsigma_abs_mean": (
                float(self._last_b2sc_delta_logsigma.abs().mean().item())
                if self._last_b2sc_delta_logsigma is not None else 0.0
            ),
        }
        if self.route_path == "none":
            _ssc_fields: dict = {
                "route_path":              self.route_path,
                "route_state":             self.route_state,
                "san_ablation_mode":       self.san_ablation_mode,
                "ssc_enable":             self.ssc_enable,
                "ssc_mode":               self._ssc_mode,
                "ssc_decay_rho":          self.ssc_decay_rho,
                "ssc_trend_mem_abs_mean": (
                    float(self._ssc_trend_mem.abs().mean().item())
                    if self._ssc_trend_mem is not None else 0.0
                ),
                "ssc_queue_len": len(self._ssc_pred_queue),
            }
            return {**_b2sc_fields, **_ssc_fields}
        diag: dict = {
            "route_path":  self._route_path_name,
            "route_state": self._route_state_name,
            "san_ablation_mode": self.san_ablation_mode,
        }
        if self.route_path == "timeapn_correction":
            diag["apn_loss_mean"]  = self._last_apn_loss_mean
            diag["apn_loss_phase"] = self._last_apn_loss_phase
            diag["apn_loss_recon"] = self._last_apn_loss_recon
            diag["apn_conv_effect_rms"] = self._last_apn_conv_effect_rms
            diag["apn_conv_effect_ratio"] = self._last_apn_conv_effect_ratio
            diag["apn_pred_residual_rms"] = self._last_apn_pred_residual_rms
            diag["apn_oracle_residual_rms"] = self._last_apn_oracle_residual_rms
            diag["apn_residual_rms_ratio"] = self._last_apn_residual_rms_ratio
            diag["apn_kernel_l1"] = self._last_apn_kernel_l1
            diag["apn_kernel_l2"] = self._last_apn_kernel_l2
            diag["apn_kernel_max_ratio"] = self._last_apn_kernel_max_ratio
            diag["apn_kernel_center_ratio"] = self._last_apn_kernel_center_ratio
            diag["apn_kernel_entropy"] = self._last_apn_kernel_entropy
            diag["apn_phase_abs_err"] = self._last_apn_phase_abs_err
            diag["apn_phase_cos"] = self._last_apn_phase_cos
            if self._apn_pred_ms is not None:
                pred_m, pred_s = self._apn_pred_ms
                diag["apn_pred_m_mean"] = float(pred_m.mean().item())
                diag["apn_pred_s_mean"] = float(pred_s.mean().item())
        elif self.route_path == "lp_state_correction":
            diag["lp_mu_hist_mean"]       = (
                float(self._lp_mu_hist.mean().item())
                if self._lp_mu_hist is not None else 0.0
            )
            diag["lp_mu_fut_hat_mean"]    = (
                float(self._lp_mu_fut_hat.mean().item())
                if self._lp_mu_fut_hat is not None else 0.0
            )
            diag["oracle_lp_mu_fut_mean"] = (
                float(self._oracle_lp_mu_fut.mean().item())
                if self._oracle_lp_mu_fut is not None else 0.0
            )
            diag["lp_mu_abs_error"] = (
                float(
                    (self._lp_mu_fut_hat - self._oracle_lp_mu_fut).abs().mean().item()
                )
                if self._lp_mu_fut_hat is not None and self._oracle_lp_mu_fut is not None
                else 0.0
            )
            diag["base_std_fut_mean"] = (
                float(self._base_std_fut_hat.mean().item())
                if self._base_std_fut_hat is not None else 0.0
            )
        elif self.route_path_impl is not None and hasattr(
            self.route_path_impl, "get_route_diagnostics"
        ):
            diag.update(self.route_path_impl.get_route_diagnostics())
        diag.update(_b2sc_fields)
        return diag

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
# Internal classes for pure SAN causal-EMA sequence predictor
# ---------------------------------------------------------------------------


class _PatchCausalEMAEncoder(nn.Module):
    """Causal EMA encoder over a patch sequence.

    Processes tokens of shape ``(B*C, P_hist, H)`` and returns the final
    hidden state ``(B*C, H)`` via the stable recursion:

        h_t = a * h_{t-1} + (1 – a) * u_t,   h_0 = 0

    where ``a = sigmoid(a_logit)`` is a learnable per-dimension smoothing
    coefficient initialised near 0.9 (smooth, long-memory).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # sigmoid(log 9) ≈ 0.9 → start with strong memory
        init_val = torch.log(torch.tensor(9.0))
        self.a_logit = nn.Parameter(torch.full((hidden_dim,), float(init_val)))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: ``(BC, P, H)`` — projected patch tokens
        Returns:
            ``(BC, H)`` — final EMA hidden state
        """
        a = torch.sigmoid(self.a_logit)   # (H,)
        BC, P, H = u.shape
        h = u.new_zeros(BC, H)
        one_minus_a = 1.0 - a
        for t in range(P):
            h = a * h + one_minus_a * u[:, t, :]
        return h


class _PureSANSeqPredictor(nn.Module):
    """Pure-SAN patch-sequence predictor (causal EMA encoder).

    Replaces the flatten-then-Linear approach of ``_SANBasePredictor`` for
    ``route_path='none'``.  All other route paths continue to use
    ``_SANBasePredictor`` unchanged.

    Interface is intentionally different from ``_SANBasePredictor.predict``
    (it takes patch-form raw input, not a flat xbar), so there is no risk of
    accidental cross-use.

    Construction-time initialisation guarantees:
      - ``gamma_mu  = 0``  →  ``mu_fut``  starts at *anchor* (constant extrapolation).
      - ``gamma_std = 0``  →  ``std_fut`` starts at exp(logstd_anchor)
                              = mean of historical patch stds.
    """

    def __init__(
        self,
        hist_stat_len: int,
        pred_stat_len: int,
        window_len: int,
        enc_in: int,
        sigma_min: float = 1e-3,
        hidden_dim: int = 128,
        use_phase: bool = False,
        phase_k: int = 8,
        phase_zero_init: bool = True,
        state_residual_scale_init: float = 0.1,
        use_amp: bool = False,
    ):
        super().__init__()
        self.pred_stat_len = pred_stat_len
        self.hist_stat_len = hist_stat_len
        self.sigma_min     = sigma_min
        self.enc_in        = enc_in
        self.phase_k       = phase_k
        self.use_phase     = use_phase
        self.use_amp       = use_amp and use_phase  # amp only valid with phase
        self.state_residual_scale_init = float(state_residual_scale_init)

        # phase_head is always None in the new implementation.
        # Kept as an attribute so that any external code checking
        # ``predictor.phase_head is not None`` safely returns False.
        self.phase_head: Optional[nn.Linear] = None

        token_len = window_len + 1   # 1 stat token + window_len raw values

        # ---- mean branch ----
        self.mean_in_proj = nn.Linear(token_len, hidden_dim)
        self.mean_encoder = _PatchCausalEMAEncoder(hidden_dim)
        self.mean_post    = nn.Linear(hidden_dim, hidden_dim)
        self.mean_out     = nn.Linear(hidden_dim, pred_stat_len)
        nn.init.zeros_(self.mean_out.weight)
        nn.init.zeros_(self.mean_out.bias)
        self.gamma_mu = nn.Parameter(torch.full((enc_in,), float(state_residual_scale_init)))

        # ---- std branch (only exercised when use_phase=False) ----
        self.std_in_proj = nn.Linear(token_len, hidden_dim)
        self.std_encoder = _PatchCausalEMAEncoder(hidden_dim)
        self.std_post    = nn.Linear(hidden_dim, hidden_dim)
        self.std_out     = nn.Linear(hidden_dim, pred_stat_len)
        nn.init.zeros_(self.std_out.weight)
        nn.init.zeros_(self.std_out.bias)
        self.gamma_std = nn.Parameter(torch.full((enc_in,), float(state_residual_scale_init)))

        # ---- phase branch (only created when use_phase=True) ----
        # Processes per-channel phase features extracted via rfft on
        # mean-normalized patches.  Each patch token is the (re, im) pair
        # of bins 1..K for one channel: token size = 2*K.
        # Runs in (B*C, P_hist, 2K) space, parallel to mean branch.
        if use_phase:
            phase_token_len = 2 * phase_k   # per-channel phase token per patch
            self.phase_in_proj  = nn.Linear(phase_token_len, hidden_dim)
            self.phase_encoder  = _PatchCausalEMAEncoder(hidden_dim)
            self.phase_post     = nn.Linear(hidden_dim, hidden_dim)
            # Output: P_fut * K * 2 per channel, then reshaped to (B, P_fut, K, C, 2)
            self.phase_out_proj = nn.Linear(hidden_dim, pred_stat_len * phase_k * 2)
            if phase_zero_init:
                nn.init.zeros_(self.phase_out_proj.weight)
                nn.init.zeros_(self.phase_out_proj.bias)

        # ---- amp branch (only created when use_phase=True and use_amp=True) ----
        # Processes per-channel historical log residual RMS sequence.
        # Each patch token is a scalar (log_amp for one channel): token size = 1.
        # At init: amp_out zero-initialized → log_amp_fut = log_amp_anchor
        # (mean of historical log amplitudes), i.e. constant-amplitude extrapolation.
        if self.use_amp:
            self.amp_in_proj = nn.Linear(1, hidden_dim)
            self.amp_encoder = _PatchCausalEMAEncoder(hidden_dim)
            self.amp_post    = nn.Linear(hidden_dim, hidden_dim)
            self.amp_out     = nn.Linear(hidden_dim, pred_stat_len)
            nn.init.zeros_(self.amp_out.weight)
            nn.init.zeros_(self.amp_out.bias)
            self.gamma_amp = nn.Parameter(torch.full((enc_in,), float(state_residual_scale_init)))

    def predict_mean_std_seq(
        self,
        mu_hist:     torch.Tensor,   # (B, P_hist, C)
        std_hist:    torch.Tensor,   # (B, P_hist, C)
        anchor:      torch.Tensor,   # (B, 1, C)
        raw_patches: torch.Tensor,   # (B, P_hist, W, C)
        z_patches:   torch.Tensor,   # (B, P_hist, W, C)
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict future mean/std using the causal sequence architecture.

        This is the no-phase ablation entry point: it reuses the patch-sequence
        mean encoder and causal EMA std branch, predicts future log-std, and
        exponentiates back to positive std.  Zero-initialized output heads make
        std start at the mean historical log-std.
        """
        B, P_hist, W, C = raw_patches.shape
        BC = B * C

        # Mean branch: identical tokenization to the phase predictor's mean head.
        mu_center    = mu_hist - anchor
        raw_center   = raw_patches - anchor.unsqueeze(2)
        mu_center_t  = mu_center.permute(0, 2, 1).reshape(BC, P_hist, 1)
        raw_center_t = raw_center.permute(0, 3, 1, 2).reshape(BC, P_hist, W)
        mean_token   = torch.cat([mu_center_t, raw_center_t], dim=-1)

        mean_h = self.mean_encoder(self.mean_in_proj(mean_token))
        mean_h = mean_h + self.mean_post(F.gelu(mean_h))
        mu_res = self.mean_out(mean_h)
        mu_res = mu_res.reshape(B, C, self.pred_stat_len).permute(0, 2, 1)
        mu_fut = (
            anchor.expand(-1, self.pred_stat_len, -1)
            + self.gamma_mu.view(1, 1, -1) * mu_res
        )

        # Std branch: causal EMA over historical log-std plus z-patch shape.
        std_anchor    = std_hist.mean(dim=1, keepdim=True).clamp_min(self.sigma_min)
        logstd_hist   = torch.log(std_hist.clamp(min=self.sigma_min))
        logstd_anchor = torch.log(std_anchor)
        logstd_center = logstd_hist - logstd_anchor
        logstd_center_t = logstd_center.permute(0, 2, 1).reshape(BC, P_hist, 1)
        z_patch_t = z_patches.permute(0, 3, 1, 2).reshape(BC, P_hist, W)
        std_token = torch.cat([logstd_center_t, z_patch_t], dim=-1)

        std_h = self.std_encoder(self.std_in_proj(std_token))
        std_h = std_h + self.std_post(F.gelu(std_h))
        logstd_res = self.std_out(std_h)
        logstd_res = logstd_res.reshape(B, C, self.pred_stat_len).permute(0, 2, 1)
        logstd_fut = (
            logstd_anchor.expand(-1, self.pred_stat_len, -1)
            + self.gamma_std.view(1, 1, -1) * logstd_res
        )
        std_fut = logstd_fut.exp().clamp(min=self.sigma_min)
        return mu_fut, std_fut

    def predict(
        self,
        mu_hist:     torch.Tensor,   # (B, P_hist, C)
        std_hist:    torch.Tensor,   # (B, P_hist, C)
        anchor:      torch.Tensor,   # (B, 1, C)
        raw_patches: torch.Tensor,   # (B, P_hist, window_len, C)
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu_fut, std_fut), each of shape (B, P_fut, C)."""
        B, P_hist, W, C = raw_patches.shape
        BC = B * C

        # ----------------------------------------------------------------
        # Mean branch
        # ----------------------------------------------------------------
        mu_center    = mu_hist - anchor                                      # (B, P_hist, C)
        raw_center   = raw_patches - anchor.unsqueeze(2)                     # (B, P_hist, W, C)

        # rearrange to (BC, P_hist, token_len)
        mu_center_t  = mu_center.permute(0, 2, 1).reshape(BC, P_hist, 1)    # (BC, P_hist, 1)
        raw_center_t = raw_center.permute(0, 3, 1, 2).reshape(BC, P_hist, W)# (BC, P_hist, W)
        mean_token   = torch.cat([mu_center_t, raw_center_t], dim=-1)        # (BC, P_hist, W+1)

        mean_h = self.mean_encoder(self.mean_in_proj(mean_token))            # (BC, hidden)
        mean_h = mean_h + self.mean_post(F.gelu(mean_h))
        mu_res = self.mean_out(mean_h)                                        # (BC, P_fut)
        mu_res = mu_res.reshape(B, C, self.pred_stat_len).permute(0, 2, 1)   # (B, P_fut, C)
        mu_fut = anchor.expand(-1, self.pred_stat_len, -1) + self.gamma_mu.view(1, 1, -1) * mu_res

        # ----------------------------------------------------------------
        # Std branch (all in log-sigma space)
        # ----------------------------------------------------------------
        logstd_hist   = torch.log(std_hist.clamp(min=self.sigma_min))        # (B, P_hist, C)
        logstd_anchor = logstd_hist.mean(dim=1, keepdim=True)                # (B, 1,      C)
        logstd_center = logstd_hist - logstd_anchor                          # (B, P_hist, C)
        z_patch       = (raw_patches - mu_hist.unsqueeze(2)) / (std_hist.unsqueeze(2) + eps)  # (B, P_hist, W, C)

        logstd_center_t = logstd_center.permute(0, 2, 1).reshape(BC, P_hist, 1)   # (BC, P_hist, 1)
        z_patch_t       = z_patch.permute(0, 3, 1, 2).reshape(BC, P_hist, W)       # (BC, P_hist, W)
        std_token       = torch.cat([logstd_center_t, z_patch_t], dim=-1)          # (BC, P_hist, W+1)

        std_h      = self.std_encoder(self.std_in_proj(std_token))                  # (BC, hidden)
        std_h      = std_h + self.std_post(F.gelu(std_h))
        logstd_res = self.std_out(std_h)                                            # (BC, P_fut)
        logstd_res = logstd_res.reshape(B, C, self.pred_stat_len).permute(0, 2, 1) # (B, P_fut, C)
        logstd_fut = logstd_anchor.expand(-1, self.pred_stat_len, -1) + self.gamma_std.view(1, 1, -1) * logstd_res
        std_fut    = logstd_fut.exp().clamp(min=self.sigma_min)

        return mu_fut, std_fut

    def predict_mean_phase(
        self,
        mu_hist:     torch.Tensor,   # (B, P_hist, C)
        anchor:      torch.Tensor,   # (B, 1, C)
        raw_patches: torch.Tensor,   # (B, P_hist, W, C)  raw (un-normalized)
        z_patches:   torch.Tensor,   # (B, P_hist, W, C)  mean-normalized patches
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict future mean and future phase rotation.

        Used exclusively by the pure-baseline + use_phase=True path.
        ``predict()`` (mu/std) is NOT called in this mode.

        Returns
        -------
        mu_fut_hat:        (B, P_fut, C)
        phase_rot_fut_hat: (B, P_fut, K, C, 2)
            Residual rotation parameters (a, b).  The actual unit-complex
            rotation is normalize(1+a, b).  Zero-initialized → identity.
        """
        B, P_hist, W, C = raw_patches.shape
        BC = B * C
        K  = self.phase_k

        # ----------------------------------------------------------------
        # Mean branch — identical to predict()
        # ----------------------------------------------------------------
        mu_center    = mu_hist - anchor                                        # (B, P_hist, C)
        raw_center   = raw_patches - anchor.unsqueeze(2)                       # (B, P_hist, W, C)

        mu_center_t  = mu_center.permute(0, 2, 1).reshape(BC, P_hist, 1)      # (BC, P_hist, 1)
        raw_center_t = raw_center.permute(0, 3, 1, 2).reshape(BC, P_hist, W)  # (BC, P_hist, W)
        mean_token   = torch.cat([mu_center_t, raw_center_t], dim=-1)          # (BC, P_hist, W+1)

        mean_h = self.mean_encoder(self.mean_in_proj(mean_token))              # (BC, hidden)
        mean_h = mean_h + self.mean_post(F.gelu(mean_h))
        mu_res = self.mean_out(mean_h)                                          # (BC, P_fut)
        mu_res = mu_res.reshape(B, C, self.pred_stat_len).permute(0, 2, 1)     # (B, P_fut, C)
        mu_fut = (
            anchor.expand(-1, self.pred_stat_len, -1)
            + self.gamma_mu.view(1, 1, -1) * mu_res
        )

        # ----------------------------------------------------------------
        # Phase branch
        # Compute rfft of mean-normalized patches along the window dim.
        # Bins 1..K carry the low-frequency phase information; DC (bin 0)
        # is skipped because it encodes mean (already handled above).
        # ----------------------------------------------------------------
        # z_patches: (B, P_hist, W, C) → rearrange to (BC, P_hist, W)
        z_bc = z_patches.permute(0, 3, 1, 2).reshape(BC, P_hist, W)           # (BC, P_hist, W)
        Z = torch.fft.rfft(z_bc, dim=-1)                                        # (BC, P_hist, W//2+1)

        K_eff = min(K, Z.shape[-1] - 1)                                        # exclude DC bin 0
        Z_bins = Z[:, :, 1:K_eff + 1]                                          # (BC, P_hist, K_eff) complex

        # Amplitude-normalize → pure phase unit complex
        Z_abs    = Z_bins.abs().clamp(min=1e-6)
        Z_unit   = Z_bins / Z_abs                                               # (BC, P_hist, K_eff)

        phase_re = Z_unit.real                                                  # (BC, P_hist, K_eff)
        phase_im = Z_unit.imag

        # Pad to K bins if K_eff < K (handles tiny window_len edge cases)
        if K_eff < K:
            pad = phase_re.new_zeros(BC, P_hist, K - K_eff)
            phase_re = torch.cat([phase_re, pad], dim=-1)
            phase_im = torch.cat([phase_im, pad], dim=-1)

        # Per-patch token: interleave re/im → (BC, P_hist, 2K)
        phase_token = torch.stack([phase_re, phase_im], dim=-1).reshape(BC, P_hist, K * 2)

        phase_h = self.phase_encoder(self.phase_in_proj(phase_token))          # (BC, hidden)
        phase_h = phase_h + self.phase_post(F.gelu(phase_h))
        phase_res = self.phase_out_proj(phase_h)                               # (BC, P_fut*K*2)
        # Reshape to (B, P_fut, K, C, 2)
        phase_rot = (
            phase_res
            .reshape(B, C, self.pred_stat_len, K, 2)
            .permute(0, 2, 3, 1, 4)                                            # (B, P_fut, K, C, 2)
        )

        return mu_fut, phase_rot

    def predict_mean_phase_amp(
        self,
        mu_hist:     torch.Tensor,   # (B, P_hist, C)
        amp_hist:    torch.Tensor,   # (B, P_hist, C) — sqrt(mean(z**2)+eps) per patch/channel
        anchor:      torch.Tensor,   # (B, 1, C)
        raw_patches: torch.Tensor,   # (B, P_hist, W, C)  raw (un-normalized)
        z_patches:   torch.Tensor,   # (B, P_hist, W, C)  mean-normalized patches
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict future mean, phase rotation, and log residual amplitude.

        Extends ``predict_mean_phase`` with an additional amplitude branch that
        predicts the future patch-level residual RMS from the historical
        log-amplitude sequence.

        Requires ``use_amp=True`` at construction time (enforced by caller).

        Returns
        -------
        mu_fut_hat        : (B, P_fut, C)
        phase_rot_fut_hat : (B, P_fut, K, C, 2)
        log_amp_fut_hat   : (B, P_fut, C) — log of predicted future residual RMS
        """
        mu_fut, phase_rot = self.predict_mean_phase(
            mu_hist=mu_hist,
            anchor=anchor,
            raw_patches=raw_patches,
            z_patches=z_patches,
            eps=eps,
        )

        if not self.use_amp:
            # Fallback: return zeros as log_amp (gain=1 everywhere)
            log_amp_zero = mu_fut.new_zeros(mu_fut.shape)
            return mu_fut, phase_rot, log_amp_zero

        B, P_hist, C = mu_hist.shape
        BC = B * C

        # ----------------------------------------------------------------
        # Amp branch: causal EMA over log-amplitude sequence
        # ----------------------------------------------------------------
        log_amp_hist   = torch.log(amp_hist.clamp(min=eps))              # (B, P_hist, C)
        log_amp_anchor = log_amp_hist.mean(dim=1, keepdim=True)          # (B, 1, C)
        log_amp_center = log_amp_hist - log_amp_anchor                   # (B, P_hist, C)

        # Rearrange to (BC, P_hist, 1) for EMA encoder
        log_amp_center_t = (
            log_amp_center.permute(0, 2, 1).reshape(BC, P_hist, 1)
        )   # (BC, P_hist, 1)

        amp_h    = self.amp_encoder(self.amp_in_proj(log_amp_center_t))  # (BC, hidden)
        amp_h    = amp_h + self.amp_post(F.gelu(amp_h))
        log_amp_res = self.amp_out(amp_h)                                 # (BC, P_fut)
        log_amp_res = (
            log_amp_res.reshape(B, C, self.pred_stat_len).permute(0, 2, 1)
        )   # (B, P_fut, C)

        log_amp_fut = (
            log_amp_anchor.expand(-1, self.pred_stat_len, -1)
            + self.gamma_amp.view(1, 1, -1) * log_amp_res
        )   # (B, P_fut, C)

        return mu_fut, phase_rot, log_amp_fut


# ---------------------------------------------------------------------------
# Internal MLP and predictor classes (SAN-based routes only)
# ---------------------------------------------------------------------------


class _SANBaseMLP(nn.Module):
    """Lightweight MLP for future-stats prediction.

    mode='mu':    anchor + residual with Tanh + learnable blend weight.
    mode='sigma': direct sigma prediction with GELU + Softplus.
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
        self.raw_input   = nn.Linear(raw_hist_len,  hidden_dim)
        self.output      = nn.Linear(2 * hidden_dim, pred_stat_len)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        if mode == "mu":
            self.activation       = nn.Tanh()
            self.final_activation = nn.Identity()
            self.weight = nn.Parameter(torch.ones(2, enc_in))
        else:
            self.activation       = nn.GELU()
            self.final_activation = nn.Softplus()
            self.weight = None

    def forward(
        self,
        x_stats: torch.Tensor,
        x_raw:   torch.Tensor,
        anchor:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xs = x_stats.permute(0, 2, 1)
        xr = x_raw.permute(0, 2, 1)
        feat = torch.cat([self.stats_input(xs), self.raw_input(xr)], dim=-1)
        pred = self.final_activation(
            self.output(self.activation(feat))
        ).permute(0, 2, 1)

        if self.mode == "mu" and anchor is not None:
            anch = anchor.expand(-1, self.pred_stat_len, -1)
            pred = (
                pred * self.weight[0].view(1, 1, -1)
                + anch * self.weight[1].view(1, 1, -1)
            )
        return pred


class _SANBasePredictor(nn.Module):
    """Predicts future (mean, std) from historical window stats and input."""

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
        mu_hist:   torch.Tensor,
        std_hist:  torch.Tensor,
        xbar:      Optional[torch.Tensor],
        anchor:    torch.Tensor,
        xbar_raw:  Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if xbar_raw is not None:
            raw_len       = xbar_raw.shape[1]
            anchor_broad  = anchor.expand(-1, raw_len, -1)
            mu_fut  = self.mean_head(
                mu_hist - anchor.expand(-1, mu_hist.shape[1], -1),
                xbar_raw - anchor_broad,
                anchor,
            )
            std_fut = self.std_head(std_hist, xbar_raw, None).clamp(min=self.sigma_min)
        else:
            mu_fut  = self.mean_head(
                mu_hist - anchor.expand(-1, mu_hist.shape[1], -1), xbar, anchor
            )
            std_fut = self.std_head(std_hist, xbar, None).clamp(min=self.sigma_min)
        return mu_fut, std_fut


class RouteStatePredictor(nn.Module):
    """Shared predictor for future route state — state-agnostic."""

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
        self.state_input = nn.Linear(hist_stat_len, hidden_dim)
        self.raw_input   = nn.Linear(raw_hist_len,  hidden_dim)
        self.activation  = nn.Tanh()
        self.output      = nn.Linear(2 * hidden_dim, pred_stat_len)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        hist_state: torch.Tensor,
        xbar:       torch.Tensor,
    ) -> torch.Tensor:
        hs  = hist_state.permute(0, 2, 1)
        xr  = xbar.permute(0, 2, 1)
        feat = torch.cat([self.state_input(hs), self.raw_input(xr)], dim=-1)
        out  = self.output(self.activation(feat))
        return out.permute(0, 2, 1)
