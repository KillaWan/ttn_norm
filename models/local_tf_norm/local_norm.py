from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .stft import STFT
from .gate import LocalTFGate


@dataclass
class LocalTFNormState:
    x_tf: torch.Tensor
    n_tf: torch.Tensor
    g_local: torch.Tensor
    n_time: torch.Tensor
    length: int
    pred_n_time: Optional[torch.Tensor] = None
    pred_n_tf: Optional[torch.Tensor] = None
    mean: Optional[torch.Tensor] = None   # (B, 1, C) instance mean for RevIN
    std: Optional[torch.Tensor] = None    # (B, 1, C) instance std for RevIN


class SimpleTFPredictor(nn.Module):
    """
    Time-frequency predictor for non-stationary component.
    
    Input: real/imaginary feature tensor
    - pred_input="n_tf": (B, C, F, T, 2) where 2 is real/imag
    - pred_input="n_tf_x_tf": (B, C, F, T, 4) where 4 is [n_real, n_imag, x_real, x_imag]
    
    Processing: reshape to (B*C*F, T*feat_dim), pass through MLP, reshape back to (B, C, F, out_T, 2)
    """
    def __init__(
        self,
        in_frames: int,
        out_frames: int,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        input_feature_dim: int = 2,
    ):
        """
        Args:
            in_frames: number of input frames (time steps)
            out_frames: number of output frames
            hidden_dim: hidden dimension in MLP (0 = linear)
            dropout: dropout rate
            input_feature_dim: feature dimension (2 for n_tf real+imag, 4 for n_tf_x_tf context)
        """
        super().__init__()
        # Input: T * feat_dim; Output: out_T * 2 (always real+imag)
        in_features = in_frames * input_feature_dim
        out_features = out_frames * 2
        
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_features),
            )
        else:
            self.net = nn.Linear(in_features, out_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, F, T, feat_dim) where feat_dim in {2, 4}
        
        Returns:
            output: (B, C, F, out_T, 2) as real tensor, will be converted to complex outside
        """
        b, c, f, t, feat_dim = features.shape
        # sanity: feature dim should be 2 (real+imag) for n_tf, or 4 for n_tf_x_tf
        assert feat_dim in (2, 4), f"unexpected feature dim {feat_dim}, expected 2 or 4"

        # Reshape to (B*C*F, T*feat_dim) for MLP
        features_flat = features.reshape(b * c * f, t * feat_dim)
        
        # MLP: (B*C*F, T*feat_dim) -> (B*C*F, out_T*2)
        output_flat = self.net(features_flat)
        
        # Reshape to (B, C, F, out_T, 2)
        out_frames = output_flat.shape[-1] // 2
        output = output_flat.reshape(b, c, f, out_frames, 2)
        
        # Convert to complex
        return torch.view_as_complex(output.contiguous())


class LocalTFNorm(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_fft: int | None = None,
        hop_length: int | None = None,
        win_length: int | None = None,
        gate_type: str = "depthwise",
        gate_log_mag: bool = True,
        gate_arch: str = "pointwise",
        gate_threshold_mode: str = "shift",
        gate_entropy_weight: float = 0.0,
        gate_use_log_mag: Optional[bool] = None,
        stationarity_loss_weight: float = 0.0,
        stationarity_chunks: int = 4,
        future_mode: str = "repeat_last",
        predict_n_time: bool = True,
        pred_hidden_dim: int = 64,
        pred_dropout: float = 0.1,
        pred_loss_weight: float = 1.0,
        gate_threshold: float = 0.0,
        gate_temperature: float = 1.0,
        gate_smooth_weight: float = 0.0,
        gate_ratio_weight: float = 0.0,
        gate_ratio_target: float = 0.3,
        gate_mode: str = "sigmoid",
        gate_budget_dim: str = "freq",
        pred_input: str = "n_tf",
        gate_temporal_smooth_weight: float = 0.0,
        use_instance_norm: bool = True,
        lambda_E: float = 1.0,
        lambda_P: float = 1.0,
        eps_E: float = 1e-6,
        delta_E: float = 0.0,
        delta_P: float = 0.0,
        trigger_mask: bool = True,
        delta_E_mask: float = 0.0,
        delta_P_mask: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.use_instance_norm = use_instance_norm
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.gate_log_mag = gate_log_mag
        self.gate_arch = gate_arch
        self.gate_threshold_mode = gate_threshold_mode
        self.gate_entropy_weight = float(gate_entropy_weight)
        # gate_use_log_mag may be None -> fall back to legacy gate_log_mag
        self.gate_use_log_mag = gate_use_log_mag
        self.stationarity_loss_weight = stationarity_loss_weight
        self.stationarity_chunks = stationarity_chunks
        self.future_mode = future_mode
        self.predict_n_time = predict_n_time
        self.pred_loss_weight = pred_loss_weight
        self.gate_smooth_weight = gate_smooth_weight
        self.gate_temporal_smooth_weight = gate_temporal_smooth_weight
        self.gate_ratio_weight = gate_ratio_weight
        self.gate_ratio_target = gate_ratio_target
        self.gate_mode = gate_mode
        self.pred_input = pred_input
        self.lambda_E = float(lambda_E)
        self.lambda_P = float(lambda_P)
        self.eps_E = float(eps_E)
        self.delta_E = float(delta_E)
        self.delta_P = float(delta_P)
        self.trigger_mask = bool(trigger_mask)
        self.delta_E_mask = float(delta_E_mask)
        self.delta_P_mask = float(delta_P_mask)

        if n_fft is None:
            n_fft = min(64, max(16, seq_len // 4))
        n_fft = min(n_fft, seq_len)
        self.stft = STFT(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann"
        )
        self.gate = LocalTFGate(
            enc_in,
            gate_type=gate_type,
            use_threshold=True,
            init_threshold=gate_threshold,
            temperature=gate_temperature,
            gate_mode=gate_mode,
            gate_budget_dim=gate_budget_dim,
            gate_ratio_target=gate_ratio_target,
            gate_arch=self.gate_arch,
            gate_threshold_mode=self.gate_threshold_mode,
        )
        in_frames = self.stft.time_bins(self.seq_len)
        out_frames = self.stft.time_bins(self.pred_len)
        
        # Determine predictor input feature dimension
        # n_tf only: 2 features (real, imag)
        # n_tf_x_tf: 4 features (n_real, n_imag, x_real, x_imag)
        input_feature_dim = 4 if pred_input == "n_tf_x_tf" else 2
        
        self.n_tf_predictor = (
            SimpleTFPredictor(
                in_frames,
                out_frames,
                hidden_dim=pred_hidden_dim,
                dropout=pred_dropout,
                input_feature_dim=input_feature_dim,
            )
            if predict_n_time
            else None
        )
        
        # Store last gate stats for monitoring
        self._last_gate_mean = 0.0
        self._last_gate_sum_f = 0.0
        self._last_gate_max_f = 0.0
        self._last_gate_ent_f = 0.0

        self._last_state: Optional[LocalTFNormState] = None
        self._last_residual: Optional[torch.Tensor] = None
        self._last_gate: Optional[torch.Tensor] = None
        self._last_r_tf: Optional[torch.Tensor] = None
        self._last_L_E: float = 0.0
        self._last_L_P: float = 0.0
        self._last_stationarity_loss: float = 0.0

        # Extended diagnostic caches (detached, no grad)
        self._last_x_time: Optional[torch.Tensor] = None      # (B,T,C) after RevIN
        self._last_x_tf: Optional[torch.Tensor] = None        # (B,C,F,TT) complex
        self._last_n_tf: Optional[torch.Tensor] = None        # (B,C,F,TT) complex
        self._last_n_time: Optional[torch.Tensor] = None      # (B,T,C)
        self._last_r_time: Optional[torch.Tensor] = None      # (B,T,C) == _last_residual
        self._last_pred_n_tf: Optional[torch.Tensor] = None   # (B,C,F,TT) or None
        self._last_pred_n_time: Optional[torch.Tensor] = None # (B,T,C) or None
        self._last_inst_mean: Optional[torch.Tensor] = None   # (B,1,C) or None
        self._last_inst_std: Optional[torch.Tensor] = None    # (B,1,C) or None
        self._last_dE: Optional[torch.Tensor] = None          # (B,C,T-1) log-energy steps
        self._last_dP: Optional[torch.Tensor] = None          # (B,C,T-1) shape steps

        # Denormalize branch usage counters (reset each epoch by caller)
        self._denorm_used_pred: int = 0
        self._denorm_used_input: int = 0
        self._denorm_used_extrap: int = 0

        # Trigger mask diagnostic caches (all detached)
        self._last_mask_rate: float = 1.0
        self._last_mask_trig_rate: float = 0.0
        self._last_mask_delta_E: float = float("nan")
        self._last_mask_delta_P: float = float("nan")
        self._last_trigger_mask_t: Optional[torch.Tensor] = None

        # Trigger mask coverage diagnostics
        self._last_mask_cov: float = float("nan")
        self._last_mask_min_rate_cov95: float = float("nan")
        self._last_mask_min_rate_cov99: float = float("nan")
        self._last_mask_min_rate_cov995: float = float("nan")
        self._last_mask_p_table: dict = {}  # key: p (float), value: (mask_rate, coverage)

        # Pre-RevIN raw input cache (needed for teacher extraction)
        self._last_x_raw: Optional[torch.Tensor] = None

        # Prediction supervision loss cache
        self._last_pred_sup_loss: float = 0.0

    def _dE_dP_from_tf(
        self, tf_complex: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-transition log-energy and spectral-shape differences.

        Uses the same formula as loss() but accepts any TF tensor as input.

        Args:
            tf_complex: (B, C, F, T) complex tensor

        Returns:
            dE: (B, C, T-1) log-energy differences
            dP: (B, C, T-1) spectral-shape differences
        """
        eps = 1e-8
        P = tf_complex.abs() ** 2                               # (B, C, F, T)
        E = P.mean(dim=2)                                       # (B, C, T)
        p = P / (P.sum(dim=2, keepdim=True) + eps)              # (B, C, F, T)
        logE = torch.log(E + self.eps_E)
        dE = torch.abs(logE[..., 1:] - logE[..., :-1])          # (B, C, T-1)
        dP = torch.abs(p[..., :, 1:] - p[..., :, :-1]).mean(dim=2)  # (B, C, T-1)
        return dE, dP

    def _make_gate(self, x_tf: torch.Tensor) -> torch.Tensor:
        magnitude = x_tf.abs()
        # decide whether to use log-mag: explicit override wins, else legacy flag
        if self.gate_use_log_mag is None:
            use_log_mag = bool(self.gate_log_mag)
        else:
            use_log_mag = bool(self.gate_use_log_mag)
        if use_log_mag:
            magnitude = torch.log1p(magnitude)
        return self.gate(magnitude)

    def _compute_gate_stats(self, g_local: torch.Tensor) -> None:
        """
        Compute gate statistics based on last forward pass.
        g_local: (B, C, F, T) gate activations
        
        Stats:
        - gate_mean: mean activation across all dims
        - gate_sum_f: sum along freq dim (F), then take mean over batch/channel/time
        - gate_max_f: max along freq dim (F), then take mean over batch/channel/time
        - ent_f: normalized entropy along freq dim, then take mean over batch/channel/time
        """
        # gate_mean: global mean
        self._last_gate_mean = float(g_local.mean().detach().cpu().item())
        
        # Sum and max along frequency dimension (dim=2)
        # From (B, C, F, T), sum/max over F to get (B, C, T)
        g_sum_f = g_local.sum(dim=2)  # (B, C, T)
        g_max_f = g_local.max(dim=2)[0]  # (B, C, T)
        
        # Take mean over batch/channel/time to get scalar
        self._last_gate_sum_f = float(g_sum_f.mean().detach().cpu().item())
        self._last_gate_max_f = float(g_max_f.mean().detach().cpu().item())
        
        # Entropy: compute per (batch, channel, time) point along freq
        # First normalize g_local along freq dimension to get probability
        g_norm = g_local / (g_local.sum(dim=2, keepdim=True) + 1e-10)  # (B, C, F, T)
        
        # Compute entropy: -sum(p * log(p))
        eps = 1e-10
        entropy = -(g_norm * torch.log(g_norm + eps)).sum(dim=2)  # (B, C, T)
        
        # Normalize by max entropy: log(F)
        f_size = g_local.shape[2]
        max_entropy = np.log(f_size)
        if max_entropy > 0:
            entropy_norm = entropy / max_entropy
        else:
            entropy_norm = entropy
        
        # Take mean over batch/channel/time
        self._last_gate_ent_f = float(entropy_norm.mean().detach().cpu().item())

    def get_last_gate_stats(self) -> dict[str, float]:
        """Return last computed gate statistics for monitoring."""
        return {
            "gate_mean": self._last_gate_mean,
            "gate_sum_f": self._last_gate_sum_f,
            "gate_max_f": self._last_gate_max_f,
            "gate_ent_f": self._last_gate_ent_f,
        }

    def get_last_stationarity_stats(self) -> dict[str, float]:
        """Return last computed stationarity loss components for logging."""
        return {
            "stationarity_loss": self._last_stationarity_loss,
            "L_E": self._last_L_E,
            "L_P": self._last_L_P,
        }

    def normalize(
        self, batch_x: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, LocalTFNormState]:
        # Cache raw (pre-RevIN) input for teacher extraction in pred_supervision_loss
        self._last_x_raw = batch_x.detach()

        # RevIN-style instance normalization: remove per-sample mean and std
        inst_mean = inst_std = None
        if self.use_instance_norm:
            inst_mean = batch_x.mean(dim=1, keepdim=True)          # (B, 1, C)
            inst_std = batch_x.std(dim=1, keepdim=True) + 1e-8     # (B, 1, C)
            batch_x = (batch_x - inst_mean) / inst_std

        x_tf = self.stft(batch_x)
        g_local = self._make_gate(x_tf)

        # Trigger mask: restrict effective gate to boundary frames only
        if self.trigger_mask:
            with torch.no_grad():
                dE_x, dP_x = self._dE_dP_from_tf(x_tf)
            de = self.delta_E_mask if self.delta_E_mask > 0.0 else self.delta_E
            dp = self.delta_P_mask if self.delta_P_mask > 0.0 else self.delta_P
            if de <= 0.0 or dp <= 0.0:
                raise RuntimeError(
                    "trigger_mask enabled but delta_E_mask/delta_P_mask not set; "
                    "run calibration or set manually"
                )
            # trig: (B, C, T-1) — True wherever a transition exceeds threshold
            trig = (dE_x > de) | (dP_x > dp)
            T = x_tf.shape[-1]
            mask_t = torch.zeros(
                x_tf.shape[0], x_tf.shape[1], T,
                dtype=torch.bool, device=x_tf.device,
            )
            mask_t[..., :-1] |= trig   # left frame of each triggered transition
            mask_t[..., 1:] |= trig    # right frame of each triggered transition
            mask_f = mask_t.unsqueeze(2).float()  # (B, C, 1, T) broadcast over freq
            g_eff = g_local * mask_f              # (B, C, F, T)
            # Update trigger mask diagnostics (all detached)
            self._last_trigger_mask_t = mask_t.detach()
            self._last_mask_rate = float(mask_t.float().mean().cpu().item())
            self._last_mask_trig_rate = float(trig.float().mean().cpu().item())
            self._last_mask_delta_E = float(de)
            self._last_mask_delta_P = float(dp)

            # ---- Coverage diagnostics ----
            # Transition strength: s_trans[i] = max(dE[i]/de, dP[i]/dp) shape (B,C,T-1)
            _s_trans = torch.maximum(dE_x / de, dP_x / dp)
            # Map to frame strength via left-right coverage rule: same as mask_t
            _s_frame = torch.zeros(
                x_tf.shape[0], x_tf.shape[1], T,
                device=x_tf.device, dtype=_s_trans.dtype,
            )
            _s_frame[..., :-1] = torch.maximum(_s_frame[..., :-1], _s_trans)
            _s_frame[..., 1:] = torch.maximum(_s_frame[..., 1:], _s_trans)
            _cov_den = float(_s_frame.sum().cpu().item()) + 1e-12
            _cov_num = float((_s_frame * mask_t.float()).sum().cpu().item())
            self._last_mask_cov = _cov_num / _cov_den

            # p-table: for p in {0.95, 0.99, 0.995}, compute alt thresholds + coverage
            _p_table: dict = {}
            _dE_flat = dE_x.detach().float().flatten()
            _dP_flat = dP_x.detach().float().flatten()
            for _p in (0.95, 0.99, 0.995):
                _de_p = float(torch.quantile(_dE_flat, _p).item())
                _dp_p = float(torch.quantile(_dP_flat, _p).item())
                _trig_p = (dE_x > _de_p) | (dP_x > _dp_p)
                _mask_t_p = torch.zeros(
                    x_tf.shape[0], x_tf.shape[1], T,
                    dtype=torch.bool, device=x_tf.device,
                )
                _mask_t_p[..., :-1] |= _trig_p
                _mask_t_p[..., 1:] |= _trig_p
                _mr_p = float(_mask_t_p.float().mean().cpu().item())
                # coverage at this p threshold
                _de_p_safe = _de_p if _de_p > 1e-15 else 1e-15
                _dp_p_safe = _dp_p if _dp_p > 1e-15 else 1e-15
                _s_trans_p = torch.maximum(dE_x / _de_p_safe, dP_x / _dp_p_safe)
                _s_frame_p = torch.zeros_like(_s_frame)
                _s_frame_p[..., :-1] = torch.maximum(_s_frame_p[..., :-1], _s_trans_p)
                _s_frame_p[..., 1:] = torch.maximum(_s_frame_p[..., 1:], _s_trans_p)
                _cdn_p = float(_s_frame_p.sum().cpu().item()) + 1e-12
                _cnm_p = float((_s_frame_p * _mask_t_p.float()).sum().cpu().item())
                _p_table[_p] = (_mr_p, _cnm_p / _cdn_p)
            self._last_mask_p_table = _p_table
            self._last_mask_min_rate_cov95 = _p_table[0.95][0]
            self._last_mask_min_rate_cov99 = _p_table[0.99][0]
            self._last_mask_min_rate_cov995 = _p_table[0.995][0]
        else:
            g_eff = g_local
            self._last_trigger_mask_t = None
            self._last_mask_rate = 1.0
            self._last_mask_trig_rate = 0.0
            self._last_mask_delta_E = float("nan")
            self._last_mask_delta_P = float("nan")
            self._last_mask_cov = float("nan")
            self._last_mask_min_rate_cov95 = float("nan")
            self._last_mask_min_rate_cov99 = float("nan")
            self._last_mask_min_rate_cov995 = float("nan")
            self._last_mask_p_table = {}

        # keep effective gate for debugging (reflects actual decomposition used)
        self._last_gate = g_eff
        n_tf = g_eff * x_tf
        r_tf = x_tf - n_tf
        self._last_r_tf = r_tf
        length = batch_x.shape[1]
        residual = self.stft.inverse(r_tf, length=length)
        n_time = self.stft.inverse(n_tf, length=length)

        # Compute gate statistics on effective gate
        self._compute_gate_stats(g_eff)

        pred_n_time = None
        pred_n_tf = None
        if self.n_tf_predictor is not None:
            # Prepare predictor input features
            if self.pred_input == "n_tf_x_tf":
                # Concatenate n_tf and x_tf as real/imag features
                # n_tf: (B, C, F, T) complex -> (B, C, F, T, 2) real
                # x_tf: (B, C, F, T) complex -> (B, C, F, T, 2) real
                # concat: (B, C, F, T, 4) = [n_real, n_imag, x_real, x_imag]
                n_ri = torch.view_as_real(n_tf)
                x_ri = torch.view_as_real(x_tf)
                features = torch.cat([n_ri, x_ri], dim=-1)  # (B, C, F, T, 4)
            else:
                # Only n_tf: (B, C, F, T, 2)
                features = torch.view_as_real(n_tf)
            
            # Pass features to predictor: (B, C, F, T, feat_dim) -> complex (B, C, F, out_T)
            pred_n_tf = self.n_tf_predictor(features)
            pred_n_time = self.stft.inverse(pred_n_tf, length=self.pred_len)

        state = LocalTFNormState(
            x_tf=x_tf,
            n_tf=n_tf,
            g_local=g_local,
            n_time=n_time,
            length=length,
            pred_n_time=pred_n_time,
            pred_n_tf=pred_n_tf,
            mean=inst_mean,
            std=inst_std,
        )
        self._last_state = state
        self._last_residual = residual

        # Extended diagnostic caches (all detached)
        self._last_x_time = batch_x.detach()
        self._last_x_tf = x_tf.detach()
        self._last_n_tf = n_tf.detach()
        self._last_n_time = n_time.detach()
        self._last_r_time = residual.detach()
        self._last_inst_mean = inst_mean.detach() if inst_mean is not None else None
        self._last_inst_std = inst_std.detach() if inst_std is not None else None
        self._last_pred_n_tf = pred_n_tf.detach() if pred_n_tf is not None else None
        self._last_pred_n_time = pred_n_time.detach() if pred_n_time is not None else None

        if return_state:
            return residual, state
        return residual

    def denormalize(
        self, batch_x: torch.Tensor, state: Optional[LocalTFNormState] = None
    ) -> torch.Tensor:
        if state is None:
            state = self._last_state
        if state is None:
            raise RuntimeError("LocalTFNorm denormalize requires a stored state.")
        target_len = batch_x.shape[1]
        if state.pred_n_time is not None and target_len == state.pred_n_time.shape[1]:
            self._denorm_used_pred += 1
            result = batch_x + state.pred_n_time
        elif target_len == state.length:
            self._denorm_used_input += 1
            result = batch_x + state.n_time
        else:
            self._denorm_used_extrap += 1
            n_time = self._extrapolate_n_time(state.n_time, target_len)
            result = batch_x + n_time
        # Reverse RevIN: restore instance mean and std
        if state.mean is not None and state.std is not None:
            result = result * state.std + state.mean
        return result

    def loss(self, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Stationarity loss computed in the TF domain from the last normalize() call.

        L_stat = lambda_E * L_E + lambda_P * L_P

          P    = |r_tf|^2                      (B, C, F, T)  power spectrum of TF residual
          E    = mean_F(P)                     (B, C, T)     per-frame energy
          logE = log(E + eps_E)                (B, C, T)     log-energy
          p    = P / (sum_F(P) + eps)          (B, C, F, T)  normalised spectral shape
          dE   = |logE[...,1:] - logE[...,:-1]|              (B, C, T-1)  log-energy step
          dP   = mean_F(|p[...,:,1:] - p[...,:,:-1]|)        (B, C, T-1)  shape step
          L_E  = mean(relu(dE - delta_E))      margin-triggered log-energy TV
          L_P  = mean(relu(dP - delta_P))      margin-triggered shape TV
        """
        device = self._device()
        zero = torch.tensor(0.0, device=device)

        if self._last_r_tf is None:
            self._last_L_E = 0.0
            self._last_L_P = 0.0
            self._last_stationarity_loss = 0.0
            return zero

        r_tf = self._last_r_tf  # (B, C, F, T) complex
        eps = 1e-8

        P = r_tf.abs() ** 2  # (B, C, F, T)

        # Per-frame energy: mean over frequency bins
        E = P.mean(dim=2)  # (B, C, T)

        # Normalised spectral shape
        p = P / (P.sum(dim=2, keepdim=True) + eps)  # (B, C, F, T)

        # Log-energy margin-triggered TV along time
        logE = torch.log(E + self.eps_E)
        dE = torch.abs(logE[..., 1:] - logE[..., :-1])          # (B, C, T-1)
        L_E = torch.mean(torch.relu(dE - self.delta_E))

        # Spectral-shape margin-triggered TV along time (averaged over freq first)
        dP = torch.abs(p[..., :, 1:] - p[..., :, :-1]).mean(dim=2)  # (B, C, T-1)
        L_P = torch.mean(torch.relu(dP - self.delta_P))

        # Cache per-transition tensors for diagnostics
        self._last_dE = dE.detach()
        self._last_dP = dP.detach()

        L_stat = self.lambda_E * L_E + self.lambda_P * L_P

        # Cache scalar values for logging (detached)
        self._last_L_E = float(L_E.detach().cpu().item())
        self._last_L_P = float(L_P.detach().cpu().item())
        self._last_stationarity_loss = float(L_stat.detach().cpu().item())

        return L_stat

    def loss_with_target(self, true: torch.Tensor) -> torch.Tensor:
        """Stationarity loss only — no predictor supervision."""
        return self.loss()

    def pred_supervision_loss(self, true: torch.Tensor) -> torch.Tensor:
        """Prediction supervision loss: MSE(pred_n_time, oracle_n_future).

        The oracle is derived via teacher extraction: concatenate the cached
        pre-RevIN input window with `true`, apply the same normalization and
        gate+trigger_mask pipeline via _extract_n_time, and take the last
        pred_len frames as the oracle target.

        Args:
            true: (B, pred_len, C) ground truth in the same scale as the
                  original input batch_x (pre-RevIN).

        Returns:
            Scalar MSE tensor (for backprop); also caches the float value in
            self._last_pred_sup_loss.
        """
        state = self._last_state
        if state is None or state.pred_n_time is None:
            self._last_pred_sup_loss = 0.0
            return torch.tensor(0.0, device=self._device())

        x_raw = self._last_x_raw  # (B, T, C) pre-RevIN
        if x_raw is None:
            self._last_pred_sup_loss = 0.0
            return torch.tensor(0.0, device=self._device())

        dev = self._device()
        true_dev = true.to(dev)
        x_full = torch.cat([x_raw, true_dev], dim=1)  # (B, T+pred_len, C)

        # Save and restore gate._last_logits to avoid polluting debug state
        gate = getattr(self, "gate", None)
        saved_logits = gate._last_logits if gate is not None else None

        with torch.no_grad():
            _, oracle_n_full = self._extract_n_time(
                x_full, mean=state.mean, std=state.std
            )
            oracle_n_future = oracle_n_full[:, -self.pred_len:, :]  # (B, pred_len, C)

        if gate is not None and saved_logits is not None:
            gate._last_logits = saved_logits

        pred_n_time = state.pred_n_time  # (B, pred_len, C)
        if pred_n_time.shape != oracle_n_future.shape:
            self._last_pred_sup_loss = 0.0
            return torch.tensor(0.0, device=dev)

        loss = torch.nn.functional.mse_loss(pred_n_time, oracle_n_future)
        self._last_pred_sup_loss = float(loss.detach().cpu().item())
        return loss

    def get_last_mask_coverage_stats(self) -> dict:
        """Return last computed trigger-mask coverage statistics."""
        return {
            "cov": self._last_mask_cov,
            "min_rate_cov95": self._last_mask_min_rate_cov95,
            "min_rate_cov99": self._last_mask_min_rate_cov99,
            "min_rate_cov995": self._last_mask_min_rate_cov995,
            "p_table": self._last_mask_p_table,
        }

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _extract_n_time(
        self,
        batch_x: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract n_tf and n_time with the same gate+trigger_mask logic as normalize().

        In teacher path (called from pred_supervision_loss) no diagnostic caches
        are updated; gate._last_logits is saved and restored by the caller.
        """
        if mean is not None and std is not None:
            batch_x = (batch_x - mean) / std
        x_tf = self.stft(batch_x)
        g_local = self._make_gate(x_tf)

        # Apply trigger mask if enabled (consistent with normalize(), no cache updates)
        if self.trigger_mask:
            with torch.no_grad():
                dE_x, dP_x = self._dE_dP_from_tf(x_tf)
            de = self.delta_E_mask if self.delta_E_mask > 0.0 else self.delta_E
            dp = self.delta_P_mask if self.delta_P_mask > 0.0 else self.delta_P
            if de > 0.0 and dp > 0.0:
                trig = (dE_x > de) | (dP_x > dp)
                T = x_tf.shape[-1]
                mask_t = torch.zeros(
                    x_tf.shape[0], x_tf.shape[1], T,
                    dtype=torch.bool, device=x_tf.device,
                )
                mask_t[..., :-1] |= trig
                mask_t[..., 1:] |= trig
                mask_f = mask_t.unsqueeze(2).float()
                g_eff = g_local * mask_f
            else:
                g_eff = g_local
        else:
            g_eff = g_local

        n_tf = g_eff * x_tf
        n_time = self.stft.inverse(n_tf, length=batch_x.shape[1])
        return n_tf, n_time

    @torch.no_grad()
    def extract_n_time_only(self, x_time: torch.Tensor) -> torch.Tensor:
        """Apply RevIN + STFT + gate + iSTFT and return n_time.

        Does NOT write any _last_* cache fields.  Safe to call during debug
        collection without disturbing the training-batch state.
        """
        if self.use_instance_norm:
            inst_mean = x_time.mean(dim=1, keepdim=True)
            inst_std = x_time.std(dim=1, keepdim=True) + 1e-8
            x = (x_time - inst_mean) / inst_std
        else:
            x = x_time
        x_tf = self.stft(x)
        g_local = self._make_gate(x_tf)
        n_tf = g_local * x_tf
        return self.stft.inverse(n_tf, length=x.shape[1])

    def _extrapolate_n_time(self, n_time: torch.Tensor, target_len: int) -> torch.Tensor:
        # n_time: (B, T, C)
        source_len = n_time.shape[1]
        if target_len <= source_len:
            return n_time[:, :target_len, :]

        if self.future_mode == "repeat_last":
            last = n_time[:, -1:, :]
            repeat = last.expand(-1, target_len, -1)
            return repeat

        if self.future_mode == "linear":
            if source_len < 2:
                last = n_time[:, -1:, :]
                return last.expand(-1, target_len, -1)
            last = n_time[:, -1:, :]
            prev = n_time[:, -2:-1, :]
            slope = last - prev
            steps = (
                torch.arange(target_len, device=n_time.device, dtype=n_time.dtype)
                .view(1, -1, 1)
            )
            return last + slope * steps

        raise ValueError(f"Unsupported future_mode: {self.future_mode}")

    def forward(
        self,
        batch_x: torch.Tensor,
        mode: str = "n",
        state: Optional[LocalTFNormState] = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LocalTFNormState]:
        if mode == "n":
            return self.normalize(batch_x, return_state=return_state)
        if mode == "d":
            return self.denormalize(batch_x, state=state)
        raise ValueError(f"Unsupported mode: {mode}")
