from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import math

from .stft import STFT
from .gate import LocalTFGate
from .losses import residual_stationarity_loss


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

    def normalize(
        self, batch_x: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, LocalTFNormState]:
        # RevIN-style instance normalization: remove per-sample mean and std
        inst_mean = inst_std = None
        if self.use_instance_norm:
            inst_mean = batch_x.mean(dim=1, keepdim=True)          # (B, 1, C)
            inst_std = batch_x.std(dim=1, keepdim=True) + 1e-8     # (B, 1, C)
            batch_x = (batch_x - inst_mean) / inst_std

        x_tf = self.stft(batch_x)
        g_local = self._make_gate(x_tf)
        # keep last gate for optional regularizers / debugging
        self._last_gate = g_local
        n_tf = g_local * x_tf
        r_tf = x_tf - n_tf
        length = batch_x.shape[1]
        residual = self.stft.inverse(r_tf, length=length)
        n_time = self.stft.inverse(n_tf, length=length)

        # Compute gate statistics
        self._compute_gate_stats(g_local)

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
            result = batch_x + state.pred_n_time
        elif target_len == state.length:
            result = batch_x + state.n_time
        else:
            n_time = self._extrapolate_n_time(state.n_time, target_len)
            result = batch_x + n_time
        # Reverse RevIN: restore instance mean and std
        if state.mean is not None and state.std is not None:
            result = result * state.std + state.mean
        return result

    def loss(self, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self._device())
        if self.stationarity_loss_weight > 0 and self._last_residual is not None:
            loss = loss + residual_stationarity_loss(
                self._last_residual, num_chunks=self.stationarity_chunks
            ) * self.stationarity_loss_weight
        if self._last_state is not None:
            g = self._last_state.g_local
            if self.gate_smooth_weight > 0:
                # Frequency-axis smoothness: penalise adjacent-freq differences
                smooth = torch.mean(torch.abs(g[:, :, 1:, :] - g[:, :, :-1, :]))
                loss = loss + smooth * self.gate_smooth_weight
            if self.gate_temporal_smooth_weight > 0:
                # Time-axis smoothness: penalise adjacent-time differences
                t_smooth = torch.mean(torch.abs(g[:, :, :, 1:] - g[:, :, :, :-1]))
                loss = loss + t_smooth * self.gate_temporal_smooth_weight
            # Ratio regularisation is meaningless when the gate already enforces a
            # hard budget via softmax/sigmoid_budget; skip it to avoid conflicting
            # gradients.
            budget_modes = {"softmax_budget", "sigmoid_budget"}
            if self.gate_ratio_weight > 0 and self.gate_mode not in budget_modes:
                ratio = g.mean()
                ratio_loss = (ratio - self.gate_ratio_target) ** 2
                loss = loss + ratio_loss * self.gate_ratio_weight
            # Optional entropy regularizer on gate (encourage peaky distribution)
            if self.gate_entropy_weight is not None and self.gate_entropy_weight > 0:
                gate = None
                if self._last_state is not None and getattr(self._last_state, 'g_local', None) is not None:
                    gate = self._last_state.g_local
                elif getattr(self, '_last_gate', None) is not None:
                    gate = self._last_gate
                if gate is not None:
                    try:
                        eps = 1e-12
                        # normalize gate into probabilities over frequency (no softmax)
                        p = gate / (gate.sum(dim=2, keepdim=True) + eps)
                        ent = -torch.sum(p * torch.log(p + eps), dim=2)  # (B,C,T)
                        logF = math.log(max(1, gate.shape[2]))
                        ent_norm = ent / (logF + 1e-12)
                        ent_mean = ent_norm.mean()
                        loss = loss + float(self.gate_entropy_weight) * ent_mean
                    except Exception:
                        pass
        return loss

    def loss_with_target(self, true: torch.Tensor) -> torch.Tensor:
        loss = self.loss()
        if (
            self.n_tf_predictor is None
            or self._last_state is None
            or self._last_state.pred_n_time is None
        ):
            return loss
        # Use no_grad so the gate does NOT receive gradients from the predictor
        # target computation.  This decouples gate training from predictor training
        # and prevents the "moving target" instability where the gate and predictor
        # co-adapt in a circular way.
        with torch.no_grad():
            _, true_n_time = self._extract_n_time(
                true,
                mean=self._last_state.mean,
                std=self._last_state.std,
            )
        if true_n_time.shape[1] != self._last_state.pred_n_time.shape[1]:
            return loss
        pred_loss = nn.functional.mse_loss(self._last_state.pred_n_time, true_n_time)
        return loss + pred_loss * self.pred_loss_weight

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _extract_n_time(
        self,
        batch_x: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mean is not None and std is not None:
            batch_x = (batch_x - mean) / std
        x_tf = self.stft(batch_x)
        g_local = self._make_gate(x_tf)
        n_tf = g_local * x_tf
        n_time = self.stft.inverse(n_tf, length=batch_x.shape[1])
        return n_tf, n_time

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
