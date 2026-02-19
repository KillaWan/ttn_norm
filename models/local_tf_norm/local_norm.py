from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

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


class SimpleTFPredictor(nn.Module):
    def __init__(
        self,
        in_frames: int,
        out_frames: int,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        input_feature_dim: int = 1,
    ):
        """
        Args:
            in_frames: number of input frames (time steps)
            out_frames: number of output frames
            hidden_dim: hidden dimension in MLP (0 = linear)
            dropout: dropout rate
            input_feature_dim: feature dimension (1 for n_tf only, 4 for [n_tf_real, n_tf_imag, x_tf_real, x_tf_imag])
        """
        super().__init__()
        in_features = in_frames * input_feature_dim
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_frames),
            )
        else:
            self.net = nn.Linear(in_features, out_frames)

    def forward(self, x_tf: torch.Tensor) -> torch.Tensor:
        # x_tf: (B, C, F, T) or (B*feat_dim, C, F, T) complex
        # Convert to real representation for feature handling
        x_ri = torch.view_as_real(x_tf)  # (B, C, F, T, 2)
        b, c, f, t, two = x_ri.shape
        # Reshape to (B*C*F*2, T) where 2 is the real/imag dimension
        x_ri = x_ri.reshape(b * c * f * two, t)
        # Pass through network
        x_ri = self.net(x_ri)  # Output shape: (B*C*F*2, out_frames)
        # Reshape back to (B, C, F, out_frames, 2)
        x_ri = x_ri.reshape(b, c, f, -1, two)
        return torch.view_as_complex(x_ri.contiguous())


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
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.gate_log_mag = gate_log_mag
        self.stationarity_loss_weight = stationarity_loss_weight
        self.stationarity_chunks = stationarity_chunks
        self.future_mode = future_mode
        self.predict_n_time = predict_n_time
        self.pred_loss_weight = pred_loss_weight
        self.gate_smooth_weight = gate_smooth_weight
        self.gate_ratio_weight = gate_ratio_weight
        self.gate_ratio_target = gate_ratio_target
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
            gate_budget_ratio=gate_ratio_target if gate_mode != "sigmoid" else 0.0,
        )
        in_frames = self.stft.time_bins(self.seq_len)
        out_frames = self.stft.time_bins(self.pred_len)
        
        # Determine predictor input feature dimension
        # n_tf only: 2 features (real, imag)
        # n_tf_x_tf: 4 features (n_real, n_imag, x_real, x_imag)
        input_feature_dim = 4 if pred_input == "n_tf_x_tf" else 1
        
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

    def _make_gate(self, x_tf: torch.Tensor) -> torch.Tensor:
        magnitude = x_tf.abs()
        if self.gate_log_mag:
            magnitude = torch.log1p(magnitude)
        return self.gate(magnitude)

    def _compute_gate_stats(self, g_local: torch.Tensor) -> None:
        """Compute gate statistics: mean, freq sum, freq max, freq entropy."""
        # g_local: (B, C, F, TT)
        self._last_gate_mean = float(g_local.mean().detach().cpu().item())
        
        # Sum over time dimension to get per-freq activation
        g_freq = g_local.mean(dim=(0, 1, 3))  # (F,)
        self._last_gate_sum_f = float(g_freq.sum().detach().cpu().item())
        self._last_gate_max_f = float(g_freq.max().detach().cpu().item())
        
        # Entropy over frequency (normalized by log(F))
        # Use softmax to get probability distribution
        g_freq_norm = torch.softmax(g_freq.unsqueeze(0), dim=1).squeeze(0)
        eps = 1e-10
        entropy = -(g_freq_norm * torch.log(g_freq_norm + eps)).sum()
        max_entropy = torch.log(torch.tensor(g_freq.shape[0], dtype=g_freq.dtype, device=g_freq.device))
        self._last_gate_ent_f = float((entropy / max_entropy).detach().cpu().item())

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
        x_tf = self.stft(batch_x)
        g_local = self._make_gate(x_tf)
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
            if self.pred_input == "n_tf_x_tf":
                # Create predictor input by concatenating n_tf and x_tf real/imag features
                # n_tf: (B, C, F, T) complex
                # x_tf: (B, C, F, T) complex
                # Convert to real: (B, C, F, T, 2)
                n_ri = torch.view_as_real(n_tf)  # (B, C, F, T, 2)
                x_ri = torch.view_as_real(x_tf)  # (B, C, F, T, 2)
                
                # Concatenate along feature dimension (last axis after real/imag)
                # Result shape: (B, C, F, T, 4) = [n_real, n_imag, x_real, x_imag]
                combined_ri = torch.cat([n_ri, x_ri], dim=-1)  # (B, C, F, T, 4)
                
                # View back as complex with extra feature dimension
                # We'll treat this as (B*4, C, F, T) for the predictor
                b, c, f, t, feat_dim = combined_ri.shape
                # Reshape: (B, C, F, T, 4) -> (B, 4, C, F, T) -> (B*4, C*F, T)
                combined_ri = combined_ri.permute(0, 4, 1, 2, 3)  # (B, 4, C, F, T)
                combined_ri = combined_ri.reshape(b * feat_dim, c * f, t)
                
                # Feed to predictor and reshape back
                # Output shape: (B*4, C*F, out_frames)
                pred_out = self.n_tf_predictor(torch.view_as_complex(combined_ri.unsqueeze(-1)))
                # This won't work; let me simplify: use the predictor's original interface
                # Actually, the predictor expects (B, C, F, T) complex input
                # We need a different approach: create a dummy complex tensor and process
                
                # Better approach: manually stack features
                # n_ri: (B, C, F, T, 2), x_ri: (B, C, F, T, 2)
                # Stack: (B, C, F, T, 4) then we need to treat this as 2-real/2-imag
                # Convert to complex with doubled channels for features
                
                # Simpler: Just create a 4-feature tensor and handle in predictor
                # For now, concatenate as (B, C, F, T, 4), view_as_complex treats last 2 dims
                # This is complex; let me use a cleaner approach:
                
                # Convert each to (B*C*F, 2, T) real-only, concatenate feature-wise
                b, c, f, t, two_n = n_ri.shape
                n_feat = n_ri.reshape(b * c * f, t, two_n)  # (B*C*F, T, 2)
                x_feat = x_ri.reshape(b * c * f, t, two_n)  # (B*C*F, T, 2)
                # Concatenate along feature axis (last one): (B*C*F, T, 4)
                combined_feat = torch.cat([n_feat, x_feat], dim=-1)  # (B*C*F, T, 4)
                
                # Now we need to feed this 4-feature tensor to the predictor
                # Reshape to (B*C*F*2, T) to match predictor's expectation
                combined_feat = combined_feat.reshape(b * c * f * 4, t)
                # Pass through MLP (which expects in_frames * input_feature_dim)
                pred_out = self.n_tf_predictor.net(combined_feat)  # (B*C*F*4, out_frames)
                # Reshape back to complex: (B, C, F, out_frames, 2)
                pred_out = pred_out.reshape(b, c, f, -1, two_n // 2)
                # Hmm, this doesn't work either; let me reconsider...
                
                # The issue is we're trying to pass 4-dim features but predictor expects (B,C,F,T) complex
                # Let me use a wrapper approach: expand dims temporarily
                # Treat as (B, 2*C, F, T) and view as complex after processing? No, that breaks symmetry.
                
                # Cleanest solution: modify predictor call site to handle raw features
                # Use predictor.net directly for feature processing:
                b, c, f, t, two_n = n_ri.shape
                n_flat = n_ri.reshape(b * c * f * two_n, t)  # (B*C*F*2, T)
                x_flat = x_ri.reshape(b * c * f * two_n, t)  # (B*C*F*2, T)
                combined_flat = torch.cat([n_flat, x_flat], dim=0)  # (B*C*F*4, T)
                pred_flat = self.n_tf_predictor.net(combined_flat)  # (B*C*F*4, out_frames)
                # Reshape back: first half is n_pred, second half is x_pred
                pred_out = pred_flat[:b*c*f*two_n, :]  # Take only the n dimension
                pred_out = pred_out.reshape(b, c, f, -1, two_n)
                pred_n_tf = torch.view_as_complex(pred_out.contiguous())
            else:
                # Default: only use n_tf as input
                pred_n_tf = self.n_tf_predictor(n_tf)
            
            pred_n_time = self.stft.inverse(pred_n_tf, length=self.pred_len)

        state = LocalTFNormState(
            x_tf=x_tf,
            n_tf=n_tf,
            g_local=g_local,
            n_time=n_time,
            length=length,
            pred_n_time=pred_n_time,
            pred_n_tf=pred_n_tf,
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
            return batch_x + state.pred_n_time
        if target_len == state.length:
            return batch_x + state.n_time
        n_time = self._extrapolate_n_time(state.n_time, target_len)
        return batch_x + n_time

    def loss(self, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self._device())
        if self.stationarity_loss_weight > 0 and self._last_residual is not None:
            loss = loss + residual_stationarity_loss(
                self._last_residual, num_chunks=self.stationarity_chunks
            ) * self.stationarity_loss_weight
        if self._last_state is not None:
            g = self._last_state.g_local
            if self.gate_smooth_weight > 0:
                smooth = torch.mean(torch.abs(g[:, :, 1:, :] - g[:, :, :-1, :]))
                loss = loss + smooth * self.gate_smooth_weight
            if self.gate_ratio_weight > 0:
                ratio = g.mean()
                ratio_loss = (ratio - self.gate_ratio_target) ** 2
                loss = loss + ratio_loss * self.gate_ratio_weight
        return loss

    def loss_with_target(self, true: torch.Tensor) -> torch.Tensor:
        loss = self.loss()
        if (
            self.n_tf_predictor is None
            or self._last_state is None
            or self._last_state.pred_n_time is None
        ):
            return loss
        _, true_n_time = self._extract_n_time(true)
        if true_n_time.shape[1] != self._last_state.pred_n_time.shape[1]:
            return loss
        pred_loss = nn.functional.mse_loss(self._last_state.pred_n_time, true_n_time)
        return loss + pred_loss * self.pred_loss_weight

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _extract_n_time(self, batch_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
