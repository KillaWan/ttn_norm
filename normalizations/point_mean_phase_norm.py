"""PointMeanPhaseNorm — pointwise mean + pointwise phase velocity normalization.

This norm combines per-channel centered moving average (mean tracking) with
phase velocity extraction and prediction. No patch-wise operations, no std
main chain, no frequency-domain rotations — purely temporal-domain processing.

Design principles:
  - normalize() computes mu_hist_seq, z_hist (residual), omega_hist_seq (phase velocity),
    and predicts future mu and omega sequences.
  - denormalize() reconstructs via phase-based temporal sampling + mean addition.
  - compute_base_aux_loss() supervises mean and phase predictions independently.
  - All convolutions use replication padding for boundary handling.
  - No STFT, iSTFT, overlap-add, Hilbert, wavelets, or multiband decomposition.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointMeanPhaseMeanPredictor(nn.Module):
    """Light-weight channel-wise MLP predicting future mean sequence.
    
    Input: mu_hist_seq (B, L_hist, C) and z_hist (B, L_hist, C).
    Output: mu_fut_seq_hat (B, H, C).
    
    Simple 2-layer MLP per channel, input projected from (L_hist+L_hist) → hidden → H.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.hidden_dim = hidden_dim

        # Concatenate mu_hist_seq and z_hist: (B, L_hist, 2*C)
        # Flatten to (B*C, 2*L_hist) for per-channel processing
        input_size = 2 * seq_len
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, pred_len)

    def forward(
        self,
        mu_hist_seq: torch.Tensor,  # (B, L_hist, C)
        z_hist: torch.Tensor,        # (B, L_hist, C)
    ) -> torch.Tensor:
        """Return (B, H, C) future mean sequence."""
        B, L_hist, C = mu_hist_seq.shape

        # Pair each channel's own two L-length sequences: (B, C, L_hist) cat → (B, C, 2L)
        a = mu_hist_seq.permute(0, 2, 1)      # (B, C, L_hist)
        b = z_hist.permute(0, 2, 1)            # (B, C, L_hist)
        x = torch.cat([a, b], dim=-1)          # (B, C, 2*L_hist)
        x = x.reshape(B * C, L_hist * 2)      # (B*C, 2*L_hist)

        # Forward through MLP
        x = self.fc1(x)  # (B*C, hidden_dim)
        x = self.act(x)
        x = self.fc2(x)  # (B*C, pred_len)

        # Reshape back: (B, C, pred_len) → (B, pred_len, C)
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return x


class PointMeanPhasePhasePredictor(nn.Module):
    """Light-weight channel-wise MLP predicting future phase velocity sequence.
    
    Input: omega_hist_seq (B, L_hist, C) and z_hist (B, L_hist, C).
    Output: omega_fut_seq_hat_raw (B, H, C), then bounded to omega_max * tanh.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.hidden_dim = hidden_dim

        input_size = 2 * seq_len
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, pred_len)

    def forward(
        self,
        omega_hist_seq: torch.Tensor,  # (B, L_hist, C)
        z_hist: torch.Tensor,           # (B, L_hist, C)
        omega_max: float,
    ) -> torch.Tensor:
        """Return (B, H, C) future phase velocity, bounded by omega_max."""
        B, L_hist, C = omega_hist_seq.shape

        # Pair each channel's own two L-length sequences: (B, C, L_hist) cat → (B, C, 2L)
        a = omega_hist_seq.permute(0, 2, 1)    # (B, C, L_hist)
        b = z_hist.permute(0, 2, 1)             # (B, C, L_hist)
        x = torch.cat([a, b], dim=-1)           # (B, C, 2*L_hist)
        x = x.reshape(B * C, L_hist * 2)       # (B*C, 2*L_hist)

        # Forward through MLP
        x = self.fc1(x)  # (B*C, hidden_dim)
        x = self.act(x)
        x = self.fc2(x)  # (B*C, pred_len)

        # Reshape back: (B, C, pred_len) → (B, pred_len, C)
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # Bound to [-omega_max, omega_max] via tanh
        x = omega_max * torch.tanh(x)
        return x


class PointMeanPhaseNorm(nn.Module):
    """Pointwise mean + pointwise phase normalization.
    
    Args:
        seq_len: Input history length.
        pred_len: Prediction horizon.
        enc_in: Number of input channels.
        pmp_mean_kernel: Size of centered-moving-average kernel (must be odd).
        pmp_smooth_kernel: Size of symmetric smoothing kernel for phase extraction (must be odd).
        pmp_diff_kernel: Size of first-difference kernel (must be odd).
        pmp_omega_max: Upper bound on phase velocity (rad/sample).
        pmp_shift_scale: Scale factor for temporal displacement from accumulated phase.
        pmp_phase_loss_weight: Weight for phase loss in aux loss.
        pmp_phase_smooth_weight: Weight for phase smoothness loss in aux loss.
        pmp_hidden_dim: Hidden dimension for predictors (default 64).
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        pmp_mean_kernel: int = 3,
        pmp_smooth_kernel: int = 3,
        pmp_diff_kernel: int = 3,
        pmp_omega_max: float = 1.0,
        pmp_shift_scale: float = 0.1,
        pmp_phase_loss_weight: float = 1.0,
        pmp_phase_smooth_weight: float = 0.1,
        pmp_hidden_dim: int = 64,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.eps = float(eps)
        
        # Validate kernel sizes are odd
        for k, name in [
            (pmp_mean_kernel, "pmp_mean_kernel"),
            (pmp_smooth_kernel, "pmp_smooth_kernel"),
            (pmp_diff_kernel, "pmp_diff_kernel"),
        ]:
            if k % 2 == 0:
                raise ValueError(f"{name} must be odd, got {k}")
        
        self.pmp_mean_kernel = int(pmp_mean_kernel)
        self.pmp_smooth_kernel = int(pmp_smooth_kernel)
        self.pmp_diff_kernel = int(pmp_diff_kernel)
        self.pmp_omega_max = float(pmp_omega_max)
        self.pmp_shift_scale = float(pmp_shift_scale)
        self.pmp_phase_loss_weight = float(pmp_phase_loss_weight)
        self.pmp_phase_smooth_weight = float(pmp_phase_smooth_weight)

        # Create convolution kernels as fixed (non-learnable) depthwise kernels
        # smooth kernel: symmetric Gaussian-like, e.g. [1, 2, 1]/4
        self._build_kernels()
        
        # Predictors
        self.mean_predictor = PointMeanPhaseMeanPredictor(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            hidden_dim=pmp_hidden_dim,
        )
        
        self.phase_predictor = PointMeanPhasePhasePredictor(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            hidden_dim=pmp_hidden_dim,
        )
        
        # Smoothing kernel for tau_fut_seq (used in denormalize)
        # Simple 1-2-1 normalized kernel
        self.denorm_smooth_size = 3
        self._reset_cache()

    def _build_kernels(self) -> None:
        """Build fixed depthwise convolution kernels."""
        # Smooth kernel: symmetric [1, 2, 1] / 4
        ks = self.pmp_smooth_kernel
        smooth = torch.ones(ks)
        if ks >= 3:
            smooth[ks // 2] = 2.0
        smooth_kernel = smooth / smooth.sum()
        self.register_buffer("smooth_kernel", smooth_kernel)
        
        # Diff kernel: [-1, 0, 1] / 2 for first-order difference
        # Size must match pmp_diff_kernel
        kd = self.pmp_diff_kernel
        diff = torch.zeros(kd)
        diff[0] = -1.0
        diff[-1] = 1.0
        diff_kernel = diff / 2.0
        self.register_buffer("diff_kernel", diff_kernel)
        
        # Mean kernel for centered moving average: uniform window
        km = self.pmp_mean_kernel
        mean_kernel = torch.ones(km) / km
        self.register_buffer("mean_kernel", mean_kernel)

    def _reset_cache(self) -> None:
        """Clear all cached tensors from normalize()."""
        # History caches
        self._mu_hist_seq: Optional[torch.Tensor] = None
        self._z_hist: Optional[torch.Tensor] = None
        self._omega_hist_seq: Optional[torch.Tensor] = None
        self._amp_hist_seq: Optional[torch.Tensor] = None
        
        # Future predictions
        self._mu_fut_seq_hat: Optional[torch.Tensor] = None
        self._omega_fut_seq_hat: Optional[torch.Tensor] = None

        # denormalize() state (phase displacement fields)
        self._tau_fut_seq: Optional[torch.Tensor] = None
        self._tau_fut_seq_smooth: Optional[torch.Tensor] = None

        # Diagnostics
        self._last_mu_loss: float = 0.0
        self._last_phase_loss: float = 0.0
        self._last_smooth_loss: float = 0.0
        self._last_base_aux_loss: float = 0.0
        self._last_aux_total: float = 0.0

    def _centered_moving_average(self, x: torch.Tensor) -> torch.Tensor:
        """Compute centered moving average with replication padding.
        
        Args:
            x: (B, L, C) tensor
        Returns:
            mu: (B, L, C) centered moving average
        """
        B, L, C = x.shape
        kernel_pad = self.pmp_mean_kernel // 2
        
        # Reshape for conv1d: (B*C, 1, L)
        x_reshaped = x.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # Pad with replication
        x_padded = F.pad(x_reshaped, (kernel_pad, kernel_pad), mode="replicate")
        
        # Apply 1D conv with kernel (depthwise-like)
        kernel = self.mean_kernel.view(1, 1, -1)
        mu_flat = F.conv1d(x_padded, kernel, padding=0)  # (B*C, 1, L)
        
        # Reshape back: (B, C, L) → (B, L, C)
        mu = mu_flat.reshape(B, C, L).permute(0, 2, 1)
        return mu

    def _extract_phase_features(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract phase features via smooth and diff convolutions.
        
        Args:
            z: (B, L, C) residual after mean removal
        Returns:
            u: (B, L, C) smoothed signal
            v: (B, L, C) first-order difference
            amp: (B, L, C) amplitude (norm of (u, v))
        """
        B, L, C = z.shape
        
        # Reshape for conv1d: (B*C, 1, L)
        z_reshaped = z.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # Smooth convolution with replication padding
        smooth_pad = self.pmp_smooth_kernel // 2
        z_smooth_padded = F.pad(z_reshaped, (smooth_pad, smooth_pad), mode="replicate")
        smooth_k = self.smooth_kernel.view(1, 1, -1)
        u_flat = F.conv1d(z_smooth_padded, smooth_k, padding=0)  # (B*C, 1, L)
        u = u_flat.reshape(B, C, L).permute(0, 2, 1)  # (B, L, C)
        
        # Diff convolution with replication padding
        diff_pad = self.pmp_diff_kernel // 2
        z_diff_padded = F.pad(z_reshaped, (diff_pad, diff_pad), mode="replicate")
        diff_k = self.diff_kernel.view(1, 1, -1)
        v_flat = F.conv1d(z_diff_padded, diff_k, padding=0)  # (B*C, 1, L)
        v = v_flat.reshape(B, C, L).permute(0, 2, 1)  # (B, L, C)
        
        # Amplitude
        amp = torch.sqrt(u.pow(2) + v.pow(2) + self.eps)
        
        return u, v, amp

    def _compute_phase_velocity(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Compute pointwise phase velocity omega from (u, v) pairs.
        
        omega_t = atan2(s_t * c_{t-1} - c_t * s_{t-1}, c_t * c_{t-1} + s_t * s_{t-1})
        where c = u/amp, s = v/amp (unit complex representation).
        
        Args:
            u: (B, L, C) cos component
            v: (B, L, C) sin component
        Returns:
            omega: (B, L, C) phase velocity, 0 at t=0
        """
        amp = torch.sqrt(u.pow(2) + v.pow(2) + self.eps)
        c = u / amp  # (B, L, C)
        s = v / amp  # (B, L, C)
        
        # Shift: c_{t-1}, s_{t-1} are t-1 versions; at t=0 use themselves (no change)
        c_prev = torch.roll(c, shifts=1, dims=1)
        s_prev = torch.roll(s, shifts=1, dims=1)
        c_prev[:, 0, :] = c[:, 0, :]  # boundary: use t=0 for t-1
        s_prev[:, 0, :] = s[:, 0, :]
        
        # atan2(s_t * c_{t-1} - c_t * s_{t-1}, c_t * c_{t-1} + s_t * s_{t-1})
        num = s * c_prev - c * s_prev
        denom = c * c_prev + s * s_prev
        omega = torch.atan2(num, denom)  # (B, L, C)
        
        return omega

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input and prepare caches.
        
        Args:
            x: (B, L, C) input time series
        Returns:
            z_hist: (B, L, C) residual (already cached)
        """
        self._reset_cache()
        B, L, C = x.shape
        
        if L != self.seq_len:
            raise ValueError(
                f"PointMeanPhaseNorm expects seq_len={self.seq_len}, got {L}."
            )
        
        # Step 1: Compute mu_hist_seq via centered moving average
        mu_hist_seq = self._centered_moving_average(x)
        self._mu_hist_seq = mu_hist_seq
        
        # Step 2: Residual
        z_hist = x - mu_hist_seq
        self._z_hist = z_hist
        
        # Step 3: Extract phase features
        u, v, amp = self._extract_phase_features(z_hist)
        self._amp_hist_seq = amp
        
        # Step 4: Compute phase velocity
        omega_hist_seq = self._compute_phase_velocity(u, v)
        self._omega_hist_seq = omega_hist_seq
        
        # Step 5: Predict future mean and phase sequences
        mu_fut_seq_hat = self.mean_predictor(mu_hist_seq, z_hist)
        self._mu_fut_seq_hat = mu_fut_seq_hat
        
        omega_fut_seq_hat = self.phase_predictor(
            omega_hist_seq, z_hist, self.pmp_omega_max
        )
        self._omega_fut_seq_hat = omega_fut_seq_hat
        
        return z_hist

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize model output.
        
        Args:
            y_norm: (B, H, C) normalized model output (residual in phase space)
        Returns:
            y_hat: (B, H, C) denormalized output
        """
        if self._mu_fut_seq_hat is None or self._omega_fut_seq_hat is None:
            return y_norm
        
        B, H, C = y_norm.shape
        
        if H != self.pred_len:
            # Non-standard horizon: just add mean
            mu_time = self._mu_fut_seq_hat
            if H > self.pred_len:
                # Extend by repeating last
                ext = mu_time[:, -1:, :].expand(B, H - self.pred_len, C)
                mu_time = torch.cat([mu_time, ext], dim=1)
            elif H < self.pred_len:
                mu_time = mu_time[:, :H, :]
            return y_norm + mu_time
        
        # Accumulate phase velocity into displacement field
        # tau_t = sum_{j<=t}(shift_scale * omega_hat_j)
        omega_fut = self._omega_fut_seq_hat  # (B, H, C)
        tau_fut_seq = torch.cumsum(
            self.pmp_shift_scale * omega_fut, dim=1
        )  # (B, H, C)
        self._tau_fut_seq = tau_fut_seq

        # Smooth tau_fut_seq to reduce jitter
        tau_smooth = self._smooth_displacement(tau_fut_seq)  # (B, H, C)
        self._tau_fut_seq_smooth = tau_smooth
        
        # Perform 1D linear interpolation sampling on y_norm
        # Sampling positions: t + tau_smooth_t, clamped to [0, H-1]
        y_shifted = self._temporal_interpolation_sample(y_norm, tau_smooth)  # (B, H, C)
        
        # Add predicted mean
        y_hat = y_shifted + self._mu_fut_seq_hat
        
        return y_hat

    def _smooth_displacement(self, tau: torch.Tensor) -> torch.Tensor:
        """Apply lightweight smoothing to displacement field.
        
        Args:
            tau: (B, H, C) displacement
        Returns:
            tau_smooth: (B, H, C) smoothed displacement
        """
        B, H, C = tau.shape
        
        # Use simple 1-2-1 kernel
        kernel = torch.tensor([1.0, 2.0, 1.0], device=tau.device) / 4.0
        
        # Reshape: (B*C, 1, H)
        tau_reshaped = tau.permute(0, 2, 1).reshape(B * C, 1, H)
        
        # Pad with replication
        tau_padded = F.pad(tau_reshaped, (1, 1), mode="replicate")
        
        # Convolve
        k = kernel.view(1, 1, 3)
        tau_smooth_flat = F.conv1d(tau_padded, k, padding=0)  # (B*C, 1, H)
        
        # Reshape back: (B, C, H) → (B, H, C)
        tau_smooth = tau_smooth_flat.reshape(B, C, H).permute(0, 2, 1)
        return tau_smooth

    def _temporal_interpolation_sample(
        self, y_norm: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Perform 1D linear interpolation on y_norm at shifted positions.
        
        Args:
            y_norm: (B, H, C) normalized output
            tau: (B, H, C) temporal displacement
        Returns:
            y_shifted: (B, H, C) interpolated output
        """
        B, H, C = y_norm.shape
        device = y_norm.device
        
        # Sampling positions: t + tau, where t ∈ [0, H-1]
        t = torch.arange(H, dtype=y_norm.dtype, device=device)  # (H,)
        t = t.unsqueeze(0).unsqueeze(-1)  # (1, H, 1)
        
        sample_pos = t + tau  # (B, H, C)
        
        # Clamp to [0, H-1]
        sample_pos = torch.clamp(sample_pos, 0, H - 1)
        
        # Linear interpolation
        # sample_pos = floor_idx + frac
        floor_idx = torch.floor(sample_pos).long()  # (B, H, C)
        frac = sample_pos - floor_idx.float()  # (B, H, C)
        
        # Gather y_norm values at floor and ceiling
        floor_idx = torch.clamp(floor_idx, 0, H - 1)
        ceil_idx = torch.clamp(floor_idx + 1, 0, H - 1)
        
        # Gather: (B, H, C) → need to extract along dim=1
        y_floor = torch.gather(
            y_norm, 1, floor_idx.permute(0, 1, 2)
        )  # (B, H, C)
        y_ceil = torch.gather(
            y_norm, 1, ceil_idx.permute(0, 1, 2)
        )  # (B, H, C)
        
        # Linear interpolation
        y_shifted = (1 - frac) * y_floor + frac * y_ceil
        
        return y_shifted

    def compute_base_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Compute base auxiliary loss (mean + phase + smoothness).
        
        Args:
            y_true: (B, H, C) ground truth
        Returns:
            total_loss: scalar loss
        """
        if (
            self._mu_fut_seq_hat is None
            or self._omega_fut_seq_hat is None
            or self._omega_hist_seq is None
        ):
            return torch.tensor(0.0, device=y_true.device)
        
        # Compute oracle stats from y_true
        # Same processing as normalize but on y_true
        mu_fut_seq_true = self._centered_moving_average(y_true)
        z_true = y_true - mu_fut_seq_true
        u_true, v_true, amp_fut_seq_true = self._extract_phase_features(z_true)
        omega_fut_seq_true = self._compute_phase_velocity(u_true, v_true)
        
        # mu_loss: MSE between predicted and oracle mean
        mu_loss = F.mse_loss(self._mu_fut_seq_hat, mu_fut_seq_true)
        self._last_mu_loss = float(mu_loss.detach().item())
        
        # phase_loss: Weighted MSE with amplitude weighting + edge masking
        # Weight by amp / (mean(amp)+eps), clamped to fixed upper limit
        B, H, C = self._omega_fut_seq_hat.shape
        
        # Compute weights
        amp_mean = amp_fut_seq_true.mean(dim=1, keepdim=True)  # (B, 1, C)
        w = amp_fut_seq_true / (amp_mean + self.eps)  # (B, H, C)
        w = torch.clamp(w, 0.0, 2.0)  # fixed upper limit
        
        # Apply edge mask: zero out edge_mask_size points at boundaries
        edge_mask_size = self.pmp_smooth_kernel // 2
        edge_mask = torch.ones(H, device=y_true.device)
        edge_mask[:edge_mask_size] = 0.0
        edge_mask[-edge_mask_size:] = 0.0
        edge_mask = edge_mask.unsqueeze(0).unsqueeze(-1)  # (1, H, 1)
        
        w = w * edge_mask
        
        # Weighted phase loss
        phase_diff = (self._omega_fut_seq_hat - omega_fut_seq_true).pow(2)
        phase_loss = (w * phase_diff).mean()
        self._last_phase_loss = float(phase_loss.detach().item())
        
        # smooth_loss: Adjacent time steps difference squared
        # omega_hat_t - omega_hat_{t-1}
        omega_hat_diff = torch.diff(self._omega_fut_seq_hat, dim=1)  # (B, H-1, C)
        smooth_loss = omega_hat_diff.pow(2).mean()
        self._last_smooth_loss = float(smooth_loss.detach().item())
        
        # Total loss
        base_aux = mu_loss
        total = (
            base_aux
            + self.pmp_phase_loss_weight * phase_loss
            + self.pmp_phase_smooth_weight * smooth_loss
        )
        
        self._last_base_aux_loss = float(base_aux.detach().item())
        self._last_aux_total = float(total.detach().item())
        
        return total

    def compute_total_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Alias for compute_base_aux_loss (no route state for PointMeanPhaseNorm)."""
        return self.compute_base_aux_loss(y_true)

    def get_last_aux_stats(self) -> dict:
        """Return diagnostic stats from last loss computation."""
        return {
            "aux_total": self._last_aux_total,
            "base_aux_loss": self._last_base_aux_loss,
            "mu_loss": self._last_mu_loss,
            "std_loss": 0.0,
            "phase_loss": self._last_phase_loss,
            "smooth_loss": self._last_smooth_loss,
            "route_state_loss": 0.0,
        }

    def parameters_base_predictor(self) -> list:
        """Return base predictor parameters (for Stage 1 optimizer)."""
        return list(self.parameters())

    def freeze_base_predictor(self) -> None:
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_base_predictor(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad_(True)

    def parameters_route_modules(self) -> list:
        """Return route module parameters (none for PointMeanPhaseNorm, pure baseline)."""
        return []

    def freeze_route_modules(self) -> None:
        """Freeze route modules (no-op for PointMeanPhaseNorm, pure baseline)."""
        pass

    def unfreeze_route_modules(self) -> None:
        """Unfreeze route modules (no-op for PointMeanPhaseNorm, pure baseline)."""
        pass

    def get_route_diagnostics(self) -> dict:
        """Return route diagnostics (empty for PointMeanPhaseNorm, pure baseline)."""
        return {}
