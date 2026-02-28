"""TF-Background normalization.

Normalizes a time series in the STFT domain by dividing out a slowly-varying
spectral background estimated from a 2-D smoothed log-power spectrogram.

Interface:
    normalize(x)           -> x_norm          (B, T, C)
    denormalize(y_norm)    -> y               (B, O, C)
    get_last_stats()       -> dict  (scale_min/max/mean, B_std)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class TFBackgroundNorm(nn.Module):
    """Multiplicative background normalization in the TF domain.

    Parameters
    ----------
    n_fft : int
        FFT size.  0 = auto (next power of 2 above T//4 at first call).
    hop_length : int
        STFT hop.  0 = auto (n_fft // 4).
    win_length : int
        Analysis window length.  0 = auto (same as n_fft).
    time_kernel : int
        Width of the 1-D average-pooling kernel along the time-frame axis.
    freq_kernel : int
        Width of the 1-D average-pooling kernel along the frequency axis.
    bmax : float
        Symmetric clamp applied to the centered background B.
    eps : float
        Floor added inside log(mag² + eps) to avoid log(0).
    """

    def __init__(
        self,
        n_fft: int = 0,
        hop_length: int = 0,
        win_length: int = 0,
        time_kernel: int = 9,
        freq_kernel: int = 5,
        bmax: float = 2.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self._n_fft_cfg   = n_fft
        self._hop_cfg     = hop_length
        self._win_cfg     = win_length
        self.time_kernel  = time_kernel
        self.freq_kernel  = freq_kernel
        self.bmax         = bmax
        self.eps          = eps

        # Lazily resolved on first _stft call
        self._n_fft:  Optional[int]          = None
        self._hop:    Optional[int]          = None
        self._win:    Optional[int]          = None
        self._window: Optional[torch.Tensor] = None

        # Cached background from last normalize() call (for denormalize)
        self._B_hist: Optional[torch.Tensor] = None

        # Diagnostic caches
        self._last_B_scale_min:  float = float("nan")
        self._last_B_scale_max:  float = float("nan")
        self._last_B_scale_mean: float = float("nan")
        self._last_B_std:        float = float("nan")

    # ------------------------------------------------------------------
    def _resolve_params(self, T: int, device: torch.device, dtype: torch.dtype):
        """Compute n_fft / hop / win from the input length if not yet set."""
        if self._n_fft is not None:
            return
        n_fft = (
            self._n_fft_cfg
            if self._n_fft_cfg > 0
            else _next_power_of_two(max(T // 4, 4))
        )
        hop = self._hop_cfg if self._hop_cfg > 0 else n_fft // 4
        win = self._win_cfg if self._win_cfg > 0 else n_fft
        self._n_fft = n_fft
        self._hop   = hop
        self._win   = win
        self._window = torch.hann_window(win, device=device, dtype=dtype)

    def _get_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return Hann window on the correct device/dtype (move lazily)."""
        if self._window.device != device or self._window.dtype != dtype:
            self._window = self._window.to(device=device, dtype=dtype)
        return self._window

    # ------------------------------------------------------------------
    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """STFT: (B, T, C) → (B, C, n_freq, TT) complex."""
        B, T, C = x.shape
        self._resolve_params(T, x.device, x.dtype)
        win = self._get_window(x.device, x.dtype)

        xr = x.permute(0, 2, 1).reshape(B * C, T)          # (B*C, T)
        X  = torch.stft(
            xr,
            n_fft=self._n_fft,
            hop_length=self._hop,
            win_length=self._win,
            window=win,
            center=True,
            return_complex=True,
        )                                                    # (B*C, n_freq, TT)
        n_freq, TT = X.shape[1], X.shape[2]
        return X.reshape(B, C, n_freq, TT)

    def _istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        """ISTFT: (B, C, n_freq, TT) complex → (B, length, C)."""
        B, C, n_freq, TT = X.shape
        win = self._get_window(X.device, torch.float32)

        Xr = X.reshape(B * C, n_freq, TT)
        xr = torch.istft(
            Xr,
            n_fft=self._n_fft,
            hop_length=self._hop,
            win_length=self._win,
            window=win,
            center=True,
            length=length,
        )                                                    # (B*C, length)
        return xr.reshape(B, C, length).permute(0, 2, 1)   # (B, length, C)

    # ------------------------------------------------------------------
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, T, C).  Background stored in self._B_hist."""
        B, T, C = x.shape

        X = self._stft(x)                              # (B, C, n_freq, TT) complex
        n_freq, TT = X.shape[2], X.shape[3]

        # Log power spectrogram
        mag = X.abs()                                  # (B, C, n_freq, TT) real
        L   = torch.log(mag * mag + self.eps)          # (B, C, n_freq, TT)

        # 2-D smoothing — reshape to (B*C, 1, n_freq, TT) for avg_pool2d
        L_4d = L.reshape(B * C, 1, n_freq, TT)

        # Time-axis smoothing: kernel (1, time_kernel)
        tk  = self.time_kernel
        tp  = tk // 2
        Bt  = F.avg_pool2d(L_4d, kernel_size=(1, tk), stride=1, padding=(0, tp))

        # Frequency-axis smoothing: kernel (freq_kernel, 1)
        fk  = self.freq_kernel
        fp  = fk // 2
        Bsm = F.avg_pool2d(Bt,   kernel_size=(fk, 1), stride=1, padding=(fp, 0))

        # Restore shape — trim if padding overshot
        Bsm = Bsm[:, :, :n_freq, :TT]                # (B*C, 1, n_freq, TT)
        Bg  = Bsm.reshape(B, C, n_freq, TT)          # (B,   C, n_freq, TT)

        # Center per (B, C) and clamp
        Bg = Bg - Bg.mean(dim=(2, 3), keepdim=True)
        Bg = torch.clamp(Bg, -self.bmax, self.bmax)

        # Multiplicative normalization: divide amplitude by exp(0.5 * B)
        scale  = torch.exp(-0.5 * Bg)                # real (B, C, n_freq, TT)
        X_norm = X * scale                            # complex × real broadcast

        x_norm = self._istft(X_norm, T)              # (B, T, C)

        # --- Cache ---
        self._B_hist = Bg.detach()
        scale_d = scale.detach()
        self._last_B_scale_min  = float(scale_d.min().item())
        self._last_B_scale_max  = float(scale_d.max().item())
        self._last_B_scale_mean = float(scale_d.mean().item())
        self._last_B_std        = float(self._B_hist.std().item())

        return x_norm

    # ------------------------------------------------------------------
    def denormalize(
        self, y_norm: torch.Tensor, station_pred=None
    ) -> torch.Tensor:
        """(B, O, C) → (B, O, C).  station_pred is accepted but unused."""
        B, O, C = y_norm.shape

        if self._B_hist is None:
            # No cached background — return as-is
            return y_norm

        Y      = self._stft(y_norm)               # (B, C, n_freq, TT_pred)
        TT_pred = Y.shape[3]

        # Extrapolate background: repeat last frame across all prediction frames
        B_last   = self._B_hist[..., -1:]         # (B, C, n_freq, 1)
        B_future = B_last.expand(-1, -1, -1, TT_pred).to(Y.device)

        scale_inv = torch.exp(0.5 * B_future)     # real (B, C, n_freq, TT_pred)
        Y_denorm  = Y * scale_inv                 # complex × real broadcast

        y = self._istft(Y_denorm, O)              # (B, O, C)
        return y

    # ------------------------------------------------------------------
    def get_last_stats(self) -> dict:
        """Diagnostics from the most recent normalize() call."""
        return {
            "scale_min":  self._last_B_scale_min,
            "scale_max":  self._last_B_scale_max,
            "scale_mean": self._last_B_scale_mean,
            "B_std":      self._last_B_std,
        }
