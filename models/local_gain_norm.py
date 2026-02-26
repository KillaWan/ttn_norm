"""LocalGainNorm: video-exposure style slow-gain normalization.

Normalizes a time series by a causal EMA-based RMS gain curve,
then predicts the future gain with a small linear or MLP predictor,
and re-scales the backbone output on denormalization.

Design principles:
- Strictly causal: gain at time t depends only on x_0..x_t.
- Low-dimensional target: the gain curve is 1-D per channel, smooth.
- Decoupled gradients: task_loss does NOT train the gain predictor
  (denormalize uses detached gain); gain_pred_loss trains the predictor.
- No STFT, no trigger mask, no gate — entirely time-domain.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGainNorm(nn.Module):
    """Causal EMA gain normalization with learned future-gain predictor.

    Interface (matches TTNModel expectations):
      normalize(batch_x: (B, T, C)) -> x_norm: (B, T, C)
      denormalize(y_norm: (B, T', C)) -> y_pred: (B, T', C)
      forward(batch_x) -> normalize(batch_x)   [for TTNModel generic fallback]
      teacher_gain_future(x_hist_raw, y_true_raw) -> g_future_true: (B, C, pred_len)
      get_last_gain_pred_future() -> Optional[(B, C, pred_len) w/ grad]
      get_last_gain_hist()        -> Optional[(B, C, seq_len) detached]
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        gain_alpha: float = 0.98,
        gain_eps: float = 1e-6,
        gain_pred_arch: str = "linear",   # "linear" | "mlp"
    ):
        super().__init__()

        if gain_pred_arch not in ("linear", "mlp"):
            raise ValueError(
                f"gain_pred_arch must be 'linear' or 'mlp', got {gain_pred_arch!r}"
            )

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.gain_alpha = float(gain_alpha)
        self.gain_eps = float(gain_eps)
        self.gain_pred_arch = gain_pred_arch

        if gain_pred_arch == "mlp":
            hidden = max(64, seq_len)
            self.gain_predictor: nn.Module = nn.Sequential(
                nn.Linear(seq_len, hidden),
                nn.GELU(),
                nn.Linear(hidden, pred_len),
            )
        else:  # "linear"
            self.gain_predictor = nn.Linear(seq_len, pred_len)

        # Caches
        self._last_gain_hist: Optional[torch.Tensor] = None         # (B, C, seq_len) detached
        self._last_gain_pred_future: Optional[torch.Tensor] = None  # (B, C, pred_len) w/ grad
        self._last_gain_pred_loss: float = 0.0

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def ema_gain(self, x: torch.Tensor) -> torch.Tensor:
        """Causal EMA of squared amplitude → RMS gain curve.

        Args:
            x: (B, C, L)

        Returns:
            g: (B, C, L)  where  g_t = sqrt(alpha*s_{t-1} + (1-alpha)*x_t^2 + eps)
        """
        alpha = self.gain_alpha
        beta = 1.0 - alpha
        eps = self.gain_eps
        B, C, L = x.shape
        s = x.new_zeros(B, C)
        g_list = []
        for t in range(L):
            s = alpha * s + beta * (x[..., t] ** 2)
            g_list.append((s + eps).sqrt())
        return torch.stack(g_list, dim=-1)   # (B, C, L)

    # ------------------------------------------------------------------
    # normalize / denormalize
    # ------------------------------------------------------------------

    def normalize(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Divide history by causal EMA gain.

        Args:
            batch_x: (B, T, C)

        Returns:
            x_norm: (B, T, C)
        """
        x = batch_x.permute(0, 2, 1)        # (B, C, T)
        g_hist = self.ema_gain(x)            # (B, C, T)
        self._last_gain_hist = g_hist.detach()
        x_norm = x / (g_hist + self.gain_eps)
        return x_norm.permute(0, 2, 1)       # (B, T, C)

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Multiply backbone output by predicted future gain.

        The gain predictor is trained only via the explicit gain_pred_loss;
        task_loss sees detached gain (same decoupling as LocalTFNorm pred path).

        Args:
            y_norm: (B, T', C)

        Returns:
            y_pred: (B, T', C)
        """
        if self._last_gain_hist is None:
            # Fallback: should not happen during normal training
            return y_norm

        g_hist = self._last_gain_hist           # (B, C, seq_len) detached
        B, C, _ = g_hist.shape

        # Predict future gain; apply softplus to guarantee positivity
        g_hist_flat = g_hist.reshape(B * C, -1)                    # (B*C, seq_len)
        raw_pred = self.gain_predictor(g_hist_flat)                # (B*C, pred_len)
        g_future_pred = F.softplus(raw_pred).reshape(B, C, -1)    # (B, C, pred_len) > 0

        self._last_gain_pred_future = g_future_pred                # keep grad for gain_pred_loss

        # Detach for task_loss path (task_loss must not train the gain predictor)
        g_scale = g_future_pred.detach().permute(0, 2, 1)         # (B, pred_len, C)
        return y_norm * (g_scale + self.gain_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for normalize(); enables TTNModel's generic fallback."""
        return self.normalize(x)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_last_gain_pred_future(self) -> Optional[torch.Tensor]:
        """Return (B, C, pred_len) predicted gain with grad, or None."""
        return self._last_gain_pred_future

    def get_last_gain_hist(self) -> Optional[torch.Tensor]:
        """Return (B, C, seq_len) history gain (detached), or None."""
        return self._last_gain_hist

    # ------------------------------------------------------------------
    # Teacher label
    # ------------------------------------------------------------------

    @torch.no_grad()
    def teacher_gain_future(
        self, x_hist_raw: torch.Tensor, y_true_raw: torch.Tensor
    ) -> torch.Tensor:
        """Compute causal ground-truth future gain via EMA on full sequence.

        The EMA is causal by construction: gain at t depends only on x_0..x_t.
        The future portion of the EMA therefore sees x_hist influence through
        the state, not through any look-ahead on y_true.

        Args:
            x_hist_raw: (B, T, C)   history in original / scaled space
            y_true_raw: (B, T', C)  ground-truth future in same space

        Returns:
            g_future_true: (B, C, pred_len)
        """
        x_full = torch.cat([x_hist_raw, y_true_raw], dim=1)   # (B, T+T', C)
        x_full_ch = x_full.permute(0, 2, 1)                   # (B, C, T+T')
        g_full = self.ema_gain(x_full_ch)                      # (B, C, T+T')
        return g_full[..., -self.pred_len:]                    # (B, C, pred_len)
