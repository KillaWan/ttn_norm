from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FTSep5Proj(nn.Module):
    """TF-separable gate projection for ftsep5 architecture.

    Takes magnitude (B, C, F, T) and returns logits (B, C, F, T).

    Internal pipeline:
      1. Build per-variable features: m (magnitude) and dm (time-diff of m).
      2. Per-(B,C) (F,T) standardization of m and dm independently.
      3. Interleave into (B, 2C, F, T) with group-compatible ordering.
      4. stem: group conv (2C→h*C, groups=C) — 2 in-ch → h out-ch per variable.
      5. block×2: depthwise freq conv(5,1) → depthwise time conv(1,5)
                  → pointwise group conv(h*C→h*C, groups=C) → GELU.
      6. head: group conv (h*C→C, groups=C) — h→1 logit per variable.
    """

    def __init__(self, channels: int, h: int = 8):
        super().__init__()
        self.channels = channels
        self.h = h
        C, hC = channels, h * channels

        # stem: 2 features per variable → h hidden channels
        self.stem = nn.Conv2d(2 * C, hC, kernel_size=1, groups=C, bias=True)

        # block 1
        self.dw_freq1 = nn.Conv2d(hC, hC, kernel_size=(5, 1), padding=(2, 0), groups=hC, bias=True)
        self.dw_time1 = nn.Conv2d(hC, hC, kernel_size=(1, 5), padding=(0, 2), groups=hC, bias=True)
        self.pw1 = nn.Conv2d(hC, hC, kernel_size=1, groups=C, bias=True)

        # block 2
        self.dw_freq2 = nn.Conv2d(hC, hC, kernel_size=(5, 1), padding=(2, 0), groups=hC, bias=True)
        self.dw_time2 = nn.Conv2d(hC, hC, kernel_size=(1, 5), padding=(0, 2), groups=hC, bias=True)
        self.pw2 = nn.Conv2d(hC, hC, kernel_size=1, groups=C, bias=True)

        # head: h channels → 1 logit per variable
        self.head = nn.Conv2d(hC, C, kernel_size=1, groups=C, bias=True)

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        # magnitude: (B, C, F, T)
        B, C, F, T = magnitude.shape

        m = magnitude  # (B, C, F, T)

        # Time difference: diff along T, pad left to restore T
        dm = F_.diff_t(m)  # (B, C, F, T)

        # Per-(B,C) standardization over (F,T)
        m_norm = _bc_standardize(m)    # (B, C, F, T)
        dm_norm = _bc_standardize(dm)  # (B, C, F, T)

        # Interleave: [m_0, dm_0, m_1, dm_1, ...] → (B, 2C, F, T)
        # stack along new dim then reshape preserves per-variable grouping
        x = torch.stack([m_norm, dm_norm], dim=2)  # (B, C, 2, F, T)
        x = x.reshape(B, 2 * C, F, T)             # (B, 2C, F, T)

        # stem
        x = torch.nn.functional.gelu(self.stem(x))  # (B, hC, F, T)

        # block 1
        x = torch.nn.functional.gelu(self.dw_freq1(x))
        x = torch.nn.functional.gelu(self.dw_time1(x))
        x = torch.nn.functional.gelu(self.pw1(x))

        # block 2
        x = torch.nn.functional.gelu(self.dw_freq2(x))
        x = torch.nn.functional.gelu(self.dw_time2(x))
        x = torch.nn.functional.gelu(self.pw2(x))

        # head → (B, C, F, T) logits
        logits = self.head(x)
        return logits


# ---------------------------------------------------------------------------
# helpers used only inside FTSep5Proj
# ---------------------------------------------------------------------------

class _F:
    """Namespace to avoid polluting module scope."""
    @staticmethod
    def diff_t(x: torch.Tensor) -> torch.Tensor:
        """Time-difference of x (B,C,F,T), left-pad to preserve shape."""
        d = x[..., 1:] - x[..., :-1]  # (B, C, F, T-1)
        return torch.nn.functional.pad(d, (1, 0))  # (B, C, F, T)


# module-level alias so FTSep5Proj.forward can reference it
F_ = _F


def _bc_standardize(x: torch.Tensor) -> torch.Tensor:
    """Standardize x per (B, C) over the (F, T) dimensions.

    Args:
        x: (B, C, F, T)
    Returns:
        (x - mean) / (std + 1e-6) of same shape
    """
    B, C, F, T = x.shape
    flat = x.reshape(B, C, F * T)
    mean = flat.mean(dim=-1, keepdim=True)          # (B, C, 1)
    std = flat.std(dim=-1, keepdim=True) + 1e-6     # (B, C, 1)
    norm = ((flat - mean) / std).reshape(B, C, F, T)
    return norm


class LocalTFGate(nn.Module):
    def __init__(
        self,
        channels: int,
        gate_type: str = "depthwise",
        use_threshold: bool = True,
        init_threshold: float = 0.0,
        temperature: float = 1.0,
        gate_mode: str = "sigmoid",
        gate_budget_dim: str = "freq",
        gate_ratio_target: float = 0.3,
        gate_arch: str = "pointwise",
        gate_threshold_mode: str = "shift",
    ):
        super().__init__()
        # Build projection depending on requested gate architecture
        self.gate_arch = gate_arch
        if gate_arch == "pointwise":
            if gate_type == "depthwise":
                self.proj = nn.Conv2d(
                    channels, channels, kernel_size=1, groups=channels, bias=True
                )
            else:
                self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        elif gate_arch == "freqconv3":
            # Depthwise conv along freq (kernel (3,1)) followed by pointwise
            self.proj = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels, bias=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            )
        elif gate_arch == "freqconv5":
            self.proj = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels, bias=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            )
        elif gate_arch == "ftsep5":
            # TF-separable gate with per-variable time-diff features
            self.proj = FTSep5Proj(channels, h=8)
        else:
            raise ValueError(f"Unsupported gate_arch: {gate_arch}")
        self.use_threshold = use_threshold
        self.temperature = float(temperature)
        self.gate_mode = gate_mode
        self.gate_budget_dim = gate_budget_dim
        self.gate_ratio_target = float(gate_ratio_target)
        self.gate_threshold_mode = gate_threshold_mode
        if use_threshold:
            # If mask mode requested, keep threshold as a buffer (non-trainable)
            if gate_threshold_mode == "mask":
                self.register_buffer("threshold", torch.full((channels, 1, 1), init_threshold))
            else:
                self.threshold = nn.Parameter(torch.full((channels, 1, 1), init_threshold))
        else:
            self.register_parameter("threshold", None)

        # Diagnostic caches (detached, no grad)
        self._last_logits: Optional[torch.Tensor] = None

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        # magnitude: (B, C, F, TT)
        logits = self.proj(magnitude)
        # Cache raw logits (before threshold/temperature) for diagnostics
        self._last_logits = logits.detach()
        # Threshold handling: shift (subtract) or mask (suppress low logits)
        if self.use_threshold and self.threshold is not None:
            if self.gate_threshold_mode == "shift":
                logits = logits - self.threshold
            elif self.gate_threshold_mode == "mask":
                # mask low logits by setting them to a large negative value
                # but protect against masking all frequencies for a given (B,C,T)
                thr = self.threshold
                orig = logits
                mask = orig < thr
                masked = orig.masked_fill(mask, float(-1e9))
                # detect positions where all frequencies are masked
                # mask shape: (B, C, F, T) -> all_masked: (B, C, T)
                all_masked = mask.all(dim=2)
                if all_masked.any():
                    # expand selector to frequency dim and restore original logits there
                    sel = all_masked.unsqueeze(2)  # (B, C, 1, T)
                    logits = torch.where(sel, orig, masked)
                else:
                    logits = masked
            else:
                logits = logits - self.threshold
        temperature = max(self.temperature, 1e-3)

        if self.gate_mode == "sigmoid":
            return torch.sigmoid(logits / temperature)

        elif self.gate_mode == "softmax_budget":
            # Soft top-k: softmax along budget_dim, then scale by target_sum
            axis = 2 if self.gate_budget_dim == "freq" else 3
            bins = logits.shape[axis]

            probs = torch.softmax(logits / temperature, dim=axis)
            target_sum = self.gate_ratio_target * bins
            g = probs * target_sum
            return torch.clamp(g, 0.0, 1.0)

        elif self.gate_mode == "sigmoid_budget":
            # Sigmoid + budget scaling
            axis = 2 if self.gate_budget_dim == "freq" else 3
            bins = logits.shape[axis]

            raw = torch.sigmoid(logits / temperature)
            raw_sum = raw.sum(dim=axis, keepdim=True)
            target_sum = self.gate_ratio_target * bins
            scale = target_sum / (raw_sum + 1e-10)
            g = raw * scale
            return torch.clamp(g, 0.0, 1.0)

        else:
            raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")
