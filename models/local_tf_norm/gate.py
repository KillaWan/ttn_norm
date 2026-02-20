from __future__ import annotations

import math

import torch
import torch.nn as nn


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
        # hf_cutoff-specific params (ignored for other gate_modes)
        hf_tau: float = 0.05,
        hf_cutoff_mode: str = "global",  # "global" or "channel"
        hf_init_ratio: float = 0.25,
    ):
        super().__init__()
        self.gate_arch = gate_arch
        self.use_threshold = use_threshold
        self.temperature = float(temperature)
        self.gate_mode = gate_mode
        self.gate_budget_dim = gate_budget_dim
        self.gate_ratio_target = float(gate_ratio_target)
        self.gate_threshold_mode = gate_threshold_mode
        self.hf_tau = float(hf_tau)
        self.hf_cutoff_mode = hf_cutoff_mode
        self.channels = channels

        if gate_mode == "hf_cutoff":
            # Structured monotone high-frequency gate; no conv projection needed.
            # Gate shape: g_high(f) = sigmoid((f - c) / tau), f in [0, 1].
            # Learnable: c = sigmoid(cutoff_raw) in (0, 1).
            # Initialise so the gate passes the top hf_init_ratio fraction:
            #   c_init = 1 - hf_init_ratio  (gate ≈ 1 for f > c_init)
            c_init = 1.0 - float(hf_init_ratio)
            c_init = max(1e-4, min(1.0 - 1e-4, c_init))  # keep logit finite
            cutoff_raw_init = math.log(c_init / (1.0 - c_init))  # logit(c_init)
            if hf_cutoff_mode == "channel":
                self.cutoff_raw = nn.Parameter(
                    torch.full((channels,), cutoff_raw_init)
                )
            else:  # global: single scalar cutoff
                self.cutoff_raw = nn.Parameter(torch.tensor(cutoff_raw_init))
            # No threshold or projection for hf_cutoff
            self.register_parameter("threshold", None)
        else:
            # Build projection depending on requested gate architecture
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
            else:
                raise ValueError(f"Unsupported gate_arch: {gate_arch}")
            if use_threshold:
                # If mask mode requested, keep threshold as a buffer (non-trainable)
                if gate_threshold_mode == "mask":
                    self.register_buffer("threshold", torch.full((channels, 1, 1), init_threshold))
                else:
                    self.threshold = nn.Parameter(torch.full((channels, 1, 1), init_threshold))
            else:
                self.register_parameter("threshold", None)

    def get_hf_cutoff_stats(self) -> dict[str, float]:
        """Return current cutoff value(s) for monitoring. Only meaningful for hf_cutoff mode."""
        if self.gate_mode != "hf_cutoff":
            return {}
        with torch.no_grad():
            c = torch.sigmoid(self.cutoff_raw.detach().cpu())
            if self.hf_cutoff_mode == "channel":
                return {
                    "cutoff_mean": float(c.mean()),
                    "cutoff_min": float(c.min()),
                    "cutoff_max": float(c.max()),
                }
            else:
                return {"cutoff": float(c)}

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        # magnitude: (B, C, F, TT)

        if self.gate_mode == "hf_cutoff":
            F = magnitude.shape[2]
            device = magnitude.device
            dtype = magnitude.dtype
            # Frequency grid in [0, 1], shape (1, 1, F, 1) – broadcasts to (B, C, F, T)
            f_grid = torch.linspace(0.0, 1.0, F, device=device, dtype=dtype).view(1, 1, F, 1)
            # Learnable cutoff c in (0, 1)
            c = torch.sigmoid(self.cutoff_raw)
            if self.hf_cutoff_mode == "channel":
                c = c.view(1, -1, 1, 1)  # (1, C, 1, 1)
            else:
                c = c.view(1, 1, 1, 1)   # (1, 1, 1, 1)
            # Monotone step: frequencies above cutoff pass (≈1), below are suppressed (≈0)
            g_high = torch.sigmoid((f_grid - c) / self.hf_tau)
            return g_high  # shape (1, C, F, 1) or (1, 1, F, 1); broadcasts downstream

        logits = self.proj(magnitude)
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
