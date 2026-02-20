from __future__ import annotations

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

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        # magnitude: (B, C, F, TT)
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


