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
    ):
        super().__init__()
        if gate_type == "depthwise":
            self.proj = nn.Conv2d(
                channels, channels, kernel_size=1, groups=channels, bias=True
            )
        elif gate_type == "pointwise":
            self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}")
        self.use_threshold = use_threshold
        self.temperature = float(temperature)
        self.gate_mode = gate_mode
        self.gate_budget_dim = gate_budget_dim
        self.gate_ratio_target = float(gate_ratio_target)
        if use_threshold:
            self.threshold = nn.Parameter(torch.full((channels, 1, 1), init_threshold))
        else:
            self.register_parameter("threshold", None)

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        # magnitude: (B, C, F, TT)
        logits = self.proj(magnitude)
        if self.use_threshold and self.threshold is not None:
            logits = logits - self.threshold
        temperature = max(self.temperature, 1e-3)
        
        if self.gate_mode == "sigmoid":
            return torch.sigmoid(logits / temperature)
        
        elif self.gate_mode == "softmax_budget":
            # Soft top-k: softmax along budget_dim, scale by target_sum
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


