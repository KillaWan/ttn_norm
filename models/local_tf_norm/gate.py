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
        gate_budget_ratio: float = 0.0,
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
        self.gate_budget_ratio = float(gate_budget_ratio)
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
            # Soft top-k gate with budget scaling
            # dim=2 for freq, dim=3 for time
            axis = 2 if self.gate_budget_dim == "freq" else 3
            bins = logits.shape[axis]
            
            # Get probability distribution via softmax
            probs = torch.softmax(logits / temperature, dim=axis)
            
            # Scale by budget: target activation sum along axis
            if self.gate_budget_ratio > 0:
                target_sum = self.gate_budget_ratio * bins
            else:
                target_sum = bins  # No budget constraint
            
            g = probs * target_sum
            return torch.clamp(g, 0.0, 1.0)
        
        elif self.gate_mode == "sigmoid_budget":
            # Sigmoid + budget scaling: scale raw sigmoid to match target sum
            axis = 2 if self.gate_budget_dim == "freq" else 3
            bins = logits.shape[axis]
            
            raw = torch.sigmoid(logits / temperature)
            
            if self.gate_budget_ratio > 0:
                target_sum = self.gate_budget_ratio * bins
                raw_sum = raw.sum(dim=axis, keepdim=True)
                # Avoid division by zero
                scale = target_sum / (raw_sum + 1e-10)
                g = raw * scale
            else:
                g = raw
            
            return torch.clamp(g, 0.0, 1.0)
        
        else:
            raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")

