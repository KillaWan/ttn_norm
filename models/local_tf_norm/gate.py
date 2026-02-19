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
            # Soft top-k gate: softmax along budget dimension, scale to budget
            # dim=2 for freq, dim=3 for time
            axis = 2 if self.gate_budget_dim == "freq" else 3
            probs = torch.softmax(logits / temperature, dim=axis)
            # Budget is implicitly 1.0 (sum along axis should be 1), clamp to [0,1]
            return torch.clamp(probs, 0.0, 1.0)
        
        elif self.gate_mode == "sigmoid_budget":
            # Sigmoid + soft constraint: raw sigmoid scaled along axis
            raw = torch.sigmoid(logits / temperature)
            axis = 2 if self.gate_budget_dim == "freq" else 3
            # Compute sum and scale factor
            axis_sum = raw.sum(dim=axis, keepdim=True)
            # Avoid division by zero
            axis_sum = torch.clamp(axis_sum, min=1e-6)
            # Scale to have average activation ~ 1.0 / (bins along axis)
            # For now just normalize to sum=1 along axis, then clamp
            scaled = raw / axis_sum
            return torch.clamp(scaled, 0.0, 1.0)
        
        else:
            raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")

