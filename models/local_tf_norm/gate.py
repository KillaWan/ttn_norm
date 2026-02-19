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
        return torch.sigmoid(logits / temperature)
