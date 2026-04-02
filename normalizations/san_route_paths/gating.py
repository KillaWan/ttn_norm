"""GatingPath — fixed 4-expert bank with state-conditioned soft gating.

Expert bank (fixed structure, not configurable per state):
    e1: identity                 y_base
    e2: local additive delta     Conv1d(kernel=3, padding=1)
    e3: smooth additive delta    2-layer dilated Conv1d
    e4: osc additive delta       depthwise Conv1d + pointwise Conv1d

All expert delta heads are zero-initialised so every expert starts as y_base.
The gating head is also zero-initialised, yielding uniform softmax weights (0.25
each), so the initial output is the average of four identical y_base terms = y_base.

Gating logits shape: (B, H, C, 4) — per-channel soft mixture.

Unified path interface:
    forward(y_norm, y_base, future_state_hat, future_state_time,
            mu_base_fut, std_base_fut, base_time_mean, base_time_std)
    -> y_route: (B, H, C)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingPath(nn.Module):
    """4-expert soft-gating path.

    Expert inputs: cat(y_base, future_state_time) along C → (B, H, 2C),
                   permuted to (B, 2C, H) for Conv1d.
    Gating input : future_state_time → (B, H, C*4) → reshape (B, H, C, 4) → softmax.
    """

    def __init__(self, pred_len: int, enc_in: int, sigma_min: float = 1e-3, **kwargs):
        super().__init__()
        C2 = 2 * enc_in
        hidden = max(enc_in, 16)

        # Expert 2: local — single Conv1d, zero-init
        self.e2_head = nn.Conv1d(C2, enc_in, kernel_size=3, padding=1)
        nn.init.zeros_(self.e2_head.weight)
        nn.init.zeros_(self.e2_head.bias)

        # Expert 3: smooth — two dilated convs, last layer zero-init
        self.e3_head = nn.Sequential(
            nn.Conv1d(C2, hidden, kernel_size=3, dilation=1, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, enc_in, kernel_size=3, dilation=2, padding=2),
        )
        nn.init.zeros_(self.e3_head[-1].weight)
        nn.init.zeros_(self.e3_head[-1].bias)

        # Expert 4: osc — depthwise + pointwise, pointwise zero-init
        self.e4_dw = nn.Conv1d(C2, C2, kernel_size=3, padding=1, groups=C2)
        self.e4_pw = nn.Conv1d(C2, enc_in, kernel_size=1)
        nn.init.zeros_(self.e4_pw.weight)
        nn.init.zeros_(self.e4_pw.bias)

        # Gating head: (B, H, C) → (B, H, C*4) → (B, H, C, 4), zero-init
        self.gate_head = nn.Linear(enc_in, enc_in * 4)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(
        self,
        y_norm: torch.Tensor,
        y_base: torch.Tensor,             # (B, H, C)
        future_state_hat: torch.Tensor,
        future_state_time: torch.Tensor,  # (B, H, C)
        mu_base_fut: torch.Tensor,
        std_base_fut: torch.Tensor,
        base_time_mean: torch.Tensor,
        base_time_std: torch.Tensor,
    ) -> torch.Tensor:
        B, H, C = y_base.shape

        # Shared expert input: (B, 2C, H)
        inp = torch.cat([y_base, future_state_time], dim=-1).permute(0, 2, 1)

        # Expert 1: identity
        e1 = y_base                                             # (B, H, C)

        # Expert 2: local
        e2 = y_base + self.e2_head(inp).permute(0, 2, 1)       # (B, H, C)

        # Expert 3: smooth
        e3 = y_base + self.e3_head(inp).permute(0, 2, 1)       # (B, H, C)

        # Expert 4: oscillatory (depthwise → GELU → pointwise)
        e4_feat = F.gelu(self.e4_dw(inp))                      # (B, 2C, H)
        e4 = y_base + self.e4_pw(e4_feat).permute(0, 2, 1)     # (B, H, C)

        # Stack experts: (B, H, C, 4)
        experts = torch.stack([e1, e2, e3, e4], dim=-1)

        # Gating: (B, H, C*4) → (B, H, C, 4) → softmax
        logits  = self.gate_head(future_state_time).view(B, H, C, 4)
        weights = F.softmax(logits, dim=-1)                     # (B, H, C, 4)

        y_route = (experts * weights).sum(dim=-1)               # (B, H, C)
        return y_route
