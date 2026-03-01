"""SANMS – Multi-Scale Seasonal Adaptive Normalization.

Like SAN but predicts future period mean/std at multiple time scales
(e.g. 1 / 2 / 4 periods per block), then soft-fuses them with a learned
uncertainty-based weight (lower uncertainty → higher weight).

Interface is kept identical to SAN so that train.py / TTNModel need no changes
beyond adding the build_model branch.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SANMS(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        enc_in: int,
        scales: tuple[int, ...] = (1, 2, 4),
        tau: float = 1.0,
        sigma_min: float = 1e-3,
        lambda_std: float = 1.0,
        ent_weight: float = 0.0,
    ):
        """
        Args:
            seq_len:    Input look-back length.
            pred_len:   Forecast horizon.
            period_len: Period length (same role as in SAN).
                        seq_len and pred_len must each be divisible by period_len.
            enc_in:     Number of input channels.
            scales:     Coarsening factors to try.  Only values s where
                        pred_len_new % s == 0 are kept.
            tau:        Temperature for softmax fusion.  Smaller → sharper.
            sigma_min:  Minimum std clamp (prevents division by near-zero).
            lambda_std: Weight of log-std error term in the per-scale loss.
            ent_weight: Weight of entropy regulariser (promotes multi-scale usage).
                        0 (default) disables it.
        """
        super().__init__()
        if seq_len % period_len != 0:
            raise ValueError(
                f"seq_len={seq_len} must be divisible by period_len={period_len}. "
                "Adjust --san-period-len."
            )
        if pred_len % period_len != 0:
            raise ValueError(
                f"pred_len={pred_len} must be divisible by period_len={period_len}. "
                "Adjust --san-period-len."
            )

        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.period_len = period_len
        self.channels   = enc_in
        self.tau        = tau
        self.sigma_min  = sigma_min
        self.lambda_std = lambda_std
        self.ent_weight = ent_weight
        self.epsilon    = 1e-5

        self.seq_len_new  = seq_len  // period_len
        self.pred_len_new = pred_len // period_len

        # Keep only scales that evenly divide pred_len_new.
        self.valid_scales: list[int] = [
            s for s in sorted(set(int(s) for s in scales))
            if s >= 1 and self.pred_len_new % s == 0
        ]
        if not self.valid_scales:
            raise ValueError(
                f"No valid scales found in {scales} for "
                f"pred_len_new={self.pred_len_new}.  "
                "Choose scales that divide pred_len // period_len."
            )

        # ── Per-scale heads (ModuleDict keys are str(s)) ──────────────────────
        self.head_mean = nn.ModuleDict()
        self.head_std  = nn.ModuleDict()
        self.head_u    = nn.ModuleDict()   # uncertainty

        for s in self.valid_scales:
            k = str(s)
            blk = self.pred_len_new // s
            self.head_mean[k] = _MLP(self.seq_len_new, blk, enc_in, period_len, mode='mean')
            self.head_std[k]  = _MLP(self.seq_len_new, blk, enc_in, period_len, mode='std')
            self.head_u[k]    = _MLP(self.seq_len_new, blk, enc_in, period_len, mode='u')

        # Blending parameter (same convention as SAN.weight)
        self.weight = nn.Parameter(torch.ones(2, enc_in))

        # ── Internal state (populated by normalize()) ─────────────────────────
        self._pred_stats: Optional[torch.Tensor] = None
        # Maps scale s → (mean_blk, std_blk, u_blk) tensors for loss()
        self._scale_outputs: Optional[dict[int, tuple]] = None
        # Scalar diagnostic: mean entropy of fusion weights
        self._last_alpha_ent: float = 0.0

    # ------------------------------------------------------------------
    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, T, C).  Stores fused pred stats in self._pred_stats."""
        bs, length, dim = input.shape
        eps = self.epsilon

        # Period-level statistics from *detached* input (same as SAN)
        x_det = input.detach()
        x_r   = x_det.reshape(bs, self.seq_len_new, self.period_len, dim)
        mean_hist = x_r.mean(dim=2, keepdim=True)                             # (B,P,1,C)
        std_hist  = x_r.std(dim=2, keepdim=True).clamp(min=self.sigma_min)   # (B,P,1,C)

        # Normalize the input (gradient flows through here)
        x_in_r    = input.reshape(bs, self.seq_len_new, self.period_len, dim)
        norm_out  = (x_in_r - mean_hist) / (std_hist + eps)

        # MLP conditioning inputs
        x_flat   = x_det                                   # (B, T, C)
        mean_all = x_flat.mean(dim=1, keepdim=True)       # (B, 1, C)

        # ── Per-scale predictions ─────────────────────────────────────────────
        mean_blks: dict[int, torch.Tensor] = {}
        std_blks:  dict[int, torch.Tensor] = {}
        u_blks:    dict[int, torch.Tensor] = {}

        mean_fulls: list[torch.Tensor] = []
        std_fulls:  list[torch.Tensor] = []
        u_fulls:    list[torch.Tensor] = []

        for s in self.valid_scales:
            k = str(s)

            # Mean head (with same SAN-style bias blending)
            m_raw   = self.head_mean[k](mean_hist.squeeze(2) - mean_all, x_flat - mean_all)
            mean_blk = m_raw * self.weight[0] + mean_all * self.weight[1]  # (B, blk, C)

            # Std head
            std_blk = self.head_std[k](std_hist.squeeze(2), x_flat).clamp(min=self.sigma_min)

            # Uncertainty head (Softplus output, strictly positive)
            u_blk = self.head_u[k](mean_hist.squeeze(2) - mean_all, x_flat - mean_all)

            # Cache block-level outputs for loss()
            mean_blks[s] = mean_blk
            std_blks[s]  = std_blk
            u_blks[s]    = u_blk

            # Broadcast blocks → pred_len_new resolution
            mean_fulls.append(mean_blk.repeat_interleave(s, dim=1))  # (B, pred_len_new, C)
            std_fulls.append(std_blk.repeat_interleave(s, dim=1))
            u_fulls.append(u_blk.repeat_interleave(s, dim=1))

        # ── Softmax fusion ────────────────────────────────────────────────────
        mean_stack = torch.stack(mean_fulls, dim=1)  # (B, K, pred_len_new, C)
        std_stack  = torch.stack(std_fulls,  dim=1)
        u_stack    = torch.stack(u_fulls,    dim=1)

        alpha = torch.softmax(-u_stack / self.tau, dim=1)  # (B, K, pred_len_new, C)

        mean_fused = (alpha * mean_stack).sum(dim=1)                            # (B, pred_len_new, C)
        std_fused  = (alpha * std_stack).sum(dim=1).clamp(min=self.sigma_min)

        # Cache pred stats (same shape convention as SAN: (B, pred_len_new, 2C))
        self._pred_stats    = torch.cat([mean_fused, std_fused], dim=-1)
        self._scale_outputs = {s: (mean_blks[s], std_blks[s], u_blks[s]) for s in self.valid_scales}

        # Entropy diagnostic (detached)
        with torch.no_grad():
            a_det = alpha.detach().clamp(min=1e-8)
            self._last_alpha_ent = float(-(a_det * a_det.log()).sum(dim=1).mean().item())

        return norm_out.reshape(bs, length, dim)

    # ------------------------------------------------------------------
    def denormalize(self, input: torch.Tensor, station_pred=None) -> torch.Tensor:
        """(B, pred_len, C) → (B, pred_len, C) using stored/provided pred stats."""
        if station_pred is None:
            station_pred = self._pred_stats
        if station_pred is not None:
            bs, length, dim = input.shape
            x    = input.reshape(bs, self.pred_len_new, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)  # (B, pred_len_new, 1, C)
            std  = station_pred[:, :, self.channels:].unsqueeze(2)
            out  = x * (std + self.epsilon) + mean
            return out.reshape(bs, length, dim)
        return input

    # ------------------------------------------------------------------
    def loss(self, true: torch.Tensor) -> torch.Tensor:
        """Multi-scale NLL-style supervision loss.

        For each scale s:
            Lₛ = mean( base_err / u  +  log(u) )
        where
            base_err = (mean_pred − mean_true)² + λ_std (log σ_pred − log σ_true)²
            u        = head_u output + 1e-6

        Optional entropy regulariser promotes uniform mixing across scales.
        """
        if self._pred_stats is None or self._scale_outputs is None:
            return torch.tensor(0.0, device=true.device)

        bs, pred_len, n = true.shape
        eps = self.epsilon
        total = true.new_zeros(())

        for s in self.valid_scales:
            blk_len      = self.pred_len_new // s
            block_samples = s * self.period_len   # time steps per block

            # Oracle block stats from future ground truth
            true_blk    = true.reshape(bs, blk_len, block_samples, n)
            mean_true_s = true_blk.mean(dim=2)                              # (B, blk, C)
            std_true_s  = true_blk.std(dim=2).clamp(min=self.sigma_min)    # (B, blk, C)

            mean_blk, std_blk, u_blk = self._scale_outputs[s]

            err_mean = (mean_blk - mean_true_s).pow(2)
            err_std  = (torch.log(std_blk + eps) - torch.log(std_true_s + eps)).pow(2)
            base_err = err_mean + self.lambda_std * err_std

            u = u_blk + 1e-6
            L_s = (base_err / u + torch.log(u)).mean()
            total = total + L_s

        total = total / len(self.valid_scales)

        # Optional entropy regulariser (ent_weight typically very small, e.g. 1e-3)
        if self.ent_weight > 0.0 and len(self.valid_scales) > 1:
            # Recompute alpha with gradients so the entropy term trains the u heads
            u_fulls = [
                self._scale_outputs[s][2].repeat_interleave(s, dim=1)
                for s in self.valid_scales
            ]
            u_stack = torch.stack(u_fulls, dim=1)          # (B, K, pred_len_new, C)
            alpha   = torch.softmax(-u_stack / self.tau, dim=1)
            neg_ent = (alpha * alpha.clamp(min=1e-8).log()).sum(dim=1).mean()
            total   = total + self.ent_weight * neg_ent    # minimising −H → maximising H

        return total

    # ------------------------------------------------------------------
    def forward(self, batch_x, mode: str = 'n', station_pred=None):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x, station_pred)


# ======================================================================
class _MLP(nn.Module):
    """MLP stat predictor for SANMS (same architecture as SAN's _MLP,
    extended with mode='u' for uncertainty output via Softplus)."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, period_len: int, mode: str):
        super().__init__()
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.channels   = enc_in
        self.period_len = period_len
        self.mode       = mode

        if mode == 'std':
            self.activation       = nn.ReLU()
            self.final_activation = nn.ReLU()
        elif mode == 'u':
            self.activation       = nn.ReLU()
            self.final_activation = nn.Softplus()
        else:  # 'mean'
            self.activation       = nn.Tanh()
            self.final_activation = nn.Identity()

        self.input     = nn.Linear(seq_len, 512)
        self.input_raw = nn.Linear(seq_len * period_len, 512)
        self.output    = nn.Linear(1024, pred_len)

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        # x:     (B, seq_len_new, C)
        # x_raw: (B, seq_len,     C)
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)  # (B, C, ...)
        x     = self.input(x)                                     # (B, C, 512)
        x_raw = self.input_raw(x_raw)                             # (B, C, 512)
        x     = self.output(self.activation(torch.cat([x, x_raw], dim=-1)))
        return self.final_activation(x).permute(0, 2, 1)          # (B, pred_len, C)
