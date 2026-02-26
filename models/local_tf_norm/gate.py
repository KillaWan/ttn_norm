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

    def __init__(self, channels: int, h: int = 8, feat_mode: str = "mdm"):
        if feat_mode not in ("mdm", "mdm_pdp"):
            raise ValueError(f"feat_mode must be 'mdm' or 'mdm_pdp', got {feat_mode!r}")
        super().__init__()
        self.channels = channels
        self.h = h
        self.feat_mode = feat_mode
        self.n_feat = 2 if feat_mode == "mdm" else 4
        C, hC = channels, h * channels

        # stem: n_feat features per variable → h hidden channels
        self.stem = nn.Conv2d(self.n_feat * C, hC, kernel_size=1, groups=C, bias=True)

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

        if self.feat_mode == "mdm":
            features = [m_norm, dm_norm]
        else:  # mdm_pdp
            # Probability over freq bins, normalised per (B,C,T)
            P = m ** 2 + 1e-8                                          # (B, C, F, T)
            p = P / (P.sum(dim=2, keepdim=True) + 1e-8)               # (B, C, F, T)
            dp = F_.diff_t(p)                                          # (B, C, F, T)
            p_norm = _bc_standardize(p)                                # (B, C, F, T)
            dp_norm = _bc_standardize(dp)                              # (B, C, F, T)
            features = [m_norm, dm_norm, p_norm, dp_norm]

        # Interleave: per-variable grouping [feat0_var0, feat1_var0, ..., feat0_var1, ...]
        # stack along new dim=2 then reshape preserves per-variable grouping
        x = torch.stack(features, dim=2)              # (B, C, n_feat, F, T)
        x = x.reshape(B, self.n_feat * C, F, T)      # (B, n_feat*C, F, T)

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


class LowRankProj(nn.Module):
    """Low-rank TF gate projection.

    Computes logits(f, t) = Σ_{k=1..r} u_k(t) · v_k(f) + optional a(t) + b(f).

    u(t) is produced by a per-variable depthwise Conv1d applied to the time-marginal
    of the STFT magnitude; v(f) comes from the freq-marginal.  v is L2-normalised
    along the frequency axis to prevent u/v scale drift.

    Args:
        channels: number of input/output channels (C)
        rank:     number of rank-1 components (r)
        time_ks:  kernel size for the time-axis depthwise Conv1d
        freq_ks:  kernel size for the freq-axis depthwise Conv1d
        use_bias: if True, add learned a(t) and b(f) scalar-bias terms
    """

    def __init__(
        self,
        channels: int,
        rank: int = 4,
        time_ks: int = 5,
        freq_ks: int = 3,
        use_bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.rank = rank
        self.use_bias = use_bias

        # u: time-marginal (B, C, T) → (B, C*rank, T) via per-variable depthwise conv
        self.u_conv = nn.Conv1d(
            channels, channels * rank,
            kernel_size=time_ks, padding=time_ks // 2,
            groups=channels, bias=True,
        )
        # v: freq-marginal (B, C, F) → (B, C*rank, F)
        self.v_conv = nn.Conv1d(
            channels, channels * rank,
            kernel_size=freq_ks, padding=freq_ks // 2,
            groups=channels, bias=True,
        )
        if use_bias:
            # a(t): time-varying scalar bias per variable
            self.a_conv = nn.Conv1d(
                channels, channels,
                kernel_size=time_ks, padding=time_ks // 2,
                groups=channels, bias=True,
            )
            # b(f): frequency scalar bias per variable
            self.b_conv = nn.Conv1d(
                channels, channels,
                kernel_size=freq_ks, padding=freq_ks // 2,
                groups=channels, bias=True,
            )

        # Diagnostic caches (last forward pass)
        self._last_u: Optional[torch.Tensor] = None      # (B, C, rank, T) detached
        self._last_u_raw: Optional[torch.Tensor] = None  # (B, C, rank, T) with grad
        self._last_mag_mean: float = 0.0
        self._last_mag_std: float = 0.0

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        # magnitude: (B, C, F, T)
        B, C, F, T = magnitude.shape
        r = self.rank

        # Per-(B,C) standardization across (F,T) for scale-stable inputs
        mag = _bc_standardize(magnitude)
        self._last_mag_mean = float(mag.detach().mean().item())
        self._last_mag_std = float(mag.detach().std(unbiased=False).item())

        # Time-marginal: average over F → (B, C, T)
        m_t = mag.mean(dim=2)
        # Freq-marginal: average over T → (B, C, F)
        m_f = mag.mean(dim=3)

        # u: (B, C, rank, T)
        u = self.u_conv(m_t).reshape(B, C, r, T)
        self._last_u_raw = u             # with grad for TV loss
        self._last_u = u.detach()        # detached for diagnostics

        # v: (B, C, rank, F), L2-normalised along F to decouple scale from u
        v = self.v_conv(m_f).reshape(B, C, r, F)
        v = torch.nn.functional.normalize(v, dim=3, eps=1e-6)

        # Low-rank logits: Σ_k u_k(t) * v_k(f) → (B, C, F, T)
        logits = torch.einsum('bckt,bckf->bcft', u, v)

        if self.use_bias:
            a = self.a_conv(m_t)   # (B, C, T)
            b = self.b_conv(m_f)   # (B, C, F)
            logits = logits + a.unsqueeze(2) + b.unsqueeze(3)

        return logits


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
        ftsep5_feat_mode: str = "mdm",
        # Low-rank gate params (used when gate_arch in {"lowrank", "lowrank_sparse"})
        lowrank_rank: int = 4,
        lowrank_time_ks: int = 5,
        lowrank_freq_ks: int = 3,
        lowrank_use_bias: bool = True,
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
            self.proj = FTSep5Proj(channels, h=8, feat_mode=ftsep5_feat_mode)
        elif gate_arch == "lowrank":
            self.proj = LowRankProj(
                channels, rank=lowrank_rank,
                time_ks=lowrank_time_ks, freq_ks=lowrank_freq_ks,
                use_bias=lowrank_use_bias,
            )
        elif gate_arch == "lowrank_sparse":
            self.proj = LowRankProj(
                channels, rank=lowrank_rank,
                time_ks=lowrank_time_ks, freq_ks=lowrank_freq_ks,
                use_bias=lowrank_use_bias,
            )
            # Sparse residual: small 3×3 depthwise conv applied to magnitude
            self.sparse_conv = nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=True,
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

        # Diagnostic caches (detached, no grad)
        self._last_logits: Optional[torch.Tensor] = None
        self._last_sparse_logits: Optional[torch.Tensor] = None  # detached, diagnostics
        self._last_sparse_raw: Optional[torch.Tensor] = None     # with grad, for L1 loss

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        # magnitude: (B, C, F, TT)
        logits = self.proj(magnitude)
        # For lowrank_sparse: add sparse residual and cache both versions
        if self.gate_arch == "lowrank_sparse":
            sparse = self.sparse_conv(magnitude)
            self._last_sparse_raw = sparse             # with grad for L1 loss
            self._last_sparse_logits = sparse.detach() # detached for diagnostics
            logits = logits + sparse
        else:
            self._last_sparse_raw = None
            self._last_sparse_logits = None
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

    def get_last_lowrank_u(self) -> Optional[torch.Tensor]:
        """Return u(t) from the last forward() call (B, C, rank, T) detached, or None."""
        proj = getattr(self, "proj", None)
        if not isinstance(proj, LowRankProj):
            return None
        return proj._last_u

    def get_last_sparse_logits(self) -> Optional[torch.Tensor]:
        """Return sparse residual logits (B, C, F, T) detached from the last forward(), or None."""
        return self._last_sparse_logits
