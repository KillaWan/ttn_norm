"""WaveBandNormB – Multi-band wavelet normalization (version B).

Decomposition scheme
--------------------
SWT (stationary / undecimated wavelet transform, implemented in pure PyTorch).
Every coefficient tensor has the same length T as the input signal.

  A_J          : low-frequency approximation  →  patch mean + std normalised
  D_j (mid)    : mid-frequency detail(s)       →  patch std normalised (zero-mean)
  D_j (high)   : high-frequency detail(s)      →  untouched

Reconstruction is exact by construction:  A_{j-1} = A_j + D_j

A lightweight MLP predictor maps history-patch stats → future-patch stats so that
denormalize() can invert the normalisation applied to the backbone output.

Interface (compatible with TTNModel generic fallback)
------------------------------------------------------
  normalize(x)         → x_norm_cat  (extra cond channels appended when cond!="none")
  denormalize(y_norm)  → y_hat
  loss(true)           → scalar aux loss
  forward(x)           → normalize(x)          (called by TTNModel generic path)
  cond_channels        → int, extra channels appended by normalize()
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Low-pass filter banks (sum-normalised so A_j ≈ local mean)
# ---------------------------------------------------------------------------
_FILTER_BANKS: dict[str, list[float]] = {
    "haar": [0.5, 0.5],
    # db2 normalised so coefficients sum to 1
    "db2":  [x / math.sqrt(2) for x in
             [0.48296291314453025, 0.8365163037378077,
              0.22414386804201339, -0.12940952255126034]],
}


# ---------------------------------------------------------------------------
# SWT helpers  (residual decomposition → trivially invertible)
# ---------------------------------------------------------------------------

def _make_atrous_kernel(h: Tensor, level: int) -> Tensor:
    """Build the à-trous (upsampled) 1-D kernel for the given level.
    Level 1 → no zeros inserted; level j → 2^(j-1)-1 zeros between taps.
    """
    step = 2 ** (level - 1)
    n = h.shape[0]
    length = (n - 1) * step + 1
    kernel = h.new_zeros(1, 1, length)
    for i in range(n):
        kernel[0, 0, i * step] = h[i]
    return kernel  # (1, 1, length)


def _swt_fwd(x: Tensor, h_lo: Tensor, levels: int) -> tuple[Tensor, list[Tensor]]:
    """Stationary WT forward (residual variant).

    A_j   = circular_conv(A_{j-1}, h_lo_upsampled)
    D_j   = A_{j-1} – A_j          ← exact reconstruction guaranteed

    Args:
        x:      (B, T, C)
        h_lo:   1-D low-pass kernel  (n,)
        levels: number of decomposition levels J
    Returns:
        A_J:    (B, T, C)
        details: [D_1, …, D_J]  each (B, T, C)
    """
    B, T, C = x.shape
    a = x.permute(0, 2, 1).reshape(B * C, 1, T)  # (B*C, 1, T)
    details: list[Tensor] = []

    for j in range(1, levels + 1):
        kernel = _make_atrous_kernel(h_lo, j)  # (1, 1, K)
        K = kernel.shape[-1]
        pad = K // 2
        a_pad = F.pad(a, (pad, pad), mode="circular")
        a_next = F.conv1d(a_pad, kernel.to(a.device, a.dtype))[:, :, :T]
        d = a - a_next
        details.append(d.reshape(B, C, T).permute(0, 2, 1))  # (B, T, C)
        a = a_next

    A_J = a.reshape(B, C, T).permute(0, 2, 1)  # (B, T, C)
    return A_J, details  # details = [D_1, …, D_J]


def _swt_inv(A_J: Tensor, details: list[Tensor]) -> Tensor:
    """Inverse SWT.  A_{j-1} = A_j + D_j  (exact reconstruction).

    Args:
        A_J:     (B, T, C)
        details: [D_1, …, D_J]  (j=1 is finest)
    Returns:
        x: (B, T, C)
    """
    a = A_J
    for d in reversed(details):   # D_J, D_{J-1}, …, D_1
        a = a + d
    return a


# ---------------------------------------------------------------------------
# Lightweight MLP predictor
# ---------------------------------------------------------------------------

class _WavPredictor(nn.Module):
    """Per-channel MLP: history patch stats → future patch stats.

    Input  (per batch element, per channel): P_hist patches × 4 features
      [mu_L, log_sig_L, log_sig_M, rho_H]
    Output (per batch element, per channel): P_fut patches × 6 raw logits
      [mu_L,  log_sigL_raw, log_sigM_raw, rho_L_raw, rho_M_raw, rho_H_raw]
    Activations applied by the caller:
      sig_L = softplus(log_sigL_raw).clamp(min=sigma_min)
      sig_M = softplus(log_sigM_raw).clamp(min=sigma_min)
      rho   = softmax([rho_L_raw, rho_M_raw, rho_H_raw], dim=-1)
    """

    N_IN  = 4
    N_OUT = 6

    def __init__(self, P_hist: int, P_fut: int, hidden: int = 256):
        super().__init__()
        self.P_hist = P_hist
        self.P_fut  = P_fut
        self.fc1 = nn.Linear(P_hist * self.N_IN, hidden)
        self.fc2 = nn.Linear(hidden, P_fut * self.N_OUT)

    def forward(self, feat: Tensor) -> Tensor:
        """
        feat: (B, P_hist, 4, C)
        returns: (B, P_fut, 6, C)
        """
        B, P, F, C = feat.shape
        x = feat.permute(0, 3, 1, 2).reshape(B, C, P * F)  # (B, C, P*4)
        x = F.relu(self.fc1(x))                              # (B, C, hidden)
        x = self.fc2(x)                                      # (B, C, P_fut*6)
        return x.reshape(B, C, self.P_fut, self.N_OUT).permute(0, 2, 3, 1)
        # → (B, P_fut, 6, C)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WaveBandNormB(nn.Module):
    """Multi-band wavelet normalization, version B.

    Args:
        seq_len:     Input look-back length T.
        pred_len:    Forecast horizon H.
        enc_in:      Number of original input channels C.
        wavelet:     "haar" | "db2"  (low-pass filter choice).
        levels:      SWT decomposition levels J.
        mid_levels:  Tuple of detail-level indices treated as mid-band.
                     Level 1 = finest detail.  Must be subset of 1..J.
        high_levels: Tuple of detail-level indices treated as high-band.
        patch_len:   Patch length p.  T and H must be divisible by p.
        cond:        Conditioning signal to append: "none" | "rho_h" | "rho_all".
        sigma_min:   Minimum std clamp.
        stats_loss_weight: Scalar weight for the stats supervision term.
        rho_loss_weight:   Scalar weight for the energy-ratio supervision term.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        wavelet: str = "haar",
        levels: int = 3,
        mid_levels: tuple[int, ...] = (2, 3),
        high_levels: tuple[int, ...] = (1,),
        patch_len: int = 24,
        cond: str = "rho_h",
        sigma_min: float = 1e-3,
        stats_loss_weight: float = 0.1,
        rho_loss_weight: float = 0.1,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by patch_len={patch_len}")
        if pred_len % patch_len != 0:
            raise ValueError(f"pred_len={pred_len} must be divisible by patch_len={patch_len}")
        if wavelet not in _FILTER_BANKS:
            raise ValueError(f"wavelet must be one of {list(_FILTER_BANKS)}; got {wavelet!r}")

        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.channels  = enc_in
        self.levels    = levels
        self.patch_len = patch_len
        self.cond      = cond
        self.sigma_min = sigma_min
        self.stats_loss_weight = stats_loss_weight
        self.rho_loss_weight   = rho_loss_weight
        self.epsilon   = 1e-6

        # ── Classify levels ───────────────────────────────────────────────────
        all_lvls  = set(range(1, levels + 1))
        self.high_set = frozenset(int(l) for l in high_levels if 1 <= l <= levels)
        self.mid_set  = frozenset(int(l) for l in mid_levels  if 1 <= l <= levels)
        # "other" levels (not high, not explicitly mid) default to mid
        self.mid_eff  = frozenset(l for l in all_lvls if l not in self.high_set)
        # (mid_eff = mid_set ∪ "others"; high_eff = high_set)

        # ── Filter bank (registered as buffer so it moves with .to(device)) ──
        h_values = _FILTER_BANKS[wavelet]
        self.register_buffer("h_lo", torch.tensor(h_values, dtype=torch.float32))

        # ── History / future patch counts ─────────────────────────────────────
        self.P_hist = seq_len  // patch_len
        self.P_fut  = pred_len // patch_len

        # ── Predictor ─────────────────────────────────────────────────────────
        self.predictor = _WavPredictor(self.P_hist, self.P_fut)

        # ── Condition channels ────────────────────────────────────────────────
        if cond == "rho_h":
            self._cond_channels = enc_in          # 1 rho value per original channel
        elif cond == "rho_all":
            self._cond_channels = 3 * enc_in      # L + M + H ratios
        else:
            self._cond_channels = 0

        # ── Internal state ────────────────────────────────────────────────────
        self._pred_stats: Optional[dict] = None   # set by normalize()
        self._pred_raw:   Optional[Tensor] = None  # cached predictor output
        # Diagnostic cache (float scalars, updated each call)
        self._last_diag: dict[str, float] = {
            "sig_L_mean": 0.0, "sig_M_mean": 0.0,
            "rho_H_mean": 0.0, "rho_H_p10": 0.0, "rho_H_p90": 0.0,
            "L_stats": 0.0, "L_rho": 0.0,
        }

    # ------------------------------------------------------------------
    @property
    def cond_channels(self) -> int:
        """Extra channels appended to x_norm by normalize()."""
        return self._cond_channels

    # ------------------------------------------------------------------
    def _patch_stats(
        self, signal: Tensor, compute_mean: bool = True
    ) -> tuple[Optional[Tensor], Tensor, Tensor]:
        """Compute patch-level mean, std, and RMS energy of *signal* (B, T, C).

        Returns:
            mu:  (B, P, C) or None if compute_mean=False
            sig: (B, P, C) clamped
            E:   (B, P, C) mean-square energy
        """
        B, T, C = signal.shape
        r = signal.reshape(B, self.P_hist, self.patch_len, C)
        mu  = r.mean(dim=2) if compute_mean else None
        sig = r.std(dim=2).clamp(min=self.sigma_min)
        E   = (r ** 2).mean(dim=2)
        return mu, sig, E

    def _patch_stats_future(
        self, signal: Tensor, compute_mean: bool = True
    ) -> tuple[Optional[Tensor], Tensor, Tensor]:
        """Same as _patch_stats but for pred_len-length tensors."""
        B, H, C = signal.shape
        r = signal.reshape(B, self.P_fut, self.patch_len, C)
        mu  = r.mean(dim=2) if compute_mean else None
        sig = r.std(dim=2).clamp(min=self.sigma_min)
        E   = (r ** 2).mean(dim=2)
        return mu, sig, E

    # ------------------------------------------------------------------
    def _build_mid_signal(self, details: list[Tensor]) -> Tensor:
        """Sum all mid-effective detail coefficients. Returns (B, T, C)."""
        out = torch.zeros_like(details[0])
        for j_idx, d in enumerate(details):
            j = j_idx + 1  # 1-indexed
            if j in self.mid_eff:
                out = out + d
        return out

    def _build_high_signal(self, details: list[Tensor]) -> Tensor:
        """Sum all high-band detail coefficients. Returns (B, T, C)."""
        out = torch.zeros_like(details[0])
        for j_idx, d in enumerate(details):
            j = j_idx + 1
            if j in self.high_set:
                out = out + d
        return out

    # ------------------------------------------------------------------
    def normalize(self, x: Tensor) -> Tensor:
        """(B, T, C) → (B, T, C + cond_channels).

        Caches self._pred_stats and self._pred_raw for use by
        denormalize() and loss().
        """
        B, T, C = x.shape
        eps = self.epsilon
        h   = self.h_lo

        # ── 1. SWT forward ────────────────────────────────────────────────────
        with torch.no_grad():
            # Stats are computed on detached input to avoid double-differentiating
            A_J_det, details_det = _swt_fwd(x.detach(), h, self.levels)
            mid_sig_det = self._build_mid_signal(details_det)
            hi_sig_det  = self._build_high_signal(details_det)

        # Full-gradient SWT for reconstruction path
        A_J, details = _swt_fwd(x, h, self.levels)

        # ── 2. History patch statistics ───────────────────────────────────────
        mu_L,  sig_L, E_L = self._patch_stats(A_J_det, compute_mean=True)
        _,     sig_M, E_M = self._patch_stats(mid_sig_det, compute_mean=False)
        _,     _,     E_H = self._patch_stats(hi_sig_det,  compute_mean=False)

        E_tot  = E_L + E_M + E_H + eps
        rho_L  = E_L / E_tot   # (B, P_hist, C)
        rho_M  = E_M / E_tot
        rho_H  = E_H / E_tot

        # ── Diagnostic cache (history-side stats, detached) ───────────────────
        with torch.no_grad():
            rho_H_flat = rho_H.reshape(-1)
            self._last_diag["sig_L_mean"] = float(sig_L.mean().item())
            self._last_diag["sig_M_mean"] = float(sig_M.mean().item())
            self._last_diag["rho_H_mean"] = float(rho_H_flat.mean().item())
            self._last_diag["rho_H_p10"]  = float(torch.quantile(rho_H_flat, 0.10).item())
            self._last_diag["rho_H_p90"]  = float(torch.quantile(rho_H_flat, 0.90).item())

        # ── 3. Predictor ──────────────────────────────────────────────────────
        feat = torch.stack(
            [mu_L, torch.log(sig_L + eps), torch.log(sig_M + eps), rho_H],
            dim=2,
        )  # (B, P_hist, 4, C)
        pred_raw = self.predictor(feat)   # (B, P_fut, 6, C)
        self._pred_raw = pred_raw

        # Split and activate
        mu_L_f   = pred_raw[:, :, 0, :]                             # (B, P_fut, C)
        sig_L_f  = F.softplus(pred_raw[:, :, 1, :]).clamp(min=self.sigma_min)
        sig_M_f  = F.softplus(pred_raw[:, :, 2, :]).clamp(min=self.sigma_min)
        rho_raw  = pred_raw[:, :, 3:, :]                            # (B, P_fut, 3, C)
        rho_f    = torch.softmax(rho_raw, dim=2)                    # (B, P_fut, 3, C)

        self._pred_stats = {
            "mu_L":  mu_L_f,
            "sig_L": sig_L_f,
            "sig_M": sig_M_f,
            "rho":   rho_f,
        }

        # ── 4. Patch-level stat vectors expanded to full length ───────────────
        mu_L_rep  = mu_L.repeat_interleave(self.patch_len, dim=1)   # (B, T, C)
        sig_L_rep = sig_L.repeat_interleave(self.patch_len, dim=1)
        sig_M_rep = sig_M.repeat_interleave(self.patch_len, dim=1)

        # ── 5. Normalise wavelet coefficients ─────────────────────────────────
        A_J_norm = (A_J - mu_L_rep) / (sig_L_rep + eps)

        details_norm: list[Tensor] = []
        for j_idx, d in enumerate(details):
            j = j_idx + 1
            if j in self.high_set:
                details_norm.append(d)                   # high: unchanged
            else:
                details_norm.append(d / (sig_M_rep + eps))  # mid (or other→mid)

        # ── 6. Reconstruct normalised time-domain signal ──────────────────────
        x_norm = _swt_inv(A_J_norm, details_norm)   # (B, T, C)

        # ── 7. Conditioning ───────────────────────────────────────────────────
        if self.cond == "rho_h":
            cond_feat = rho_H.repeat_interleave(self.patch_len, dim=1)  # (B, T, C)
            return torch.cat([x_norm, cond_feat], dim=-1)               # (B, T, 2C)
        elif self.cond == "rho_all":
            rho_H_rep = rho_H.repeat_interleave(self.patch_len, dim=1)
            rho_L_rep = rho_L.repeat_interleave(self.patch_len, dim=1)
            rho_M_rep = rho_M.repeat_interleave(self.patch_len, dim=1)
            cond_feat = torch.cat([rho_L_rep, rho_M_rep, rho_H_rep], dim=-1)
            return torch.cat([x_norm, cond_feat], dim=-1)               # (B, T, 4C)
        else:
            return x_norm                                                # (B, T, C)

    # ------------------------------------------------------------------
    def denormalize(self, y_norm: Tensor, station_pred=None) -> Tensor:
        """(B, H, C+cond) → (B, H, C).

        Applies the inverse band-wise normalisation to the backbone output
        using predicted future stats from self._pred_stats.
        """
        if self._pred_stats is None:
            # Fallback: strip cond channels and return
            return y_norm[:, :, :self.channels]

        B, H, C_full = y_norm.shape
        C = self.channels
        eps = self.epsilon

        # Strip cond channels
        y = y_norm[:, :, :C]   # (B, H, C)

        ps = self._pred_stats
        mu_L_rep  = ps["mu_L"].repeat_interleave(self.patch_len, dim=1)   # (B, H, C)
        sig_L_rep = ps["sig_L"].repeat_interleave(self.patch_len, dim=1)
        sig_M_rep = ps["sig_M"].repeat_interleave(self.patch_len, dim=1)

        # SWT forward on normalised prediction
        A_J_n, details_n = _swt_fwd(y, self.h_lo, self.levels)

        # Invert band normalisation
        A_J_hat = A_J_n * (sig_L_rep + eps) + mu_L_rep

        details_hat: list[Tensor] = []
        for j_idx, d in enumerate(details_n):
            j = j_idx + 1
            if j in self.high_set:
                details_hat.append(d)                       # high: unchanged
            else:
                details_hat.append(d * (sig_M_rep + eps))  # mid: un-scale

        return _swt_inv(A_J_hat, details_hat)   # (B, H, C)

    # ------------------------------------------------------------------
    def loss(self, true: Tensor) -> Tensor:
        """Multi-band stats supervision loss.

        Returns  stats_loss_weight * L_stats  +  rho_loss_weight * L_rho.
        """
        if self._pred_stats is None or self._pred_raw is None:
            return torch.tensor(0.0, device=true.device)

        B, H, C = true.shape
        eps = self.epsilon

        # SWT on future ground-truth (detached — supervision signal only)
        with torch.no_grad():
            A_J_t, details_t = _swt_fwd(true.detach(), self.h_lo, self.levels)
            mid_sig_t = self._build_mid_signal(details_t)

        # Future patch oracle stats
        mu_L_true, sig_L_true, E_L_true = self._patch_stats_future(A_J_t)
        _,         sig_M_true, E_M_true = self._patch_stats_future(mid_sig_t, compute_mean=False)
        hi_sig_t  = self._build_high_signal(details_t)
        _,         _,          E_H_true = self._patch_stats_future(hi_sig_t,  compute_mean=False)

        E_tot_t = E_L_true + E_M_true + E_H_true + eps
        rho_true = torch.stack(
            [E_L_true / E_tot_t, E_M_true / E_tot_t, E_H_true / E_tot_t],
            dim=2,
        )  # (B, P_fut, 3, C)

        # Predicted stats
        ps = self._pred_stats
        raw = self._pred_raw  # (B, P_fut, 6, C)

        mu_L_pred  = ps["mu_L"]
        sig_L_pred = ps["sig_L"]
        sig_M_pred = ps["sig_M"]

        # Stats loss (MSE on means, log-MSE on stds)
        L_mu   = F.mse_loss(mu_L_pred,  mu_L_true)
        L_sigL = F.mse_loss(torch.log(sig_L_pred + eps), torch.log(sig_L_true + eps))
        L_sigM = F.mse_loss(torch.log(sig_M_pred + eps), torch.log(sig_M_true + eps))
        L_stats = L_mu + L_sigL + L_sigM

        # Rho loss (KL divergence: rho_true || rho_pred using log-softmax)
        rho_logits = raw[:, :, 3:, :]         # (B, P_fut, 3, C)
        # KL(target || softmax_pred): −sum target * log_softmax(pred)
        log_rho_pred = F.log_softmax(rho_logits, dim=2)  # (B, P_fut, 3, C)
        L_rho = -(rho_true * log_rho_pred).sum(dim=2).mean()

        # Update diagnostic cache (detached scalars)
        self._last_diag["L_stats"] = float((self.stats_loss_weight * L_stats).item())
        self._last_diag["L_rho"]   = float((self.rho_loss_weight   * L_rho).item())

        return self.stats_loss_weight * L_stats + self.rho_loss_weight * L_rho

    # ------------------------------------------------------------------
    def get_last_diag(self) -> dict[str, float]:
        """Return diagnostic scalars from the most recent normalize() / loss() call."""
        return dict(self._last_diag)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, mode: str = "n", station_pred=None) -> Tensor:
        """forward(x) → normalize(x)  [TTNModel generic fallback path]."""
        if mode == "n":
            return self.normalize(x)
        elif mode == "d":
            return self.denormalize(x, station_pred)
        return self.normalize(x)
