"""WaveBandNormB – Multi-band wavelet normalization (version B).

Decomposition scheme
--------------------
SWT (stationary / undecimated wavelet transform, implemented in pure PyTorch).
Every coefficient tensor has the same length T as the input signal.

  A_J          : low-frequency approximation  →  affine normalised (mean + std)
  D_j (mid)    : mid-frequency detail(s)       →  per-level σ-normalised (zero-mean)
  D_j (high)   : high-frequency detail(s)      →  untouched (pass-through)

Reconstruction is exact by construction:  A_{j-1} = A_j + D_j

A GRU encoder-decoder predictor maps history-patch stats → future-patch stats
using residual delta prediction relative to the last history patch.

Predictor input per patch (n_mid + 5 features):
  [mu_L_local, log_sig_L_local, log_sig_Dj × n_mid, rho_H_local, mu_L_global, log_sig_L_global]

Predictor output per patch (n_mid + 2 deltas, residual from last history patch):
  [delta_mu_L, delta_log_sig_L, delta_log_sig_Dj × n_mid]

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
    """Build the à-trous (upsampled) 1-D kernel for the given level."""
    step = 2 ** (level - 1)
    n = h.shape[0]
    length = (n - 1) * step + 1
    kernel = h.new_zeros(1, 1, length)
    for i in range(n):
        kernel[0, 0, i * step] = h[i]
    return kernel  # (1, 1, length)


def _swt_fwd(
    x: Tensor,
    h_lo: Tensor,
    levels: int,
    pad_mode: str = "reflect",
) -> tuple[Tensor, list[Tensor]]:
    """Stationary WT forward (residual variant).

    A_j = conv(A_{j-1}, h_lo_upsampled);  D_j = A_{j-1} – A_j

    Args:
        x:        (B, T, C)
        h_lo:     1-D low-pass kernel  (n,)
        levels:   number of decomposition levels J
        pad_mode: padding strategy
    Returns:
        A_J:     (B, T, C)
        details: [D_1, …, D_J]  each (B, T, C)
    """
    B, T, C = x.shape
    a = x.permute(0, 2, 1).reshape(B * C, 1, T)  # (B*C, 1, T)
    details: list[Tensor] = []

    for j in range(1, levels + 1):
        kernel = _make_atrous_kernel(h_lo, j)
        K = kernel.shape[-1]
        pad = K // 2
        effective_mode = pad_mode
        if effective_mode == "reflect" and pad >= T:
            effective_mode = "replicate"
        a_pad = F.pad(a, (pad, pad), mode=effective_mode)
        a_next = F.conv1d(a_pad, kernel.to(a.device, a.dtype))[:, :, :T]
        d = a - a_next
        details.append(d.reshape(B, C, T).permute(0, 2, 1))  # (B, T, C)
        a = a_next

    A_J = a.reshape(B, C, T).permute(0, 2, 1)  # (B, T, C)
    return A_J, details  # details = [D_1, …, D_J]


def _swt_inv(A_J: Tensor, details: list[Tensor]) -> Tensor:
    """Inverse SWT.  A_{j-1} = A_j + D_j  (exact reconstruction)."""
    a = A_J
    for d in reversed(details):   # D_J, D_{J-1}, …, D_1
        a = a + d
    return a


# ---------------------------------------------------------------------------
# GRU-based predictor (residual delta prediction)
# ---------------------------------------------------------------------------

class _WavPredictor(nn.Module):
    """GRU encoder-decoder: history patch stats → future patch stats (residual deltas).

    Input  (per batch element, per channel): P_hist patches × (n_mid + 5) features
      [mu_L_local, log_sig_L_local, log_sig_Dj × n_mid, rho_H_local,
       mu_L_global, log_sig_L_global]

    Output (per batch element, per channel): P_fut patches × (n_mid + 2) absolute stats
      Computed as: anchor + delta, where anchor is last history patch stats.
      Return values are in log space for sigmas (apply exp() to get actual sigmas).
    """

    def __init__(
        self,
        P_hist: int,
        P_fut: int,
        n_mid: int,
        hidden: int = 256,
        emb_dim: int = 16,
        wave_features: int = 0,
    ):
        super().__init__()
        self.P_hist = P_hist
        self.P_fut  = P_fut
        self.n_mid  = n_mid
        self.N_IN   = n_mid + 5 + wave_features
        self.N_OUT  = n_mid + 2
        self.enc = nn.GRU(input_size=self.N_IN,  hidden_size=hidden, num_layers=1, batch_first=True)
        self.dec = nn.GRU(input_size=emb_dim,    hidden_size=hidden, num_layers=1, batch_first=True)
        self.future_embed = nn.Parameter(torch.randn(P_fut, emb_dim))
        self.head = nn.Linear(hidden, self.N_OUT)

    def forward(self, feat: Tensor, last_stat_log: Tensor) -> Tensor:
        """
        feat:          (B, P_hist, N_IN, C)
        last_stat_log: (B, N_OUT, C)  — last history patch [mu_L, log_sig_L, log_sig_Dj×n_mid]
        returns:       (B, P_fut, N_OUT, C) absolute predictions (mu_L raw, log_sig in log space)
        """
        B, P_hist, nf, C = feat.shape
        x = feat.permute(0, 3, 1, 2).reshape(B * C, P_hist, self.N_IN)  # (B*C, P_hist, N_IN)
        _, h = self.enc(x)                                               # h: (1, B*C, hidden)
        u = self.future_embed.unsqueeze(0).expand(B * C, -1, -1)        # (B*C, P_fut, emb_dim)
        out, _ = self.dec(u, h)                                          # (B*C, P_fut, hidden)
        delta = self.head(out)                                           # (B*C, P_fut, N_OUT)
        delta = delta.reshape(B, C, self.P_fut, self.N_OUT).permute(0, 2, 3, 1)
        # → (B, P_fut, N_OUT, C)

        # Residual: anchor + delta
        anchor = last_stat_log.unsqueeze(1)  # (B, 1, N_OUT, C)
        return anchor + delta                # (B, P_fut, N_OUT, C)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WaveBandNormB(nn.Module):
    """Multi-band wavelet normalization, version B.

    Args:
        seq_len:           Input look-back length T.
        pred_len:          Forecast horizon H.
        enc_in:            Number of original input channels C.
        wavelet:           "haar" | "db2".
        levels:            SWT decomposition levels J.
        mid_levels:        Tuple of detail-level indices treated as mid-band (1-indexed).
        high_levels:       Tuple of detail-level indices treated as high-band (pass-through).
        patch_len:         Patch length p.  T and H must be divisible by p.
        cond:              Conditioning: "none" | "rho_h" | "rho_all".
        sigma_min:         Minimum std clamp.
        stats_loss_weight: Scalar weight for the stats supervision term.
        pred_use_wave:     If True, append normalised waveforms to predictor input.
        ctx_patches:       Extra history patches fed to predictor.
        pad_mode:          SWT boundary padding: "reflect" | "replicate" | "circular".
        wav_use_soft_split: If True, use learnable soft band split with band_logits.
        wav_gate_tau:       Soft-split temperature (smaller → harder split).
        wav_split_reg_weight: Monotone regularisation weight for band-split gates.
        wav_split_target:   Must be "monotone".
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
        pred_use_wave: bool = False,
        ctx_patches: int = 0,
        pad_mode: str = "reflect",
        wav_use_soft_split: bool = False,
        wav_gate_tau: float = 1.0,
        wav_split_reg_weight: float = 1e-2,
        wav_split_target: str = "monotone",
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by patch_len={patch_len}")
        if pred_len % patch_len != 0:
            raise ValueError(f"pred_len={pred_len} must be divisible by patch_len={patch_len}")
        if wavelet not in _FILTER_BANKS:
            raise ValueError(f"wavelet must be one of {list(_FILTER_BANKS)}; got {wavelet!r}")
        if wav_split_target != "monotone":
            raise ValueError(f"wav_split_target must be 'monotone'; got {wav_split_target!r}")

        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.channels  = enc_in
        self.levels    = levels
        self.patch_len = patch_len
        self.cond      = cond
        self.sigma_min = sigma_min
        self.stats_loss_weight = stats_loss_weight
        self.epsilon   = 1e-6
        self.pred_use_wave = pred_use_wave
        self.ctx_patches = int(ctx_patches)
        self.pad_mode = pad_mode
        self.wav_use_soft_split = bool(wav_use_soft_split)
        self.wav_gate_tau = float(wav_gate_tau)
        self.wav_split_reg_weight = float(wav_split_reg_weight)
        self.wav_split_target = wav_split_target

        # ── Classify levels ───────────────────────────────────────────────────
        all_lvls  = set(range(1, levels + 1))
        self.high_set = frozenset(int(l) for l in high_levels if 1 <= l <= levels)
        self.mid_set  = frozenset(int(l) for l in mid_levels  if 1 <= l <= levels)
        # "other" levels (not high) default to mid
        self.mid_eff  = frozenset(l for l in all_lvls if l not in self.high_set)
        # Stable sorted list and index mapping for per-level sigma prediction
        self.mid_eff_sorted: list[int] = sorted(self.mid_eff)
        self.n_mid_eff: int = len(self.mid_eff_sorted)
        self._mid_eff_level_to_idx: dict[int, int] = {
            j: k for k, j in enumerate(self.mid_eff_sorted)
        }

        # ── Adaptive band split ───────────────────────────────────────────────
        logit_init = torch.full((levels,), -2.0)
        for l in self.high_set:
            logit_init[l - 1] = 2.0
        self.band_logits = nn.Parameter(logit_init, requires_grad=bool(wav_use_soft_split))

        # ── Filter bank ───────────────────────────────────────────────────────
        h_values = _FILTER_BANKS[wavelet]
        self.register_buffer("h_lo", torch.tensor(h_values, dtype=torch.float32))

        # ── History / future patch counts ─────────────────────────────────────
        self.P_hist = seq_len  // patch_len
        self.P_fut  = pred_len // patch_len
        self.P_hist_pred = self.P_hist + self.ctx_patches

        # ── Predictor ─────────────────────────────────────────────────────────
        wave_feat = 2 * patch_len if pred_use_wave else 0
        self.predictor = _WavPredictor(
            P_hist=self.P_hist_pred,
            P_fut=self.P_fut,
            n_mid=self.n_mid_eff,
            wave_features=wave_feat,
        )

        # ── Conditioning channels ─────────────────────────────────────────────
        if cond == "rho_h":
            self._cond_channels = enc_in
        elif cond == "rho_all":
            self._cond_channels = 3 * enc_in
        else:
            self._cond_channels = 0

        # ── Internal state ────────────────────────────────────────────────────
        self._pred_stats: Optional[dict] = None
        self._pred_raw:   Optional[Tensor] = None
        self._ctx_x: Optional[Tensor] = None
        self._oracle_future_y: Optional[Tensor] = None
        self._oracle_enabled: bool = False
        self._last_diag: dict[str, float] = {
            "sig_L_mean":       0.0,
            "sig_Dj_mean":      0.0,
            "rho_H_mean":       0.0,
            "rho_H_p10":        0.0,
            "rho_H_p90":        0.0,
            "L_stats":          0.0,
            "L_split":          0.0,
            "oracle_recon_mse": 0.0,
        }

    # ------------------------------------------------------------------
    @property
    def cond_channels(self) -> int:
        return self._cond_channels

    # ------------------------------------------------------------------
    def set_oracle_future(self, y_future: Tensor, mode: str) -> None:
        if mode in {"eval_stats", "train_eval_stats", "train_oracle"}:
            self._oracle_future_y = y_future.detach()
            self._oracle_enabled = True
        else:
            self._oracle_enabled = False
            self._oracle_future_y = None

    def clear_oracle_future(self) -> None:
        self._oracle_future_y = None
        self._oracle_enabled = False

    def set_ctx_history(self, x_ctx: Tensor) -> None:
        self._ctx_x = x_ctx.detach()

    def clear_ctx_history(self) -> None:
        self._ctx_x = None

    # ------------------------------------------------------------------
    def _patch_stats_future(
        self, signal: Tensor, compute_mean: bool = True
    ) -> tuple[Optional[Tensor], Tensor, Tensor]:
        """Patch-level stats for pred_len-length tensors. Returns (mu, sig, E)."""
        B, H, C = signal.shape
        r = signal.reshape(B, self.P_fut, self.patch_len, C)
        mu  = r.mean(dim=2) if compute_mean else None
        sig = r.std(dim=2).clamp(min=self.sigma_min)
        E   = (r ** 2).mean(dim=2)
        return mu, sig, E

    # ------------------------------------------------------------------
    def _band_w_high(self) -> Tensor:
        """Per-level high-band weight vector w_high, shape (levels,).

        wav_use_soft_split=True:  w_high[j] = sigmoid(band_logits[j] / tau) in (0,1)
        wav_use_soft_split=False: w_high[j] = 1 if level j+1 in high_set, else 0
        """
        if self.wav_use_soft_split:
            return torch.sigmoid(self.band_logits / self.wav_gate_tau)
        else:
            w = torch.zeros(self.levels, device=self.h_lo.device, dtype=self.h_lo.dtype)
            for l in self.high_set:
                w[l - 1] = 1.0
            return w

    # ------------------------------------------------------------------
    def get_split_stats(self) -> dict:
        """Return current band-split gate statistics."""
        with torch.no_grad():
            g = torch.sigmoid(self.band_logits / self.wav_gate_tau)
            g_cpu = g.detach().cpu()
            if self.wav_use_soft_split and self.levels > 1:
                split_reg = float(
                    (self.wav_split_reg_weight *
                     F.relu(g_cpu[1:] - g_cpu[:-1]).sum()).item()
                )
            else:
                split_reg = 0.0
        return {
            "g_mean":    float(g_cpu.mean().item()),
            "g_min":     float(g_cpu.min().item()),
            "g_max":     float(g_cpu.max().item()),
            "g_vec":     g_cpu.tolist(),
            "split_reg": split_reg,
        }

    # ------------------------------------------------------------------
    def _extract_patch_tokens(self, x_in: Tensor) -> dict:
        """Compute per-patch predictor tokens from a single contiguous segment.

        No cross-boundary SWT: runs _swt_fwd independently on x_in.

        Returns dict with:
            mu_L, log_sig_L, log_sig_Dj (list), rho_H   — each (B, P, C)
            sig_L, sig_Dj (list)                          — each (B, P, C)
            mu_L_global, log_sig_L_global                 — (B, 1, C)
            feat_wave                                      — (B, P, 2*patch_len, C) or None
        """
        B, T_in, C = x_in.shape
        P = T_in // self.patch_len
        eps = self.epsilon

        with torch.no_grad():
            A_J_det, details_det = _swt_fwd(x_in.detach(), self.h_lo, self.levels, self.pad_mode)
            w_high_det = self._band_w_high().detach()

        # Low-band patch stats
        r_L = A_J_det.reshape(B, P, self.patch_len, C)
        mu_L  = r_L.mean(dim=2)                                 # (B, P, C)
        sig_L = r_L.std(dim=2).clamp(min=self.sigma_min)
        E_L   = (r_L ** 2).mean(dim=2)

        # Per-level mid-band stats
        sig_Dj_list:     list[Tensor] = []
        log_sig_Dj_list: list[Tensor] = []
        E_mid = torch.zeros_like(E_L)
        for j in self.mid_eff_sorted:
            j_idx = j - 1
            r_j   = details_det[j_idx].reshape(B, P, self.patch_len, C)
            sig_j = r_j.std(dim=2).clamp(min=self.sigma_min)
            E_j   = (r_j ** 2).mean(dim=2)
            sig_Dj_list.append(sig_j)
            log_sig_Dj_list.append(torch.log(sig_j + eps))
            E_mid = E_mid + E_j

        # High-band energy ratio (using weighted high signal for consistency)
        hi_sig_det = sum(w_high_det[j] * details_det[j] for j in range(self.levels))
        r_H = hi_sig_det.reshape(B, P, self.patch_len, C)
        E_H = (r_H ** 2).mean(dim=2)
        E_tot = E_L + E_mid + E_H + eps
        rho_H = E_H / E_tot

        # Global A_J stats (across all time in segment)
        mu_L_global  = A_J_det.mean(dim=1, keepdim=True)                             # (B, 1, C)
        sig_L_global = A_J_det.std(dim=1, keepdim=True).clamp(min=self.sigma_min)    # (B, 1, C)

        tokens: dict = {
            "mu_L":             mu_L,
            "log_sig_L":        torch.log(sig_L + eps),
            "log_sig_Dj":       log_sig_Dj_list,
            "rho_H":            rho_H,
            "sig_L":            sig_L,
            "sig_Dj":           sig_Dj_list,
            "mu_L_global":      mu_L_global,
            "log_sig_L_global": torch.log(sig_L_global + eps),
            "feat_wave":        None,
        }

        if self.pred_use_wave:
            low_z = (r_L - mu_L.unsqueeze(2)) / (sig_L.unsqueeze(2) + eps)
            # Use first mid level for second wave feature
            j0    = self.mid_eff_sorted[0]
            r_j0  = details_det[j0 - 1].reshape(B, P, self.patch_len, C)
            mid0_z = r_j0 / (sig_Dj_list[0].unsqueeze(2) + eps)
            tokens["feat_wave"] = torch.cat([low_z, mid0_z], dim=2)  # (B, P, 2*patch_len, C)

        return tokens

    # ------------------------------------------------------------------
    def normalize(self, x: Tensor) -> Tensor:
        """(B, T, C) → (B, T, C + cond_channels).

        Normalises wavelet coefficients using current-window stats.
        Runs predictor to cache self._pred_stats for use by denormalize() / loss().
        """
        B, T, C = x.shape
        eps = self.epsilon
        h   = self.h_lo

        # ── 1. SWT forward ────────────────────────────────────────────────────
        w_high = self._band_w_high()  # (levels,) — may carry grad if learned

        with torch.no_grad():
            A_J_det, details_det = _swt_fwd(x.detach(), h, self.levels, self.pad_mode)
            w_high_det = w_high.detach()

        # Full-gradient SWT for reconstruction path
        A_J, details = _swt_fwd(x, h, self.levels, self.pad_mode)

        # ── 2. Current-window patch statistics ────────────────────────────────
        r_L   = A_J_det.reshape(B, self.P_hist, self.patch_len, C)
        mu_L  = r_L.mean(dim=2)                                      # (B, P_hist, C)
        sig_L = r_L.std(dim=2).clamp(min=self.sigma_min)

        sig_Dj_hist: list[Tensor] = []
        for j in self.mid_eff_sorted:
            j_idx = j - 1
            r_j   = details_det[j_idx].reshape(B, self.P_hist, self.patch_len, C)
            sig_j = r_j.std(dim=2).clamp(min=self.sigma_min)
            sig_Dj_hist.append(sig_j)

        # Diagnostic: rho_H
        hi_sig_det = sum(w_high_det[j] * details_det[j] for j in range(self.levels))
        r_H  = hi_sig_det.reshape(B, self.P_hist, self.patch_len, C)
        E_L  = (r_L ** 2).mean(dim=2)
        E_H  = (r_H ** 2).mean(dim=2)
        E_mid = torch.zeros_like(E_L)
        for j in self.mid_eff_sorted:
            r_j_e  = details_det[j - 1].reshape(B, self.P_hist, self.patch_len, C)
            E_mid += (r_j_e ** 2).mean(dim=2)
        E_tot  = E_L + E_mid + E_H + eps
        rho_H  = E_H / E_tot

        # ── Diagnostic cache ──────────────────────────────────────────────────
        with torch.no_grad():
            self._last_diag["sig_L_mean"] = float(sig_L.mean().item())
            self._last_diag["sig_Dj_mean"] = float(
                sum(s.mean().item() for s in sig_Dj_hist) / max(self.n_mid_eff, 1)
            )
            rho_H_flat = rho_H.reshape(-1)
            self._last_diag["rho_H_mean"] = float(rho_H_flat.mean().item())
            self._last_diag["rho_H_p10"]  = float(torch.quantile(rho_H_flat, 0.10).item())
            self._last_diag["rho_H_p90"]  = float(torch.quantile(rho_H_flat, 0.90).item())

        # ── 3. Predictor tokens (no cross-boundary SWT) ───────────────────────
        tok_x = self._extract_patch_tokens(x)  # P_hist patches from current window

        if self._ctx_x is not None and self.ctx_patches > 0:
            tok_ctx = self._extract_patch_tokens(self._ctx_x)
            P_ctx   = self._ctx_x.shape[1] // self.patch_len
            if P_ctx != self.ctx_patches:
                raise ValueError(
                    f"ctx segment has P_ctx={P_ctx} patches but ctx_patches={self.ctx_patches}."
                )
            mu_L_p      = torch.cat([tok_ctx["mu_L"],      tok_x["mu_L"]],      dim=1)
            log_sig_L_p = torch.cat([tok_ctx["log_sig_L"], tok_x["log_sig_L"]], dim=1)
            log_sig_Dj_p = [
                torch.cat([tok_ctx["log_sig_Dj"][k], tok_x["log_sig_Dj"][k]], dim=1)
                for k in range(self.n_mid_eff)
            ]
            rho_H_p       = torch.cat([tok_ctx["rho_H"],     tok_x["rho_H"]],     dim=1)
            mu_L_global   = tok_x["mu_L_global"]
            log_sig_L_glb = tok_x["log_sig_L_global"]
            if self.pred_use_wave:
                feat_wave = torch.cat([tok_ctx["feat_wave"], tok_x["feat_wave"]], dim=1)
        else:
            mu_L_p        = tok_x["mu_L"]
            log_sig_L_p   = tok_x["log_sig_L"]
            log_sig_Dj_p  = tok_x["log_sig_Dj"]
            rho_H_p       = tok_x["rho_H"]
            mu_L_global   = tok_x["mu_L_global"]
            log_sig_L_glb = tok_x["log_sig_L_global"]
            if self.pred_use_wave:
                feat_wave = tok_x["feat_wave"]

        # Broadcast global stats to all patches
        P_pred = mu_L_p.shape[1]
        mu_L_global_exp   = mu_L_global.expand(B, P_pred, C)
        log_sig_L_glb_exp = log_sig_L_glb.expand(B, P_pred, C)

        # Stack: [mu_L, log_sig_L, log_sig_Dj × n_mid, rho_H, mu_L_global, log_sig_L_global]
        feat_list  = [mu_L_p, log_sig_L_p] + log_sig_Dj_p + [rho_H_p, mu_L_global_exp, log_sig_L_glb_exp]
        feat_stats = torch.stack(feat_list, dim=2)   # (B, P_pred, N_IN, C)

        if self.pred_use_wave:
            feat = torch.cat([feat_stats, feat_wave], dim=2)
        else:
            feat = feat_stats   # (B, P_pred, N_IN, C)

        # Last history patch anchor for residual prediction: [mu_L, log_sig_L, log_sig_Dj × n_mid]
        last_parts  = [mu_L_p[:, -1, :], log_sig_L_p[:, -1, :]] + \
                      [log_sig_Dj_p[k][:, -1, :] for k in range(self.n_mid_eff)]
        last_stat_log = torch.stack(last_parts, dim=1)   # (B, N_OUT, C)

        pred_raw = self.predictor(feat, last_stat_log)    # (B, P_fut, N_OUT, C)
        self._pred_raw = pred_raw

        # Activate predictions (index 0: mu_L raw; 1..: log-space → exp for sigma)
        mu_L_f  = pred_raw[:, :, 0, :]                                           # (B, P_fut, C)
        sig_L_f = torch.exp(pred_raw[:, :, 1, :]).clamp(min=self.sigma_min)      # (B, P_fut, C)
        sig_Dj_f = [
            torch.exp(pred_raw[:, :, 2 + k, :]).clamp(min=self.sigma_min)
            for k in range(self.n_mid_eff)
        ]

        self._pred_stats = {
            "mu_L":   mu_L_f,
            "sig_L":  sig_L_f,
            "sig_Dj": sig_Dj_f,
        }

        # ── 4. Normalise wavelet coefficients ─────────────────────────────────
        mu_L_rep  = mu_L.repeat_interleave(self.patch_len, dim=1)    # (B, T, C)
        sig_L_rep = sig_L.repeat_interleave(self.patch_len, dim=1)
        A_J_norm  = (A_J - mu_L_rep) / (sig_L_rep + eps)

        details_norm: list[Tensor] = []
        for j_idx, d in enumerate(details):
            j = j_idx + 1  # 1-indexed level
            if j in self._mid_eff_level_to_idx:
                k         = self._mid_eff_level_to_idx[j]
                sig_Dj_rep = sig_Dj_hist[k].repeat_interleave(self.patch_len, dim=1)
                w_j        = w_high[j_idx]
                scale_j    = (1.0 - w_j) * (sig_Dj_rep + eps) + w_j * 1.0
            else:
                # High level: pass-through
                scale_j = torch.ones(1, device=d.device, dtype=d.dtype)
            details_norm.append(d / scale_j)

        x_norm = _swt_inv(A_J_norm, details_norm)   # (B, T, C)

        # ── 5. Conditioning ───────────────────────────────────────────────────
        if self.cond == "rho_h":
            cond_feat = rho_H.repeat_interleave(self.patch_len, dim=1)
            return torch.cat([x_norm, cond_feat], dim=-1)
        elif self.cond == "rho_all":
            rho_L = E_L / E_tot
            rho_M = E_mid / E_tot
            rho_H_rep = rho_H.repeat_interleave(self.patch_len, dim=1)
            rho_L_rep = rho_L.repeat_interleave(self.patch_len, dim=1)
            rho_M_rep = rho_M.repeat_interleave(self.patch_len, dim=1)
            cond_feat = torch.cat([rho_L_rep, rho_M_rep, rho_H_rep], dim=-1)
            return torch.cat([x_norm, cond_feat], dim=-1)
        else:
            return x_norm

    # ------------------------------------------------------------------
    def denormalize(self, y_norm: Tensor, station_pred=None) -> Tensor:
        """(B, H, C+cond) → (B, H, C).

        Inverts band-wise normalisation using oracle stats (if enabled)
        or predictor stats from self._pred_stats.
        """
        if self._pred_stats is None and not self._oracle_enabled:
            return y_norm[:, :, :self.channels]

        B, H, C_full = y_norm.shape
        C   = self.channels
        eps = self.epsilon

        y       = y_norm[:, :, :C]           # strip cond channels
        w_high  = self._band_w_high()        # (levels,)

        if self._oracle_enabled and self._oracle_future_y is not None:
            # Oracle path: compute per-level true future stats
            y_true = self._oracle_future_y.detach()
            with torch.no_grad():
                A_J_t, details_t = _swt_fwd(y_true, self.h_lo, self.levels, self.pad_mode)
                mu_L_f, sig_L_f, _ = self._patch_stats_future(A_J_t)
                sig_Dj_f: list[Tensor] = []
                for j in self.mid_eff_sorted:
                    r_j   = details_t[j - 1].reshape(B, self.P_fut, self.patch_len, C)
                    sig_j = r_j.std(dim=2).clamp(min=self.sigma_min)
                    sig_Dj_f.append(sig_j)
        else:
            # Predictor path
            ps      = self._pred_stats
            mu_L_f  = ps["mu_L"]
            sig_L_f = ps["sig_L"]
            sig_Dj_f = ps["sig_Dj"]

        mu_L_rep  = mu_L_f.repeat_interleave(self.patch_len, dim=1)
        sig_L_rep = sig_L_f.repeat_interleave(self.patch_len, dim=1)
        sig_Dj_rep = [s.repeat_interleave(self.patch_len, dim=1) for s in sig_Dj_f]

        # SWT forward on normalised prediction
        A_J_n, details_n = _swt_fwd(y, self.h_lo, self.levels, self.pad_mode)

        A_J_hat = A_J_n * (sig_L_rep + eps) + mu_L_rep

        details_hat: list[Tensor] = []
        for j_idx, d in enumerate(details_n):
            j = j_idx + 1
            if j in self._mid_eff_level_to_idx:
                k       = self._mid_eff_level_to_idx[j]
                w_j     = w_high[j_idx]
                scale_j = (1.0 - w_j) * (sig_Dj_rep[k] + eps) + w_j * 1.0
            else:
                scale_j = 1.0   # high level: pass-through
            details_hat.append(d * scale_j)

        y_hat = _swt_inv(A_J_hat, details_hat)

        # Oracle consistency self-check
        if self._oracle_enabled:
            with torch.no_grad():
                A_J_chk, details_chk = _swt_fwd(y_hat.detach(), self.h_lo, self.levels, self.pad_mode)
                A_J_norm_chk = (A_J_chk - mu_L_rep) / (sig_L_rep + eps)
                details_norm_chk: list[Tensor] = []
                for j_idx, d in enumerate(details_chk):
                    j = j_idx + 1
                    if j in self._mid_eff_level_to_idx:
                        k       = self._mid_eff_level_to_idx[j]
                        w_j     = w_high[j_idx].detach()
                        scale_j = (1.0 - w_j) * (sig_Dj_rep[k] + eps) + w_j * 1.0
                    else:
                        scale_j = 1.0
                    details_norm_chk.append(d / scale_j)
                y_renorm = _swt_inv(A_J_norm_chk, details_norm_chk)
                self._last_diag["oracle_recon_mse"] = float(
                    F.mse_loss(y_renorm, y.detach()).item()
                )

        return y_hat

    # ------------------------------------------------------------------
    def loss(self, true: Tensor) -> Tensor:
        """Stats supervision loss: L_mu_L + L_log_sig_L + Σ L_log_sig_Dj + L_split."""
        if self._pred_stats is None:
            return torch.tensor(0.0, device=true.device)

        B, H, C = true.shape
        eps = self.epsilon

        # Oracle future stats (no grad through supervision targets)
        with torch.no_grad():
            A_J_t, details_t = _swt_fwd(true.detach(), self.h_lo, self.levels, self.pad_mode)
            mu_L_true, sig_L_true, _ = self._patch_stats_future(A_J_t)
            sig_Dj_true: list[Tensor] = []
            for j in self.mid_eff_sorted:
                r_j   = details_t[j - 1].reshape(B, self.P_fut, self.patch_len, C)
                sig_j = r_j.std(dim=2).clamp(min=self.sigma_min)
                sig_Dj_true.append(sig_j)

        ps = self._pred_stats

        L_mu_L     = F.mse_loss(ps["mu_L"],  mu_L_true)
        L_log_sig_L = F.mse_loss(
            torch.log(ps["sig_L"] + eps), torch.log(sig_L_true + eps)
        )
        L_log_sig_Dj = sum(
            F.mse_loss(torch.log(ps["sig_Dj"][k] + eps), torch.log(sig_Dj_true[k] + eps))
            for k in range(self.n_mid_eff)
        ) if self.n_mid_eff > 0 else torch.tensor(0.0, device=true.device)

        L_stats = L_mu_L + L_log_sig_L + L_log_sig_Dj

        # Monotone regularisation on band-split gates
        if self.wav_use_soft_split and self.levels > 1:
            g       = torch.sigmoid(self.band_logits / self.wav_gate_tau)
            L_split = self.wav_split_reg_weight * F.relu(g[1:] - g[:-1]).sum()
        else:
            L_split = torch.tensor(0.0, device=true.device)

        self._last_diag["L_stats"] = float((self.stats_loss_weight * L_stats).item())
        self._last_diag["L_split"] = float(L_split.item())

        return self.stats_loss_weight * L_stats + L_split

    # ------------------------------------------------------------------
    def get_last_diag(self) -> dict[str, float]:
        return dict(self._last_diag)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, mode: str = "n", station_pred=None) -> Tensor:
        if mode == "n":
            return self.normalize(x)
        elif mode == "d":
            return self.denormalize(x, station_pred)
        return self.normalize(x)
