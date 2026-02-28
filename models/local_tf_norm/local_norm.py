from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stft import STFT
from .gate import LocalTFGate


@dataclass
class LocalTFNormState:
    x_tf: torch.Tensor
    n_tf: torch.Tensor
    g_local: torch.Tensor
    n_time: torch.Tensor
    length: int
    mean: Optional[torch.Tensor] = None   # (B, 1, C) instance mean for RevIN
    std: Optional[torch.Tensor] = None    # (B, 1, C) instance std for RevIN


class LocalTFNorm(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        n_fft: int | None = None,
        hop_length: int | None = None,
        win_length: int | None = None,
        gate_type: str = "depthwise",
        gate_log_mag: bool = True,
        gate_arch: str = "pointwise",
        gate_threshold_mode: str = "shift",
        gate_use_log_mag: Optional[bool] = None,
        future_mode: str = "repeat_last",   # also accepts "pred"
        teacher_mask_only: bool = False,
        n_pred_weight: float = 0.0,
        n_pred_arch: str = "mlp",
        gate_threshold: float = 0.0,
        gate_temperature: float = 1.0,
        use_instance_norm: bool = True,
        eps_E: float = 1e-6,
        trigger_mask: bool = True,
        delta_E_mask: float = 0.0,
        delta_P_mask: float = 0.0,
        trigger_mask_mode: str = "time",
        trigger_soft: bool = True,
        trigger_soft_tau: float = 0.25,
        ftsep5_feat_mode: str = "mdm",
        # Low-rank gate hyperparams
        gate_lowrank_rank: int = 4,
        gate_lowrank_time_ks: int = 5,
        gate_lowrank_freq_ks: int = 3,
        gate_lowrank_use_bias: bool = True,
        gate_sparse_l1_weight: float = 0.0,
        gate_lowrank_u_tv_weight: float = 0.0,
        # Oracle gate ablation
        gate_mode: str = "learned",          # "learned" | "oracle_train" | "oracle_eval"
        oracle_q: float = 0.99,              # quantile threshold for oracle trigger
        oracle_lambda_p: float = 0.25,       # phase-change weight in proxy score
        oracle_dilate: int = 1,              # extra dilation passes after base mask (1=no extra)
        # Aux loss hyperparams
        easy_ar_weight: float = 0.0,
        easy_ar_k: int = 8,
        easy_ar_ridge: float = 1e-3,
        white_acf_weight: float = 0.0,
        white_acf_lags: int = 8,
        shape_js_weight: float = 0.0,
        shape_w1_weight: float = 0.0,
        shape_weighting: str = "trigger",
        min_remove_weight: float = 0.0,
        min_remove_mode: str = "ntf_l2",
        energy_tv_weight: float = 0.0,
        n_ratio_min: float = 0.0,
        n_ratio_max: float = 0.0,
        n_ratio_weight: float = 0.0,
        n_ratio_power: int = 2,
        **kwargs,
    ):
        super().__init__()

        if future_mode not in ("repeat_last", "zero", "pred"):
            raise ValueError(f"future_mode must be 'repeat_last', 'zero', or 'pred', got {future_mode!r}")
        if future_mode == "pred" and n_pred_weight <= 0.0:
            raise ValueError(
                "future_mode='pred' requires n_pred_weight > 0. "
                "Example: --future-mode pred --n-pred-weight 1.0"
            )
        if n_pred_arch not in ("linear", "mlp"):
            raise ValueError(f"n_pred_arch must be 'linear' or 'mlp', got {n_pred_arch!r}")
        if teacher_mask_only and not trigger_mask:
            raise ValueError("teacher_mask_only=True requires trigger_mask=True")
        if trigger_mask_mode not in ("time", "tf"):
            raise ValueError(f"trigger_mask_mode must be 'time' or 'tf', got {trigger_mask_mode!r}")
        if trigger_soft_tau <= 0:
            raise ValueError(f"trigger_soft_tau must be > 0, got {trigger_soft_tau}")
        if ftsep5_feat_mode not in ("mdm", "mdm_pdp"):
            raise ValueError(f"ftsep5_feat_mode must be 'mdm' or 'mdm_pdp', got {ftsep5_feat_mode!r}")
        if shape_weighting not in ("none", "trigger"):
            raise ValueError(f"shape_weighting must be 'none' or 'trigger', got {shape_weighting!r}")
        if min_remove_mode not in ("ntf_l2", "ntf_l1", "g_l1"):
            raise ValueError(f"min_remove_mode must be 'ntf_l2', 'ntf_l1', or 'g_l1', got {min_remove_mode!r}")
        if n_ratio_power not in (1, 2):
            raise ValueError(f"n_ratio_power must be 1 or 2, got {n_ratio_power}")
        if n_ratio_weight > 0.0 and (n_ratio_max <= 0.0 or n_ratio_max <= n_ratio_min):
            raise ValueError(
                f"n_ratio_weight > 0 requires n_ratio_max > 0 and n_ratio_max > n_ratio_min. "
                f"Got n_ratio_min={n_ratio_min}, n_ratio_max={n_ratio_max}. "
                f"Example: --n-ratio-min 0.05 --n-ratio-max 0.30 --n-ratio-weight 0.1 --n-ratio-power 2"
            )
        if gate_arch in ("lowrank", "lowrank_sparse") and gate_lowrank_rank < 1:
            raise ValueError(f"gate_lowrank_rank must be >= 1, got {gate_lowrank_rank}")
        if gate_sparse_l1_weight > 0.0 and gate_arch != "lowrank_sparse":
            raise ValueError(
                f"gate_sparse_l1_weight > 0 requires gate_arch='lowrank_sparse', "
                f"got gate_arch={gate_arch!r}"
            )
        if gate_arch == "lowrank_sparse" and gate_sparse_l1_weight <= 0.0:
            raise ValueError(
                "gate_arch='lowrank_sparse' requires gate_sparse_l1_weight > 0 "
                "(e.g., 1e-3) to constrain the sparse residual. "
                "Without it the sparse conv becomes an unconstrained free 2D gate."
            )
        if gate_arch in ("lowrank", "lowrank_sparse") and (
            gate_lowrank_time_ks % 2 == 0 or gate_lowrank_freq_ks % 2 == 0
        ):
            raise ValueError(
                f"gate_lowrank_time_ks and gate_lowrank_freq_ks must be odd for symmetric padding. "
                f"Got time_ks={gate_lowrank_time_ks}, freq_ks={gate_lowrank_freq_ks}."
            )
        if gate_lowrank_u_tv_weight > 0.0 and gate_arch not in ("lowrank", "lowrank_sparse"):
            raise ValueError(
                f"gate_lowrank_u_tv_weight > 0 requires gate_arch in ('lowrank', 'lowrank_sparse'), "
                f"got gate_arch={gate_arch!r}"
            )
        if gate_mode not in ("learned", "oracle_train", "oracle_eval"):
            raise ValueError(
                f"gate_mode must be 'learned', 'oracle_train', or 'oracle_eval', got {gate_mode!r}"
            )

        self.trigger_mask_mode = trigger_mask_mode
        self.seq_len = seq_len
        self.use_instance_norm = use_instance_norm
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.gate_log_mag = gate_log_mag
        self.gate_arch = gate_arch
        self.gate_threshold_mode = gate_threshold_mode
        self.gate_use_log_mag = gate_use_log_mag
        self.future_mode = future_mode
        self.teacher_mask_only = bool(teacher_mask_only)
        self.n_pred_weight = float(n_pred_weight)
        self.n_pred_arch = n_pred_arch
        self.eps_E = float(eps_E)
        self.trigger_mask = bool(trigger_mask)
        self.delta_E_mask = float(delta_E_mask)
        self.delta_P_mask = float(delta_P_mask)
        self.trigger_soft = bool(trigger_soft)
        self.trigger_soft_tau = float(trigger_soft_tau)
        self.ftsep5_feat_mode = ftsep5_feat_mode
        self.gate_lowrank_rank = int(gate_lowrank_rank)
        self.gate_lowrank_time_ks = int(gate_lowrank_time_ks)
        self.gate_lowrank_freq_ks = int(gate_lowrank_freq_ks)
        self.gate_lowrank_use_bias = bool(gate_lowrank_use_bias)
        self.gate_sparse_l1_weight = float(gate_sparse_l1_weight)
        self.gate_lowrank_u_tv_weight = float(gate_lowrank_u_tv_weight)
        # Oracle gate
        self.gate_mode = gate_mode
        self.oracle_q = float(oracle_q)
        self.oracle_lambda_p = float(oracle_lambda_p)
        self.oracle_dilate = int(oracle_dilate)

        # Aux loss hyperparams
        self.easy_ar_weight = float(easy_ar_weight)
        self.easy_ar_k = int(easy_ar_k)
        self.easy_ar_ridge = float(easy_ar_ridge)
        self.white_acf_weight = float(white_acf_weight)
        self.white_acf_lags = int(white_acf_lags)
        self.shape_js_weight = float(shape_js_weight)
        self.shape_w1_weight = float(shape_w1_weight)
        self.shape_weighting = shape_weighting
        self.min_remove_weight = float(min_remove_weight)
        self.min_remove_mode = min_remove_mode
        self.energy_tv_weight = float(energy_tv_weight)
        self.n_ratio_min = float(n_ratio_min)
        self.n_ratio_max = float(n_ratio_max)
        self.n_ratio_weight = float(n_ratio_weight)
        self.n_ratio_power = int(n_ratio_power)

        if n_fft is None:
            n_fft = min(64, max(16, seq_len // 4))
        n_fft = min(n_fft, seq_len)
        self.stft = STFT(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann"
        )
        self.gate = LocalTFGate(
            enc_in,
            gate_type=gate_type,
            use_threshold=True,
            init_threshold=gate_threshold,
            temperature=gate_temperature,
            gate_mode="sigmoid",
            gate_arch=self.gate_arch,
            gate_threshold_mode=self.gate_threshold_mode,
            ftsep5_feat_mode=ftsep5_feat_mode,
            lowrank_rank=gate_lowrank_rank,
            lowrank_time_ks=gate_lowrank_time_ks,
            lowrank_freq_ks=gate_lowrank_freq_ks,
            lowrank_use_bias=gate_lowrank_use_bias,
        )

        # N_predictor: MLP or linear mapping n_hist (window) -> n_future (pred_len).
        # Created only when future_mode=="pred"; input shape (B, C, window).
        if future_mode == "pred":
            hidden = max(128, seq_len)
            if n_pred_arch == "mlp":
                self.n_predictor: Optional[nn.Module] = nn.Sequential(
                    nn.Linear(seq_len, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, pred_len),
                )
            else:  # "linear"
                self.n_predictor = nn.Linear(seq_len, pred_len)
        else:
            self.n_predictor = None

        # Gate scalar stats (for get_last_gate_stats)
        self._last_gate_mean = 0.0
        self._last_gate_sum_f = 0.0
        self._last_gate_max_f = 0.0
        self._last_gate_ent_f = 0.0
        # Oracle gate diagnostics
        self._last_oracle_rate: float = 0.0
        self._last_oracle_trig_rate: float = 0.0

        self._last_state: Optional[LocalTFNormState] = None

        # Caches that keep grad (used in loss())
        self._last_r_tf: Optional[torch.Tensor] = None    # (B, C, F, T) complex
        self._last_n_tf: Optional[torch.Tensor] = None    # (B, C, F, T) complex
        self._last_r_time: Optional[torch.Tensor] = None  # (B, T, C) real
        self._last_gate_eff: Optional[torch.Tensor] = None  # (B, C, F, T) real

        # Diagnostic caches (detached, no grad)
        self._last_x_tf: Optional[torch.Tensor] = None    # (B, C, F, T) complex
        self._last_n_time: Optional[torch.Tensor] = None  # (B, T, C) real
        self._last_w_trans: Optional[torch.Tensor] = None # (B,C,T-1) or (B,C,F,T-1)
        self._last_w_frame: Optional[torch.Tensor] = None # (B,C,T) or (B,C,F,T)
        self._last_x_time: Optional[torch.Tensor] = None  # (B, T, C) after RevIN
        self._last_g_raw: Optional[torch.Tensor] = None   # (B, C, F, T)
        # N-predictor caches
        self._last_n_hist_time: Optional[torch.Tensor] = None   # (B, C, window) detached
        self._last_n_pred_future: Optional[torch.Tensor] = None # (B, C, pred_len) w/ grad

        # Teacher debug caches (set by teacher_n_future())
        self._last_teacher_T_hist: int = 0
        self._last_teacher_T_full: int = 0
        self._last_teacher_mask_hist_rate: float = 0.0
        self._last_teacher_mask_full_rate: float = 0.0

        # Aux loss scalar caches (detached, for logging)
        self._last_aux_total: float = 0.0
        self._last_L_easy: float = 0.0
        self._last_L_white: float = 0.0
        self._last_L_js: float = 0.0
        self._last_L_w1: float = 0.0
        self._last_L_min: float = 0.0
        self._last_L_e_tv: float = 0.0
        self._last_ratio_n_bc_mean: float = 0.0
        self._last_ratio_n_bc_min: float = 0.0
        self._last_ratio_n_bc_max: float = 0.0
        self._last_loss_n_ratio_budget: float = 0.0
        self._last_pred_n_loss: float = 0.0
        self._last_L_sparse: float = 0.0
        self._last_L_u_tv: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dE_dP_from_tf(
        self, tf_complex: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-transition log-energy and mean spectral-shape differences.

        Args:
            tf_complex: (B, C, F, T) complex tensor

        Returns:
            dE: (B, C, T-1) log-energy differences
            dP: (B, C, T-1) mean spectral-shape differences
        """
        eps = 1e-8
        P = tf_complex.abs() ** 2                               # (B, C, F, T)
        E = P.mean(dim=2)                                       # (B, C, T)
        p = P / (P.sum(dim=2, keepdim=True) + eps)              # (B, C, F, T)
        logE = torch.log(E + self.eps_E)
        dE = torch.abs(logE[..., 1:] - logE[..., :-1])          # (B, C, T-1)
        dP = torch.abs(p[..., :, 1:] - p[..., :, :-1]).mean(dim=2)  # (B, C, T-1)
        return dE, dP

    def _dP_mean_and_perF(
        self, tf_complex: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-frequency and mean spectral-shape time-differences.

        Args:
            tf_complex: (B, C, F, T) complex tensor

        Returns:
            dP_f:    (B, C, F, T-1) per-frequency absolute shape difference
            dP_mean: (B, C, T-1)   mean of dP_f over frequency dim
        """
        eps = 1e-8
        P = tf_complex.abs() ** 2                               # (B, C, F, T)
        p = P / (P.sum(dim=2, keepdim=True) + eps)              # (B, C, F, T)
        dP_f = torch.abs(p[..., :, 1:] - p[..., :, :-1])       # (B, C, F, T-1)
        dP_mean = dP_f.mean(dim=2)                              # (B, C, T-1)
        return dP_f, dP_mean

    def _compute_gate_magnitude(self, x_tf: torch.Tensor) -> torch.Tensor:
        """Return the magnitude representation fed into the gate."""
        magnitude = x_tf.abs()
        if self.gate_use_log_mag is None:
            use_log_mag = bool(self.gate_log_mag)
        else:
            use_log_mag = bool(self.gate_use_log_mag)
        if use_log_mag:
            magnitude = torch.log1p(magnitude)
        return magnitude

    def _make_gate(self, x_tf: torch.Tensor) -> torch.Tensor:
        magnitude = self._compute_gate_magnitude(x_tf)
        return self.gate(magnitude)

    @torch.no_grad()
    def _oracle_gate_from_tf(self, x_tf_complex: torch.Tensor) -> torch.Tensor:
        """Deterministic oracle gate derived from the TF spectrum.

        Proxy score = |Δlog(|X|²)| + oracle_lambda_p * |wrap(Δphase)|,
        thresholded per (B, C) at quantile oracle_q.  The resulting transition
        trigger is dilated to a frame mask covering both sides of each triggered
        edge, then optionally grown for oracle_dilate−1 additional passes.

        Args:
            x_tf_complex: (B, C, F, T) complex STFT tensor

        Returns:
            g_oracle: (B, C, F, T) float in {0.0, 1.0}  (no grad)
        """
        import math as _math
        B, C, F, T = x_tf_complex.shape

        if T <= 1:
            self._last_oracle_trig_rate = 0.0
            self._last_oracle_rate = 0.0
            return x_tf_complex.new_zeros(B, C, F, T)

        eps = self.eps_E
        mag   = x_tf_complex.abs()                           # (B, C, F, T)
        logE  = torch.log(mag * mag + eps)                   # (B, C, F, T)
        phase = torch.angle(x_tf_complex)                    # (B, C, F, T)

        # Temporal differences → (B, C, F, T-1)
        dlogE  = torch.diff(logE,  dim=-1).abs()
        dP_raw = torch.diff(phase, dim=-1)
        dP     = ((dP_raw + _math.pi) % (2 * _math.pi) - _math.pi).abs()

        score = dlogE + self.oracle_lambda_p * dP            # (B, C, F, T-1)

        # Per-(B, C) quantile threshold (avoids cross-sample scale issues)
        flat = score.reshape(B, C, F * (T - 1))
        thr  = torch.quantile(flat, self.oracle_q, dim=-1, keepdim=True)  # (B, C, 1)
        thr  = thr.unsqueeze(-1)                                           # (B, C, 1, 1)
        trig = score >= thr                                                # (B, C, F, T-1)

        self._last_oracle_trig_rate = float(trig.float().mean().item())

        # Spread each triggered edge to both neighbouring frames → (B, C, F, T)
        mask_t = torch.zeros(B, C, F, T, device=x_tf_complex.device, dtype=torch.bool)
        mask_t[..., :-1] |= trig
        mask_t[...,  1:] |= trig

        # Extra dilation passes (oracle_dilate=1 means no extra pass beyond base)
        for _ in range(self.oracle_dilate - 1):
            expanded = mask_t.clone()
            expanded[..., :-1] |= mask_t[..., 1:]
            expanded[...,  1:] |= mask_t[..., :-1]
            mask_t = expanded

        g_oracle = mask_t.float()
        self._last_oracle_rate = float(g_oracle.mean().item())
        return g_oracle

    def _compute_gate_stats(self, g_local: torch.Tensor) -> None:
        """Compute gate statistics from gate activations g_local (B, C, F, T)."""
        self._last_gate_mean = float(g_local.detach().mean().cpu().item())

        g_sum_f = g_local.sum(dim=2)   # (B, C, T)
        g_max_f = g_local.max(dim=2)[0]  # (B, C, T)
        self._last_gate_sum_f = float(g_sum_f.detach().mean().cpu().item())
        self._last_gate_max_f = float(g_max_f.detach().mean().cpu().item())

        g_norm = g_local / (g_local.sum(dim=2, keepdim=True) + 1e-10)
        eps = 1e-10
        entropy = -(g_norm * torch.log(g_norm + eps)).sum(dim=2)  # (B, C, T)
        f_size = g_local.shape[2]
        max_entropy = math.log(f_size) if f_size > 1 else 1.0
        entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy
        self._last_gate_ent_f = float(entropy_norm.detach().mean().cpu().item())

    # ------------------------------------------------------------------
    # Aux loss private methods (differentiable)
    # ------------------------------------------------------------------

    def _loss_ridge_ar(
        self, r_time: torch.Tensor, k: int, ridge: float
    ) -> torch.Tensor:
        """Per-channel ridge AR(k) one-step-ahead MSE.

        For each channel c, flattens all B batches into N = B*(T-k) samples,
        builds lag matrix X (N, k) and target y (N,), solves the normal
        equations (XᵀX + ridge·I)a = Xᵀy via torch.linalg.solve, and returns
        the mean squared residual across all channels.

        Args:
            r_time: (B, T, C) residual time series
            k:      number of AR lags
            ridge:  L2 regularisation coefficient

        Returns:
            scalar MSE tensor (differentiable w.r.t. r_time)
        """
        B, T, C = r_time.shape
        if T <= k:
            return r_time.new_tensor(0.0)

        # Build lag matrix: lags_list[j] holds lag-(j+1) values
        # lags_list[0] = r_time[:, k-1:T-1, :]  (lag 1)
        # lags_list[k-1] = r_time[:, 0:T-k, :]  (lag k)
        lags_list = [r_time[:, k - 1 - j : T - 1 - j, :] for j in range(k)]
        X_bcnk = torch.stack(lags_list, dim=2)  # (B, T-k, k, C)
        y_bcn = r_time[:, k:, :]               # (B, T-k, C)

        N = B * (T - k)
        # Reshape to (C, N, k) and (C, N, 1)
        X = X_bcnk.reshape(N, k, C).permute(2, 0, 1)       # (C, N, k)
        y = y_bcn.reshape(N, C).T.unsqueeze(-1)             # (C, N, 1)

        # Normal equations
        XtX = torch.bmm(X.transpose(1, 2), X)              # (C, k, k)
        I = torch.eye(k, device=r_time.device, dtype=r_time.dtype).unsqueeze(0)
        A = XtX + ridge * I                                 # (C, k, k)
        Xty = torch.bmm(X.transpose(1, 2), y)              # (C, k, 1)
        a = torch.linalg.solve(A, Xty)                     # (C, k, 1)

        y_pred = torch.bmm(X, a)                           # (C, N, 1)
        return ((y - y_pred) ** 2).mean()

    def _loss_white_acf(
        self, r_time: torch.Tensor, lags: int
    ) -> torch.Tensor:
        """Mean squared normalized autocorrelation for lags 1..lags.

        Args:
            r_time: (B, T, C)
            lags:   number of lags to test

        Returns:
            scalar mean(corr_k^2) averaged over lags and (B, C)
        """
        B, T, C = r_time.shape
        r = r_time - r_time.mean(dim=1, keepdim=True)  # (B, T, C) demeaned
        var = (r ** 2).mean(dim=1, keepdim=True) + 1e-8  # (B, 1, C)

        n_lags = min(lags, T - 1)
        if n_lags <= 0:
            return r_time.new_tensor(0.0)

        acf_sq_sum = r_time.new_tensor(0.0)
        for lag in range(1, n_lags + 1):
            corr_k = (r[:, lag:, :] * r[:, :T - lag, :]).mean(dim=1) / var.squeeze(1)
            acf_sq_sum = acf_sq_sum + (corr_k ** 2).mean()

        return acf_sq_sum / n_lags

    def _loss_shape_js(
        self,
        r_tf: torch.Tensor,
        w_trans_or_none: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """JS divergence between adjacent STFT frames.

        Computes P = |r_tf|^2, normalises over F to get spectral shape p,
        then computes JS(p_t || p_{t-1}) = 0.5*(KL(p_t||m) + KL(p_{t-1}||m))
        where m = 0.5*(p_t + p_{t-1}).

        Args:
            r_tf:            (B, C, F, T) complex residual TF tensor
            w_trans_or_none: (B, C, T-1) broadcastable weights, or None

        Returns:
            scalar JS loss
        """
        eps = 1e-8
        P = r_tf.abs() ** 2                                        # (B, C, F, T)
        p = P / (P.sum(dim=2, keepdim=True) + eps)                 # (B, C, F, T)
        p_curr = p[..., 1:]                                        # (B, C, F, T-1)
        p_prev = p[..., :-1]                                       # (B, C, F, T-1)
        m = 0.5 * (p_curr + p_prev)                                # (B, C, F, T-1)
        kl_curr = (p_curr * torch.log((p_curr + eps) / (m + eps))).sum(dim=2)  # (B, C, T-1)
        kl_prev = (p_prev * torch.log((p_prev + eps) / (m + eps))).sum(dim=2)  # (B, C, T-1)
        js = 0.5 * (kl_curr + kl_prev)                            # (B, C, T-1)

        if w_trans_or_none is None:
            return js.mean()
        w = w_trans_or_none  # (B, C, T-1) broadcastable
        return (js * w).sum() / (w.sum() + eps)

    def _loss_shape_w1(
        self,
        r_tf: torch.Tensor,
        w_trans_or_none: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Wasserstein-1 distance between adjacent STFT frames.

        Computes p (spectral shape), CDF along F, then W1 = mean_F(|cdf_t - cdf_{t-1}|).

        Args:
            r_tf:            (B, C, F, T) complex residual TF tensor
            w_trans_or_none: (B, C, T-1) broadcastable weights, or None

        Returns:
            scalar W1 loss
        """
        eps = 1e-8
        P = r_tf.abs() ** 2                                   # (B, C, F, T)
        p = P / (P.sum(dim=2, keepdim=True) + eps)            # (B, C, F, T)
        cdf = torch.cumsum(p, dim=2)                          # (B, C, F, T)
        w1 = (cdf[..., 1:] - cdf[..., :-1]).abs().mean(dim=2)  # (B, C, T-1)

        if w_trans_or_none is None:
            return w1.mean()
        w = w_trans_or_none  # (B, C, T-1) broadcastable
        return (w1 * w).sum() / (w.sum() + eps)

    # ------------------------------------------------------------------
    # Gate stats
    # ------------------------------------------------------------------

    def get_last_gate_stats(self) -> dict[str, float]:
        """Return last computed gate statistics for monitoring."""
        return {
            "gate_mean": self._last_gate_mean,
            "gate_sum_f": self._last_gate_sum_f,
            "gate_max_f": self._last_gate_max_f,
            "gate_ent_f": self._last_gate_ent_f,
        }

    # ------------------------------------------------------------------
    # normalize / denormalize
    # ------------------------------------------------------------------

    def normalize(
        self, batch_x: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, LocalTFNormState]:
        # RevIN-style instance normalization
        inst_mean = inst_std = None
        if self.use_instance_norm:
            inst_mean = batch_x.mean(dim=1, keepdim=True)          # (B, 1, C)
            inst_std = batch_x.std(dim=1, keepdim=True) + 1e-8     # (B, 1, C)
            batch_x = (batch_x - inst_mean) / inst_std

        x_tf = self.stft(batch_x)                                   # (B, C, F, T)
        gate_magnitude = self._compute_gate_magnitude(x_tf)
        g_raw = self.gate(gate_magnitude)                           # (B, C, F, T)

        # Oracle gate ablation: replace g_raw with deterministic gate (no grad)
        _use_oracle = (
            self.gate_mode == "oracle_train"
            or (self.gate_mode == "oracle_eval" and not self.training)
        )
        if _use_oracle:
            g_raw = self._oracle_gate_from_tf(x_tf)
        else:
            self._last_oracle_rate = 0.0
            self._last_oracle_trig_rate = 0.0

        # Trigger mask: restrict gate to non-stationary boundary frames
        if self.trigger_mask:
            de = self.delta_E_mask
            dp = self.delta_P_mask
            if de <= 0.0 or dp <= 0.0:
                raise RuntimeError(
                    "trigger_mask=True but delta_E_mask or delta_P_mask is <= 0; "
                    "set both delta_E_mask > 0 and delta_P_mask > 0"
                )
            B, C, F, T = x_tf.shape
            with torch.no_grad():
                dE_x, dP_mean_x = self._dE_dP_from_tf(x_tf)       # (B, C, T-1)
                if self.trigger_mask_mode == "tf":
                    dP_f_x, _ = self._dP_mean_and_perF(x_tf)       # (B, C, F, T-1)

            if self.trigger_mask_mode == "time":
                sE_t = dE_x / (de + 1e-8)                          # (B, C, T-1)
                sP_t = dP_mean_x / (dp + 1e-8)
                s_t = torch.maximum(sE_t, sP_t)
                # Force hard binary mask when teacher_mask_only (align decomp with causal teacher)
                if self.teacher_mask_only or not self.trigger_soft:
                    w_trans = (s_t > 1.0).float()
                else:
                    w_trans = torch.sigmoid((s_t - 1.0) / self.trigger_soft_tau)
                # Spread transition weights to frame weights
                w_frame = torch.zeros(B, C, T, device=x_tf.device, dtype=w_trans.dtype)
                w_frame[..., :-1] = torch.maximum(w_frame[..., :-1], w_trans)
                w_frame[..., 1:] = torch.maximum(w_frame[..., 1:], w_trans)
                if self.teacher_mask_only:
                    g_eff = w_frame.unsqueeze(2)                    # (B, C, 1, T) broadcasts
                else:
                    g_eff = g_raw * w_frame.unsqueeze(2)            # (B, C, F, T)
            else:  # "tf"
                sE_tf = dE_x / (de + 1e-8)                         # (B, C, T-1)
                sP_tf = dP_f_x / (dp + 1e-8)                       # (B, C, F, T-1)
                s_tf = torch.maximum(sE_tf.unsqueeze(2), sP_tf)    # (B, C, F, T-1)
                # Force hard binary mask when teacher_mask_only (align decomp with causal teacher)
                if self.teacher_mask_only or not self.trigger_soft:
                    w_trans = (s_tf > 1.0).float()
                else:
                    w_trans = torch.sigmoid((s_tf - 1.0) / self.trigger_soft_tau)
                w_frame = torch.zeros(B, C, F, T, device=x_tf.device, dtype=w_trans.dtype)
                w_frame[..., :-1] = torch.maximum(w_frame[..., :-1], w_trans)
                w_frame[..., 1:] = torch.maximum(w_frame[..., 1:], w_trans)
                if self.teacher_mask_only:
                    g_eff = w_frame                                 # (B, C, F, T) pure mask
                else:
                    g_eff = g_raw * w_frame                         # (B, C, F, T)
        else:
            g_eff = g_raw
            w_trans = None
            w_frame = None

        # Decompose x_tf into non-stationary (n_tf) and residual (r_tf)
        n_tf = g_eff * x_tf                                         # (B, C, F, T) complex
        r_tf = x_tf - n_tf                                         # (B, C, F, T) complex

        length = batch_x.shape[1]
        residual = self.stft.inverse(r_tf, length=length)          # (B, T, C)
        n_time = self.stft.inverse(n_tf, length=length)            # (B, T, C)

        # Cache N history for N_predictor (detached: no coupling back to gate)
        if self.future_mode == "pred":
            self._last_n_hist_time = n_time.detach().permute(0, 2, 1)  # (B, C, window)

        # Gate scalar diagnostics (always from g_raw so stats reflect learnable gate)
        self._compute_gate_stats(g_raw)

        state = LocalTFNormState(
            x_tf=x_tf,
            n_tf=n_tf,
            g_local=g_raw,
            n_time=n_time,
            length=length,
            mean=inst_mean,
            std=inst_std,
        )
        self._last_state = state

        # Caches for loss() — keep grad
        self._last_r_tf = r_tf
        self._last_n_tf = n_tf
        self._last_r_time = residual
        self._last_gate_eff = g_eff

        # Diagnostic caches — detach
        self._last_x_tf = x_tf.detach()
        self._last_n_time = n_time.detach()
        self._last_x_time = batch_x.detach()
        self._last_g_raw = g_raw.detach()
        self._last_w_trans = w_trans.detach() if w_trans is not None else None
        self._last_w_frame = w_frame.detach() if w_frame is not None else None

        if return_state:
            return residual, state
        return residual

    def denormalize(
        self, batch_x: torch.Tensor, state: Optional[LocalTFNormState] = None
    ) -> torch.Tensor:
        if state is None:
            state = self._last_state
        if state is None:
            raise RuntimeError("LocalTFNorm denormalize requires a stored state.")
        target_len = batch_x.shape[1]
        if self.future_mode == "pred" and self.n_predictor is not None:
            # Use learned N_predictor to add back the predicted future N component
            n_hist = self._last_n_hist_time              # (B, C, window) detached
            n_pred_future = self.n_predictor(n_hist)     # (B, C, pred_len)
            self._last_n_pred_future = n_pred_future     # w/ grad for pred_n_loss
            n_time = n_pred_future.detach().transpose(1, 2)  # detached: task_loss must not train predictor
        elif target_len == state.length:
            n_time = state.n_time
        else:
            n_time = self._extrapolate_n_time(state.n_time, target_len)
        result = batch_x + n_time
        # Reverse RevIN
        if state.mean is not None and state.std is not None:
            result = result * state.std + state.mean
        return result

    # ------------------------------------------------------------------
    # N_predictor API
    # ------------------------------------------------------------------

    def get_last_n_pred_future(self) -> Optional[torch.Tensor]:
        """Return the predicted N future from the last denormalize() call.

        Returns (B, C, pred_len) with grad, or None if not in pred mode.
        """
        return self._last_n_pred_future

    @torch.no_grad()
    def teacher_n_future(
        self, x_hist_raw: torch.Tensor, y_true_raw: torch.Tensor
    ) -> torch.Tensor:
        """Compute causal ground-truth N future via teacher (trigger-mask) decomposition.

        The future mask is derived ONLY from history frames (causal). The history mask is
        expanded to cover future STFT frames by repeating the last history frame.
        Always uses a hard binary trigger (no soft sigmoid), aligned with the
        teacher_mask_only hard-mask decomposition in normalize().

        Args:
            x_hist_raw: (B, window, C) — raw history in the same space as model input
            y_true_raw: (B, pred_len, C) — ground-truth future in the same space

        Returns:
            (B, C, pred_len) N future in per-sample instance-normed space
        """
        if self.delta_E_mask <= 0.0 or self.delta_P_mask <= 0.0:
            raise RuntimeError(
                "teacher_n_future() requires delta_E_mask > 0 and delta_P_mask > 0"
            )
        de = self.delta_E_mask
        dp = self.delta_P_mask

        # Instance norm using HISTORY stats only (mirrors normalize()); clamp std
        if self.use_instance_norm:
            inst_mean = x_hist_raw.mean(dim=1, keepdim=True)                   # (B, 1, C)
            inst_std  = x_hist_raw.std(dim=1, keepdim=True).clamp_min(1e-6)    # (B, 1, C)
            x_hist = (x_hist_raw - inst_mean) / inst_std
            y_true = (y_true_raw - inst_mean) / inst_std
        else:
            x_hist = x_hist_raw
            y_true = y_true_raw

        x_full = torch.cat([x_hist, y_true], dim=1)                            # (B, window+pred_len, C)
        full_len = x_full.shape[1]

        # STFT history (for causal mask) and full sequence (for N extraction)
        x_hist_tf = self.stft(x_hist)                                          # (B, C, F, T_hist)
        x_full_tf = self.stft(x_full)                                          # (B, C, F, T_full)
        B, C, F, T_hist = x_hist_tf.shape
        T_full = x_full_tf.shape[-1]

        # Causal hard-binary mask from history frames ONLY
        dE_hist, dP_mean_hist = self._dE_dP_from_tf(x_hist_tf)                # (B, C, T_hist-1)

        if self.trigger_mask_mode == "time":
            # Hard trigger: (dE > de) OR (dP_mean > dp)  [equiv. to max(dE/de, dP/dp) > 1]
            trig_hist = (dE_hist > de) | (dP_mean_hist > dp)                  # (B, C, T_hist-1)
            mask_hist_t = torch.zeros(B, C, T_hist, device=x_hist_tf.device, dtype=torch.float32)
            mask_hist_t[..., :-1] = torch.maximum(mask_hist_t[..., :-1], trig_hist.float())
            mask_hist_t[..., 1:]  = torch.maximum(mask_hist_t[..., 1:],  trig_hist.float())
            # Expand to full length: repeat last history STFT frame for future frames
            last         = mask_hist_t[..., -1:]                              # (B, C, 1)
            tail         = last.expand(-1, -1, T_full - T_hist)              # (B, C, T_full-T_hist)
            mask_full_t  = torch.cat([mask_hist_t, tail], dim=-1)            # (B, C, T_full)
            mask_full_tf = mask_full_t.unsqueeze(2)                          # (B, C, 1, T_full) -> bcast F
            # Cache debug stats
            self._last_teacher_T_hist = T_hist
            self._last_teacher_T_full = T_full
            self._last_teacher_mask_hist_rate = float(mask_hist_t.mean().item())
            self._last_teacher_mask_full_rate = float(mask_full_t.mean().item())
        else:  # "tf"
            dP_f_hist, _ = self._dP_mean_and_perF(x_hist_tf)                 # (B, C, F, T_hist-1)
            # Hard trigger per (freq, time) cell; energy trigger broadcast across freq
            trig_E       = (dE_hist > de).unsqueeze(2)                       # (B, C, 1, T_hist-1)
            trig_P       = (dP_f_hist > dp)                                  # (B, C, F, T_hist-1)
            trig_hist_tf = (trig_E | trig_P).float()                         # (B, C, F, T_hist-1)
            mask_hist_tf = torch.zeros(B, C, F, T_hist, device=x_hist_tf.device, dtype=torch.float32)
            mask_hist_tf[..., :-1] = torch.maximum(mask_hist_tf[..., :-1], trig_hist_tf)
            mask_hist_tf[..., 1:]  = torch.maximum(mask_hist_tf[..., 1:],  trig_hist_tf)
            # Expand to full length: repeat last history STFT frame for future frames
            last         = mask_hist_tf[..., -1:]                            # (B, C, F, 1)
            tail         = last.expand(-1, -1, -1, T_full - T_hist)         # (B, C, F, T_full-T_hist)
            mask_full_tf = torch.cat([mask_hist_tf, tail], dim=-1)          # (B, C, F, T_full)
            # Cache debug stats
            self._last_teacher_T_hist = T_hist
            self._last_teacher_T_full = T_full
            self._last_teacher_mask_hist_rate = float(mask_hist_tf.mean().item())
            self._last_teacher_mask_full_rate = float(mask_full_tf.mean().item())

        n_full_tf   = mask_full_tf * x_full_tf                                # (B, C, F, T_full)
        n_full_time = self.stft.inverse(n_full_tf, length=full_len)           # (B, window+pred_len, C)
        n_future_true = n_full_time[:, -self.pred_len:, :]                    # (B, pred_len, C)
        return n_future_true.permute(0, 2, 1)                                 # (B, C, pred_len)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aux loss combining AR whitening, ACF, spectral-shape, and energy terms.

        All terms are skipped (zero) when their weight is 0.0.
        Caches float scalars for logging via get_last_aux_stats().
        """
        device = self._device()
        zero = torch.tensor(0.0, device=device)

        # Reset pred_n_loss each call; training loop sets it after loss() if applicable
        self._last_pred_n_loss = 0.0

        # Reset lowrank loss caches
        self._last_L_sparse = 0.0
        self._last_L_u_tv = 0.0

        if self._last_r_tf is None:
            self._last_aux_total = self._last_L_easy = self._last_L_white = 0.0
            self._last_L_js = self._last_L_w1 = self._last_L_min = self._last_L_e_tv = 0.0
            return zero

        # Prepare shape-loss transition weights
        if self.shape_weighting == "trigger" and self._last_w_trans is not None:
            w_shape = self._last_w_trans
            if w_shape.dim() == 4:  # tf mode: (B, C, F, T-1) -> (B, C, T-1)
                w_shape = w_shape.mean(dim=2)
        else:
            w_shape = None

        aux_total = zero

        # L_easy: ridge AR whitening on residual
        if self.easy_ar_weight > 0.0:
            L_easy = self._loss_ridge_ar(
                self._last_r_time, self.easy_ar_k, self.easy_ar_ridge
            )
            aux_total = aux_total + self.easy_ar_weight * L_easy
            self._last_L_easy = float(L_easy.detach())
        else:
            self._last_L_easy = 0.0

        # L_white: normalized ACF test statistic
        if self.white_acf_weight > 0.0:
            L_white = self._loss_white_acf(self._last_r_time, self.white_acf_lags)
            aux_total = aux_total + self.white_acf_weight * L_white
            self._last_L_white = float(L_white.detach())
        else:
            self._last_L_white = 0.0

        # L_js: JS divergence between adjacent spectral frames
        if self.shape_js_weight > 0.0:
            L_js = self._loss_shape_js(self._last_r_tf, w_shape)
            aux_total = aux_total + self.shape_js_weight * L_js
            self._last_L_js = float(L_js.detach())
        else:
            self._last_L_js = 0.0

        # L_w1: W1 distance between adjacent spectral frames
        if self.shape_w1_weight > 0.0:
            L_w1 = self._loss_shape_w1(self._last_r_tf, w_shape)
            aux_total = aux_total + self.shape_w1_weight * L_w1
            self._last_L_w1 = float(L_w1.detach())
        else:
            self._last_L_w1 = 0.0

        # L_min: minimise non-stationary component energy
        if self.min_remove_weight > 0.0:
            if self.min_remove_mode == "ntf_l2":
                L_min = self._last_n_tf.abs().pow(2).mean()
            elif self.min_remove_mode == "ntf_l1":
                L_min = self._last_n_tf.abs().mean()
            else:  # "g_l1"
                L_min = self._last_gate_eff.mean()
            aux_total = aux_total + self.min_remove_weight * L_min
            self._last_L_min = float(L_min.detach())
        else:
            self._last_L_min = 0.0

        # L_e_tv: TV of log-energy of residual along time (no margin)
        if self.energy_tv_weight > 0.0:
            P_r = self._last_r_tf.abs() ** 2                        # (B, C, F, T)
            E_r = P_r.mean(dim=2)                                   # (B, C, T)
            logE_r = torch.log(E_r + self.eps_E)
            L_e_tv = torch.abs(logE_r[..., 1:] - logE_r[..., :-1]).mean()
            aux_total = aux_total + self.energy_tv_weight * L_e_tv
            self._last_L_e_tv = float(L_e_tv.detach())
        else:
            self._last_L_e_tv = 0.0

        # L_n_ratio: hinge budget on per-(B,C) n/x energy ratio
        if (
            self.n_ratio_weight > 0.0
            and self.n_ratio_max > 0.0
            and self.n_ratio_max > self.n_ratio_min
            and self._last_n_tf is not None
            and self._last_x_tf is not None
        ):
            eps_ratio = 1e-8
            # P_n keeps grad via _last_n_tf = g_eff * x_tf (computed in normalize())
            P_n = self._last_n_tf.abs() ** 2                          # (B, C, F, T) w/ grad
            # P_x uses the detached x_tf — denominator is treated as constant
            P_x = self._last_x_tf.abs() ** 2                          # (B, C, F, T) detached
            E_n = P_n.mean(dim=(2, 3))                                 # (B, C)
            E_x = P_x.mean(dim=(2, 3))                                 # (B, C) detached
            ratio_bc = E_n / (E_x + eps_ratio)                        # (B, C) w/ grad

            low  = torch.relu(self.n_ratio_min - ratio_bc)
            high = torch.relu(ratio_bc - self.n_ratio_max)
            hinge = low + high
            if self.n_ratio_power == 2:
                loss_n_ratio_budget = (hinge ** 2).mean()
            else:
                loss_n_ratio_budget = hinge.mean()

            aux_total = aux_total + self.n_ratio_weight * loss_n_ratio_budget
            self._last_ratio_n_bc_mean = float(ratio_bc.detach().mean().item())
            self._last_ratio_n_bc_min = float(ratio_bc.detach().min().item())
            self._last_ratio_n_bc_max = float(ratio_bc.detach().max().item())
            self._last_loss_n_ratio_budget = float(loss_n_ratio_budget.detach().item())
        else:
            self._last_ratio_n_bc_mean = 0.0
            self._last_ratio_n_bc_min = 0.0
            self._last_ratio_n_bc_max = 0.0
            self._last_loss_n_ratio_budget = 0.0

        # L_sparse: L1 of sparse residual logits (lowrank_sparse only)
        if self.gate_sparse_l1_weight > 0.0:
            sparse_raw = getattr(self.gate, "_last_sparse_raw", None)
            if sparse_raw is not None:
                L_sparse = sparse_raw.abs().mean()
                aux_total = aux_total + self.gate_sparse_l1_weight * L_sparse
                self._last_L_sparse = float(L_sparse.detach().item())
            else:
                self._last_L_sparse = 0.0
        else:
            self._last_L_sparse = 0.0

        # L_u_tv: total-variation of u(t) along time (lowrank / lowrank_sparse)
        if self.gate_lowrank_u_tv_weight > 0.0:
            proj = getattr(self.gate, "proj", None)
            u_raw = getattr(proj, "_last_u_raw", None)
            if u_raw is not None and u_raw.shape[-1] > 1:
                L_u_tv = u_raw[..., 1:].sub(u_raw[..., :-1]).abs().mean()
                aux_total = aux_total + self.gate_lowrank_u_tv_weight * L_u_tv
                self._last_L_u_tv = float(L_u_tv.detach().item())
            else:
                self._last_L_u_tv = 0.0
        else:
            self._last_L_u_tv = 0.0

        self._last_aux_total = float(aux_total.detach())
        return aux_total

    def get_last_aux_stats(self) -> dict[str, float]:
        """Return cached aux loss components for logging."""
        return {
            "aux_total": self._last_aux_total,
            "L_easy": self._last_L_easy,
            "L_white": self._last_L_white,
            "L_js": self._last_L_js,
            "L_w1": self._last_L_w1,
            "L_min": self._last_L_min,
            "L_e_tv": self._last_L_e_tv,
            "ratio_n_bc_mean": self._last_ratio_n_bc_mean,
            "ratio_n_bc_min": self._last_ratio_n_bc_min,
            "ratio_n_bc_max": self._last_ratio_n_bc_max,
            "loss_n_ratio_budget": self._last_loss_n_ratio_budget,
            "pred_n_loss": self._last_pred_n_loss,
            "L_sparse": self._last_L_sparse,
            "L_u_tv": self._last_L_u_tv,
        }

    def get_last_decomp_stats(self) -> dict[str, float]:
        """Return trigger/soft and decomposition diagnostics."""
        w_mean = float("nan")
        w_max = float("nan")
        if self._last_w_frame is not None:
            w_mean = float(self._last_w_frame.mean().item())
            w_max = float(self._last_w_frame.max().item())

        g_raw_mean = float("nan")
        if self._last_g_raw is not None:
            g_raw_mean = float(self._last_g_raw.mean().item())

        g_eff_mean = float("nan")
        if self._last_gate_eff is not None:
            g_eff_mean = float(self._last_gate_eff.detach().mean().item())

        n_energy = float("nan")
        if self._last_n_tf is not None:
            n_energy = float(self._last_n_tf.detach().abs().pow(2).mean().item())

        ratio_n_time = float("nan")
        if self._last_n_time is not None and self._last_x_time is not None:
            n_e = float(self._last_n_time.pow(2).mean().item())
            x_e = float(self._last_x_time.pow(2).mean().item())
            ratio_n_time = n_e / (x_e + 1e-10)

        corr_x_r = float("nan")
        if self._last_x_time is not None and self._last_r_time is not None:
            x_flat = self._last_x_time.detach().float().flatten()
            r_flat = self._last_r_time.detach().float().flatten()
            xc = x_flat - x_flat.mean()
            rc = r_flat - r_flat.mean()
            cov = (xc * rc).mean()
            corr_x_r = float((cov / (xc.std() * rc.std() + 1e-10)).item())

        return {
            "w_mean": w_mean,
            "w_max": w_max,
            "g_raw_mean": g_raw_mean,
            "g_eff_mean": g_eff_mean,
            "n_energy": n_energy,
            "ratio_n_time": ratio_n_time,
            "corr_x_r": corr_x_r,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def extract_n_time_only(self, x_time: torch.Tensor) -> torch.Tensor:
        """Apply RevIN + STFT + gate + iSTFT and return n_time.

        Does NOT write any _last_* cache fields.
        """
        if self.use_instance_norm:
            inst_mean = x_time.mean(dim=1, keepdim=True)
            inst_std = x_time.std(dim=1, keepdim=True) + 1e-8
            x = (x_time - inst_mean) / inst_std
        else:
            x = x_time
        x_tf = self.stft(x)
        g_local = self._make_gate(x_tf)
        n_tf = g_local * x_tf
        return self.stft.inverse(n_tf, length=x.shape[1])

    def _extrapolate_n_time(self, n_time: torch.Tensor, target_len: int) -> torch.Tensor:
        """Deterministic extrapolation of n_time to target_len frames."""
        source_len = n_time.shape[1]
        if target_len <= source_len:
            return n_time[:, :target_len, :]
        if self.future_mode == "repeat_last":
            last = n_time[:, -1:, :]
            return last.expand(-1, target_len, -1)
        if self.future_mode == "zero":
            return torch.zeros(
                n_time.shape[0], target_len, n_time.shape[2],
                device=n_time.device, dtype=n_time.dtype,
            )
        raise ValueError(f"Unsupported future_mode: {self.future_mode}")

    def forward(
        self,
        batch_x: torch.Tensor,
        mode: str = "n",
        state: Optional[LocalTFNormState] = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LocalTFNormState]:
        if mode == "n":
            return self.normalize(batch_x, return_state=return_state)
        if mode == "d":
            return self.denormalize(batch_x, state=state)
        raise ValueError(f"Unsupported mode: {mode}")
