from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math

import numpy as np
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
    pred_n_time: Optional[torch.Tensor] = None
    pred_n_tf: Optional[torch.Tensor] = None
    mean: Optional[torch.Tensor] = None   # (B, 1, C) instance mean for RevIN
    std: Optional[torch.Tensor] = None    # (B, 1, C) instance std for RevIN


class SimpleTFPredictor(nn.Module):
    """
    Time-frequency predictor for non-stationary component.
    
    Input: real/imaginary feature tensor
    - pred_input="n_tf": (B, C, F, T, 2) where 2 is real/imag
    - pred_input="n_tf_x_tf": (B, C, F, T, 4) where 4 is [n_real, n_imag, x_real, x_imag]
    
    Processing: reshape to (B*C*F, T*feat_dim), pass through MLP, reshape back to (B, C, F, out_T, 2)
    """
    def __init__(
        self,
        in_frames: int,
        out_frames: int,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        input_feature_dim: int = 2,
    ):
        """
        Args:
            in_frames: number of input frames (time steps)
            out_frames: number of output frames
            hidden_dim: hidden dimension in MLP (0 = linear)
            dropout: dropout rate
            input_feature_dim: feature dimension (2 for n_tf real+imag, 4 for n_tf_x_tf context)
        """
        super().__init__()
        # Input: T * feat_dim; Output: out_T * 2 (always real+imag)
        in_features = in_frames * input_feature_dim
        out_features = out_frames * 2
        
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_features),
            )
        else:
            self.net = nn.Linear(in_features, out_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, F, T, feat_dim) where feat_dim in {2, 4}
        
        Returns:
            output: (B, C, F, out_T, 2) as real tensor, will be converted to complex outside
        """
        b, c, f, t, feat_dim = features.shape
        # sanity: feature dim should be 2 (real+imag) for n_tf, or 4 for n_tf_x_tf
        assert feat_dim in (2, 4), f"unexpected feature dim {feat_dim}, expected 2 or 4"

        # Reshape to (B*C*F, T*feat_dim) for MLP
        features_flat = features.reshape(b * c * f, t * feat_dim)
        
        # MLP: (B*C*F, T*feat_dim) -> (B*C*F, out_T*2)
        output_flat = self.net(features_flat)
        
        # Reshape to (B, C, F, out_T, 2)
        out_frames = output_flat.shape[-1] // 2
        output = output_flat.reshape(b, c, f, out_frames, 2)
        
        # Convert to complex
        return torch.view_as_complex(output.contiguous())


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
        gate_entropy_weight: float = 0.0,
        gate_use_log_mag: Optional[bool] = None,
        stationarity_loss_weight: float = 0.0,
        stationarity_chunks: int = 4,
        future_mode: str = "repeat_last",
        predict_n_time: bool = False,
        pred_hidden_dim: int = 64,
        pred_dropout: float = 0.1,
        pred_loss_weight: float = 0.0,
        gate_threshold: float = 0.0,
        gate_temperature: float = 1.0,
        gate_smooth_weight: float = 0.0,
        gate_ratio_weight: float = 0.0,
        gate_ratio_target: float = 0.3,
        gate_mode: str = "sigmoid",
        gate_budget_dim: str = "freq",
        pred_input: str = "n_tf",
        gate_temporal_smooth_weight: float = 0.0,
        use_instance_norm: bool = True,
        lambda_E: float = 1.0,
        lambda_P: float = 1.0,
        eps_E: float = 1e-6,
        delta_E: float = 0.0,
        delta_P: float = 0.0,
        trigger_mask: bool = True,
        delta_E_mask: float = 0.0,
        delta_P_mask: float = 0.0,
        shape_loss_mode: str = "meanF",
        trigger_mask_mode: str = "time",
        gate_sup_enable: bool = True,
        gate_sup_weight: float = 0.2,
        gate_sup_tau: float = 0.3,
        gate_sup_loss: str = "bce",
        gate_sup_pos_weight: float = 5.0,
        gate_sup_on: str = "raw",
        gate_sup_target: str = "soft",
        gate_sup_source: str = "P_only",
        gate_sup_wE: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if shape_loss_mode not in ("meanF", "perF"):
            raise ValueError(f"shape_loss_mode must be 'meanF' or 'perF', got {shape_loss_mode!r}")
        if trigger_mask_mode not in ("time", "tf"):
            raise ValueError(f"trigger_mask_mode must be 'time' or 'tf', got {trigger_mask_mode!r}")
        if gate_sup_loss not in ("bce", "focal"):
            raise ValueError(f"gate_sup_loss must be 'bce' or 'focal', got {gate_sup_loss!r}")
        if gate_sup_on not in ("raw", "eff"):
            raise ValueError(f"gate_sup_on must be 'raw' or 'eff', got {gate_sup_on!r}")
        if gate_sup_target not in ("soft", "hard"):
            raise ValueError(f"gate_sup_target must be 'soft' or 'hard', got {gate_sup_target!r}")
        if gate_sup_source not in ("P_only", "EP_max"):
            raise ValueError(f"gate_sup_source must be 'P_only' or 'EP_max', got {gate_sup_source!r}")
        if gate_sup_wE < 0:
            raise ValueError(f"gate_sup_wE must be >= 0, got {gate_sup_wE}")
        self.shape_loss_mode = shape_loss_mode
        self.trigger_mask_mode = trigger_mask_mode
        self.seq_len = seq_len
        self.use_instance_norm = use_instance_norm
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.gate_log_mag = gate_log_mag
        self.gate_arch = gate_arch
        self.gate_threshold_mode = gate_threshold_mode
        self.gate_entropy_weight = float(gate_entropy_weight)
        # gate_use_log_mag may be None -> fall back to legacy gate_log_mag
        self.gate_use_log_mag = gate_use_log_mag
        self.stationarity_loss_weight = stationarity_loss_weight
        self.stationarity_chunks = stationarity_chunks
        self.future_mode = future_mode
        self.predict_n_time = predict_n_time
        self.pred_loss_weight = pred_loss_weight
        self.gate_smooth_weight = gate_smooth_weight
        self.gate_temporal_smooth_weight = gate_temporal_smooth_weight
        self.gate_ratio_weight = gate_ratio_weight
        self.gate_ratio_target = gate_ratio_target
        self.gate_mode = gate_mode
        self.pred_input = pred_input
        self.lambda_E = float(lambda_E)
        self.lambda_P = float(lambda_P)
        self.eps_E = float(eps_E)
        self.delta_E = float(delta_E)
        self.delta_P = float(delta_P)
        self.trigger_mask = bool(trigger_mask)
        self.delta_E_mask = float(delta_E_mask)
        self.delta_P_mask = float(delta_P_mask)
        self.gate_sup_enable = bool(gate_sup_enable)
        self.gate_sup_weight = float(gate_sup_weight)
        self.gate_sup_tau = float(gate_sup_tau)
        self.gate_sup_loss = gate_sup_loss
        self.gate_sup_pos_weight = float(gate_sup_pos_weight)
        self.gate_sup_on = gate_sup_on
        self.gate_sup_target = gate_sup_target
        self.gate_sup_source = gate_sup_source
        self.gate_sup_wE = float(gate_sup_wE)

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
            gate_mode=gate_mode,
            gate_budget_dim=gate_budget_dim,
            gate_ratio_target=gate_ratio_target,
            gate_arch=self.gate_arch,
            gate_threshold_mode=self.gate_threshold_mode,
        )
        in_frames = self.stft.time_bins(self.seq_len)
        out_frames = self.stft.time_bins(self.pred_len)
        
        # Determine predictor input feature dimension
        # n_tf only: 2 features (real, imag)
        # n_tf_x_tf: 4 features (n_real, n_imag, x_real, x_imag)
        input_feature_dim = 4 if pred_input == "n_tf_x_tf" else 2
        
        self.n_tf_predictor = (
            SimpleTFPredictor(
                in_frames,
                out_frames,
                hidden_dim=pred_hidden_dim,
                dropout=pred_dropout,
                input_feature_dim=input_feature_dim,
            )
            if predict_n_time
            else None
        )
        
        # Store last gate stats for monitoring
        self._last_gate_mean = 0.0
        self._last_gate_sum_f = 0.0
        self._last_gate_max_f = 0.0
        self._last_gate_ent_f = 0.0

        self._last_state: Optional[LocalTFNormState] = None
        self._last_residual: Optional[torch.Tensor] = None
        self._last_gate: Optional[torch.Tensor] = None
        self._last_r_tf: Optional[torch.Tensor] = None
        self._last_L_E: float = 0.0
        self._last_L_P: float = 0.0
        self._last_stationarity_loss: float = 0.0

        # Extended diagnostic caches (detached, no grad)
        self._last_x_time: Optional[torch.Tensor] = None      # (B,T,C) after RevIN
        self._last_x_tf: Optional[torch.Tensor] = None        # (B,C,F,TT) complex
        self._last_n_tf: Optional[torch.Tensor] = None        # (B,C,F,TT) complex
        self._last_n_time: Optional[torch.Tensor] = None      # (B,T,C)
        self._last_r_time: Optional[torch.Tensor] = None      # (B,T,C) == _last_residual
        self._last_pred_n_tf: Optional[torch.Tensor] = None   # (B,C,F,TT) or None
        self._last_pred_n_time: Optional[torch.Tensor] = None # (B,T,C) or None
        self._last_inst_mean: Optional[torch.Tensor] = None   # (B,1,C) or None
        self._last_inst_std: Optional[torch.Tensor] = None    # (B,1,C) or None
        self._last_dE: Optional[torch.Tensor] = None          # (B,C,T-1) log-energy steps
        self._last_dP: Optional[torch.Tensor] = None          # (B,C,T-1) shape steps

        # Denormalize branch usage counters (reset each epoch by caller)
        self._denorm_used_pred: int = 0
        self._denorm_used_input: int = 0
        self._denorm_used_extrap: int = 0

        # Trigger mask diagnostic caches (all detached)
        self._last_mask_rate: float = 1.0
        self._last_mask_trig_rate: float = 0.0
        self._last_mask_delta_E: float = float("nan")
        self._last_mask_delta_P: float = float("nan")
        self._last_trigger_mask_t: Optional[torch.Tensor] = None

        # Trigger mask coverage diagnostics
        self._last_mask_cov: float = float("nan")
        self._last_mask_min_rate_cov95: float = float("nan")
        self._last_mask_min_rate_cov99: float = float("nan")
        self._last_mask_min_rate_cov995: float = float("nan")
        self._last_mask_p_table: dict = {}  # key: p (float), value: (mask_rate, coverage)

        # Pre-RevIN raw input cache (needed for teacher extraction)
        self._last_x_raw: Optional[torch.Tensor] = None

        # Prediction supervision loss cache
        self._last_pred_sup_loss: float = 0.0

        # Gate quality (GQ) diagnostic caches (detached, no grad)
        self._last_gq_entF_norm: float = float("nan")
        self._last_gq_topk_mass: float = float("nan")
        self._last_gq_maxF_meanF: float = float("nan")
        self._last_gq_corr_mag: float = float("nan")

        # TF-mask diagnostics (only populated when trigger_mask_mode=="tf")
        self._last_mask_rate_tf: float = float("nan")
        self._last_mask_trig_rate_tf: float = float("nan")

        # Gate supervision diagnostic caches (detached, no grad)
        self._dbg_graw_stdF: float = float("nan")
        self._dbg_geff_stdF: float = float("nan")
        self._dbg_graw_entF: float = float("nan")
        self._dbg_geff_entF: float = float("nan")
        self._dbg_graw_maxF_meanF: float = float("nan")
        self._dbg_geff_maxF_meanF: float = float("nan")
        self._dbg_graw_topk_mass: float = float("nan")
        self._dbg_geff_topk_mass: float = float("nan")
        self._dbg_sup_pos_mean: float = float("nan")
        self._dbg_sup_neg_mean: float = float("nan")
        self._dbg_sup_pos_neg_ratio: float = float("nan")
        self._dbg_sup_bce: float = float("nan")
        self._dbg_sup_pos_rate: float = float("nan")
        self._dbg_sup_mixed_rate: float = float("nan")
        self._dbg_sup_y_stdF: float = float("nan")

        # Gate supervision loss tensor (with grad) for loss() aggregation
        self._last_L_sup_tensor: Optional[torch.Tensor] = None

    def _dE_dP_from_tf(
        self, tf_complex: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-transition log-energy and spectral-shape differences.

        Uses the same formula as loss() but accepts any TF tensor as input.

        Args:
            tf_complex: (B, C, F, T) complex tensor

        Returns:
            dE: (B, C, T-1) log-energy differences
            dP: (B, C, T-1) spectral-shape differences
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
        """Return the magnitude representation fed into the gate (same as _make_gate logic)."""
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
        # decide whether to use log-mag: explicit override wins, else legacy flag
        return self.gate(magnitude)

    def _compute_gate_stats(self, g_local: torch.Tensor) -> None:
        """
        Compute gate statistics based on last forward pass.
        g_local: (B, C, F, T) gate activations
        
        Stats:
        - gate_mean: mean activation across all dims
        - gate_sum_f: sum along freq dim (F), then take mean over batch/channel/time
        - gate_max_f: max along freq dim (F), then take mean over batch/channel/time
        - ent_f: normalized entropy along freq dim, then take mean over batch/channel/time
        """
        # gate_mean: global mean
        self._last_gate_mean = float(g_local.mean().detach().cpu().item())
        
        # Sum and max along frequency dimension (dim=2)
        # From (B, C, F, T), sum/max over F to get (B, C, T)
        g_sum_f = g_local.sum(dim=2)  # (B, C, T)
        g_max_f = g_local.max(dim=2)[0]  # (B, C, T)
        
        # Take mean over batch/channel/time to get scalar
        self._last_gate_sum_f = float(g_sum_f.mean().detach().cpu().item())
        self._last_gate_max_f = float(g_max_f.mean().detach().cpu().item())
        
        # Entropy: compute per (batch, channel, time) point along freq
        # First normalize g_local along freq dimension to get probability
        g_norm = g_local / (g_local.sum(dim=2, keepdim=True) + 1e-10)  # (B, C, F, T)
        
        # Compute entropy: -sum(p * log(p))
        eps = 1e-10
        entropy = -(g_norm * torch.log(g_norm + eps)).sum(dim=2)  # (B, C, T)
        
        # Normalize by max entropy: log(F)
        f_size = g_local.shape[2]
        max_entropy = np.log(f_size)
        if max_entropy > 0:
            entropy_norm = entropy / max_entropy
        else:
            entropy_norm = entropy
        
        # Take mean over batch/channel/time
        self._last_gate_ent_f = float(entropy_norm.mean().detach().cpu().item())

    def get_last_gate_stats(self) -> dict[str, float]:
        """Return last computed gate statistics for monitoring."""
        return {
            "gate_mean": self._last_gate_mean,
            "gate_sum_f": self._last_gate_sum_f,
            "gate_max_f": self._last_gate_max_f,
            "gate_ent_f": self._last_gate_ent_f,
        }

    def get_last_stationarity_stats(self) -> dict[str, float]:
        """Return last computed stationarity loss components for logging."""
        return {
            "stationarity_loss": self._last_stationarity_loss,
            "L_E": self._last_L_E,
            "L_P": self._last_L_P,
        }

    @torch.no_grad()
    def _compute_gq_stats(
        self,
        g_local: torch.Tensor,
        gate_magnitude: torch.Tensor,
        mask_t: Optional[torch.Tensor],
    ) -> None:
        """Compute gate quality diagnostics (all detached, no grad).

        Args:
            g_local:        (B, C, F, T) raw gate output before trigger mask.
            gate_magnitude: (B, C, F, T) magnitude fed into the gate.
            mask_t:         (B, C, T) bool mask of valid frames, or None for all frames.
        """
        g = g_local.detach().float()
        mag = gate_magnitude.detach().float()
        B, C, F, T = g.shape

        # Build per-(B,C,T) validity mask
        if mask_t is not None and mask_t.any():
            valid = mask_t.detach().bool()  # (B, C, T)
        else:
            valid = torch.ones(B, C, T, dtype=torch.bool, device=g.device)

        valid_f = valid.float()  # (B, C, T) for weighting

        # ---- (a) entF_norm: per-(B,C,T) softmax of g(f) → H / ln(F) ----
        probs = torch.softmax(g, dim=2)  # (B, C, F, T)
        eps = 1e-10
        ent = -(probs * torch.log(probs + eps)).sum(dim=2)  # (B, C, T)
        log_F = float(np.log(F)) if F > 1 else 1.0
        ent_norm = ent / log_F  # (B, C, T), in [0, 1]
        n_valid = float(valid_f.sum().item())
        if n_valid > 0:
            entF_norm = float((ent_norm * valid_f).sum().item() / n_valid)
        else:
            entF_norm = float("nan")

        # ---- (b) topk_mass: top-k probability mass ----
        k = min(5, F)
        topk_probs, _ = torch.topk(probs, k, dim=2)  # (B, C, k, T)
        topk_mass_t = topk_probs.sum(dim=2)            # (B, C, T)
        if n_valid > 0:
            topk_mass = float((topk_mass_t * valid_f).sum().item() / n_valid)
        else:
            topk_mass = float("nan")

        # ---- (c) maxF/meanF ----
        g_max_f = g.max(dim=2)[0]            # (B, C, T)
        g_mean_f = g.mean(dim=2)             # (B, C, T)
        ratio = g_max_f / (g_mean_f + 1e-10) # (B, C, T)
        if n_valid > 0:
            maxF_meanF = float((ratio * valid_f).sum().item() / n_valid)
        else:
            maxF_meanF = float("nan")

        # ---- (d) corr_mag: Pearson correlation of g_local vs gate_magnitude ----
        # Select entries where valid_f expanded over F
        valid_4d = valid.unsqueeze(2).expand_as(g)  # (B, C, F, T) bool
        g_sel = g[valid_4d]      # (N,)
        mag_sel = mag[valid_4d]  # (N,)
        if g_sel.numel() > 1:
            g_c = g_sel - g_sel.mean()
            m_c = mag_sel - mag_sel.mean()
            cov = (g_c * m_c).mean()
            g_std = g_c.std() + 1e-10
            m_std = m_c.std() + 1e-10
            corr_mag = float((cov / (g_std * m_std)).item())
        else:
            corr_mag = float("nan")

        # Cache results
        self._last_gq_entF_norm = entF_norm
        self._last_gq_topk_mass = topk_mass
        self._last_gq_maxF_meanF = maxF_meanF
        self._last_gq_corr_mag = corr_mag

    def get_last_gq_stats(self) -> dict[str, float]:
        """Return last computed gate quality diagnostics."""
        return {
            "entF_norm": self._last_gq_entF_norm,
            "topk_mass": self._last_gq_topk_mass,
            "maxF_meanF": self._last_gq_maxF_meanF,
            "corr_mag": self._last_gq_corr_mag,
        }

    @torch.no_grad()
    def _freq_stats(self, g: torch.Tensor, topk: int = 5) -> tuple[float, float, float, float]:
        """Compute frequency-axis gate statistics for diagnostics.

        Args:
            g: (B, C, F, T) gate tensor
            topk: number of top frequency bins for mass computation

        Returns:
            (stdF_mean, entF_norm, maxF_meanF, topk_mass)
        """
        g = g.detach().float()
        F = g.shape[2]
        stdF_mean = g.std(dim=2).mean().item()
        g_sum = g.sum(dim=2, keepdim=True) + 1e-8
        p = g / g_sum
        log_F = math.log(F) if F > 1 else 1.0
        entF_norm = (-(p * (p + 1e-8).log()).sum(dim=2) / log_F).mean().item()
        maxF_meanF = (g.max(dim=2).values / (g.mean(dim=2) + 1e-8)).mean().item()
        k = min(topk, F)
        topk_mass = p.topk(k=k, dim=2).values.sum(dim=2).mean().item()
        return stdF_mean, entF_norm, maxF_meanF, topk_mass

    def _gate_sup_target_from_stats(
        self,
        dE_x: torch.Tensor,
        dP_f_x: torch.Tensor,
        de_thr: float,
        dp_thr: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-TF-cell supervision target from change statistics.

        Args:
            dE_x:   (B, C, T-1)   per-transition log-energy diff
            dP_f_x: (B, C, F, T-1) per-frequency spectral-shape diff
            de_thr: energy threshold (scalar)
            dp_thr: shape threshold (scalar)

        Returns:
            y_tf:    (B, C, F, T-1)  supervision target in [0, 1]
            trig_tf: (B, C, F, T-1)  boolean trigger mask (s > 1)
        """
        sP = dP_f_x / (dp_thr + 1e-8)               # (B, C, F, T-1)
        if self.gate_sup_source == "P_only":
            s = sP
        else:  # "EP_max"
            sE = dE_x.unsqueeze(2) / (de_thr + 1e-8)  # (B, C, 1, T-1) → broadcast over F
            s = torch.maximum(self.gate_sup_wE * sE, sP)  # (B, C, F, T-1)
        trig_tf = s > 1.0                            # (B, C, F, T-1) bool
        if self.gate_sup_target == "hard":
            y_tf = trig_tf.float()
        else:  # "soft"
            y_tf = torch.sigmoid((s - 1.0) / self.gate_sup_tau)
        return y_tf, trig_tf

    def _weighted_bce(
        self, g: torch.Tensor, y: torch.Tensor, pos_weight: float
    ) -> torch.Tensor:
        """BCE loss with positive-class weighting."""
        eps = 1e-7
        w = pos_weight * y + (1.0 - y)
        loss = -(w * (y * torch.log(g + eps) + (1.0 - y) * torch.log(1.0 - g + eps)))
        return loss.mean()

    def _focal_bce(
        self,
        g: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: float = 5.0,
    ) -> torch.Tensor:
        """Focal BCE loss with positive-class weighting."""
        eps = 1e-7
        bce = -(y * torch.log(g + eps) + (1.0 - y) * torch.log(1.0 - g + eps))
        p_t = g * y + (1.0 - g) * (1.0 - y)
        focal = (1.0 - p_t) ** gamma
        w = pos_weight * y + (1.0 - y)
        return (alpha * focal * bce * w).mean()

    def normalize(
        self, batch_x: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, LocalTFNormState]:
        # Cache raw (pre-RevIN) input for teacher extraction in pred_supervision_loss
        self._last_x_raw = batch_x.detach()

        # RevIN-style instance normalization: remove per-sample mean and std
        inst_mean = inst_std = None
        if self.use_instance_norm:
            inst_mean = batch_x.mean(dim=1, keepdim=True)          # (B, 1, C)
            inst_std = batch_x.std(dim=1, keepdim=True) + 1e-8     # (B, 1, C)
            batch_x = (batch_x - inst_mean) / inst_std

        x_tf = self.stft(batch_x)
        gate_magnitude = self._compute_gate_magnitude(x_tf)
        g_local = self.gate(gate_magnitude)
        g_raw = g_local  # always the pre-mask gate output

        # Always compute x_tf change statistics (needed for trigger mask and/or supervision)
        with torch.no_grad():
            dE_x, dP_mean_x = self._dE_dP_from_tf(x_tf)
            dP_f_x, _ = self._dP_mean_and_perF(x_tf)

        # Default supervision thresholds (overwritten inside trigger_mask block)
        _sup_de: float = max(float(self.delta_E), 1e-8)
        _sup_dp: float = max(float(self.delta_P), 1e-8)

        # Supervision targets — computed unconditionally after the if/else block
        y_tf_sup: Optional[torch.Tensor] = None
        trig_tf_sup: Optional[torch.Tensor] = None

        # Trigger mask: restrict effective gate to boundary frames only
        if self.trigger_mask:
            de = self.delta_E_mask if self.delta_E_mask > 0.0 else self.delta_E
            dp = self.delta_P_mask if self.delta_P_mask > 0.0 else self.delta_P
            if de <= 0.0 or dp <= 0.0:
                raise RuntimeError(
                    "trigger_mask enabled but delta_E_mask/delta_P_mask not set; "
                    "run calibration or set manually"
                )
            _sup_de = de
            _sup_dp = dp
            B_sz, C_sz, F_sz, T = x_tf.shape[0], x_tf.shape[1], x_tf.shape[2], x_tf.shape[-1]
            self._last_mask_delta_E = float(de)
            self._last_mask_delta_P = float(dp)

            if self.trigger_mask_mode == "time":
                # ---- TIME mode: existing behaviour (broadcast over freq) ----
                trig = (dE_x > de) | (dP_mean_x > dp)           # (B, C, T-1)
                mask_t = torch.zeros(B_sz, C_sz, T, dtype=torch.bool, device=x_tf.device)
                mask_t[..., :-1] |= trig
                mask_t[..., 1:] |= trig
                mask_f = mask_t.unsqueeze(2).float()             # (B, C, 1, T)
                g_eff = g_local * mask_f                         # (B, C, F, T)

                self._last_trigger_mask_t = mask_t.detach()
                self._last_mask_rate = float(mask_t.float().mean().cpu().item())
                self._last_mask_trig_rate = float(trig.float().mean().cpu().item())
                self._last_mask_rate_tf = float("nan")
                self._last_mask_trig_rate_tf = float("nan")

                # Coverage: strength in (B,C,T-1) / (B,C,T) space
                _s_trans = torch.maximum(dE_x / de, dP_mean_x / dp)
                _s_frame = torch.zeros(B_sz, C_sz, T, device=x_tf.device, dtype=_s_trans.dtype)
                _s_frame[..., :-1] = torch.maximum(_s_frame[..., :-1], _s_trans)
                _s_frame[..., 1:] = torch.maximum(_s_frame[..., 1:], _s_trans)
                _cov_den = float(_s_frame.sum().cpu().item()) + 1e-12
                _cov_num = float((_s_frame * mask_t.float()).sum().cpu().item())
                self._last_mask_cov = _cov_num / _cov_den

                # p-table
                _p_table: dict = {}
                _dE_flat = dE_x.detach().float().flatten()
                _dP_flat = dP_mean_x.detach().float().flatten()
                for _p in (0.95, 0.99, 0.995):
                    _de_p = float(torch.quantile(_dE_flat, _p).item())
                    _dp_p = float(torch.quantile(_dP_flat, _p).item())
                    _trig_p = (dE_x > _de_p) | (dP_mean_x > _dp_p)
                    _mask_t_p = torch.zeros(B_sz, C_sz, T, dtype=torch.bool, device=x_tf.device)
                    _mask_t_p[..., :-1] |= _trig_p
                    _mask_t_p[..., 1:] |= _trig_p
                    _mr_p = float(_mask_t_p.float().mean().cpu().item())
                    _de_p_safe = _de_p if _de_p > 1e-15 else 1e-15
                    _dp_p_safe = _dp_p if _dp_p > 1e-15 else 1e-15
                    _s_trans_p = torch.maximum(dE_x / _de_p_safe, dP_mean_x / _dp_p_safe)
                    _s_frame_p = torch.zeros_like(_s_frame)
                    _s_frame_p[..., :-1] = torch.maximum(_s_frame_p[..., :-1], _s_trans_p)
                    _s_frame_p[..., 1:] = torch.maximum(_s_frame_p[..., 1:], _s_trans_p)
                    _cdn_p = float(_s_frame_p.sum().cpu().item()) + 1e-12
                    _cnm_p = float((_s_frame_p * _mask_t_p.float()).sum().cpu().item())
                    _p_table[_p] = (_mr_p, _cnm_p / _cdn_p)
                self._last_mask_p_table = _p_table

                # min_rate_cov*
                _sf_flat = _s_frame.detach().float().flatten()
                _sf_total = float(_sf_flat.sum().item())

            else:
                # ---- TF mode: frequency-aware mask ----
                trig_tf = (dE_x.unsqueeze(2) > de) | (dP_f_x > dp)  # (B, C, F, T-1)
                mask_tf = torch.zeros(B_sz, C_sz, F_sz, T, dtype=torch.bool, device=x_tf.device)
                mask_tf[..., :-1] |= trig_tf
                mask_tf[..., 1:] |= trig_tf
                g_eff = g_local * mask_tf.float()                    # (B, C, F, T)

                # Legacy (B,C,T) fields for GQ and existing logging
                mask_t_any = mask_tf.any(dim=2)                      # (B, C, T)
                self._last_trigger_mask_t = mask_t_any.detach()
                self._last_mask_rate = float(mask_t_any.float().mean().cpu().item())
                trig_t_any = trig_tf.any(dim=2)                      # (B, C, T-1)
                self._last_mask_trig_rate = float(trig_t_any.float().mean().cpu().item())
                # TF-specific rates
                self._last_mask_rate_tf = float(mask_tf.float().mean().cpu().item())
                self._last_mask_trig_rate_tf = float(trig_tf.float().mean().cpu().item())

                # Coverage: strength in (B,C,F,T-1) / (B,C,F,T) space
                _s_trans_tf = torch.maximum(
                    dE_x.unsqueeze(2) / (de + 1e-15),
                    dP_f_x / (dp + 1e-15),
                )  # (B, C, F, T-1)
                _s_frame_tf = torch.zeros(B_sz, C_sz, F_sz, T, device=x_tf.device, dtype=_s_trans_tf.dtype)
                _s_frame_tf[..., :-1] = torch.maximum(_s_frame_tf[..., :-1], _s_trans_tf)
                _s_frame_tf[..., 1:] = torch.maximum(_s_frame_tf[..., 1:], _s_trans_tf)
                _cov_den = float(_s_frame_tf.sum().cpu().item()) + 1e-12
                _cov_num = float((_s_frame_tf * mask_tf.float()).sum().cpu().item())
                self._last_mask_cov = _cov_num / _cov_den

                # p-table (dP quantile over dP_f_x elements)
                _p_table = {}
                _dE_flat = dE_x.detach().float().flatten()
                _dP_flat = dP_f_x.detach().float().flatten()
                for _p in (0.95, 0.99, 0.995):
                    _de_p = float(torch.quantile(_dE_flat, _p).item())
                    _dp_p = float(torch.quantile(_dP_flat, _p).item())
                    _trig_tf_p = (dE_x.unsqueeze(2) > _de_p) | (dP_f_x > _dp_p)
                    _mask_tf_p = torch.zeros(B_sz, C_sz, F_sz, T, dtype=torch.bool, device=x_tf.device)
                    _mask_tf_p[..., :-1] |= _trig_tf_p
                    _mask_tf_p[..., 1:] |= _trig_tf_p
                    _mr_p = float(_mask_tf_p.float().mean().cpu().item())
                    _de_p_safe = _de_p if _de_p > 1e-15 else 1e-15
                    _dp_p_safe = _dp_p if _dp_p > 1e-15 else 1e-15
                    _s_trans_tf_p = torch.maximum(dE_x.unsqueeze(2) / _de_p_safe, dP_f_x / _dp_p_safe)
                    _s_frame_tf_p = torch.zeros_like(_s_frame_tf)
                    _s_frame_tf_p[..., :-1] = torch.maximum(_s_frame_tf_p[..., :-1], _s_trans_tf_p)
                    _s_frame_tf_p[..., 1:] = torch.maximum(_s_frame_tf_p[..., 1:], _s_trans_tf_p)
                    _cdn_p = float(_s_frame_tf_p.sum().cpu().item()) + 1e-12
                    _cnm_p = float((_s_frame_tf_p * _mask_tf_p.float()).sum().cpu().item())
                    _p_table[_p] = (_mr_p, _cnm_p / _cdn_p)
                self._last_mask_p_table = _p_table

                # min_rate_cov*
                _sf_flat = _s_frame_tf.detach().float().flatten()
                _sf_total = float(_sf_flat.sum().item())

            # ---- Shared min_rate_cov* computation (same logic, different _sf_flat) ----
            if _sf_total > 0.0 and _sf_flat.numel() > 0:
                _sf_sorted, _ = torch.sort(_sf_flat, descending=True)
                _csum = torch.cumsum(_sf_sorted, dim=0)
                _N = float(_sf_flat.numel())
                for _tgt, _attr in (
                    (0.95, "_last_mask_min_rate_cov95"),
                    (0.99, "_last_mask_min_rate_cov99"),
                    (0.995, "_last_mask_min_rate_cov995"),
                ):
                    _k = int(torch.searchsorted(_csum, _tgt * _sf_total).item())
                    setattr(self, _attr, (_k + 1) / _N)
            else:
                self._last_mask_min_rate_cov95 = float("nan")
                self._last_mask_min_rate_cov99 = float("nan")
                self._last_mask_min_rate_cov995 = float("nan")

        else:
            g_eff = g_local
            self._last_trigger_mask_t = None
            self._last_mask_rate = 1.0
            self._last_mask_trig_rate = 0.0
            self._last_mask_delta_E = float("nan")
            self._last_mask_delta_P = float("nan")
            self._last_mask_cov = float("nan")
            self._last_mask_min_rate_cov95 = float("nan")
            self._last_mask_min_rate_cov99 = float("nan")
            self._last_mask_min_rate_cov995 = float("nan")
            self._last_mask_p_table = {}
            self._last_mask_rate_tf = float("nan")
            self._last_mask_trig_rate_tf = float("nan")

        # Always build supervision targets unconditionally (1.4)
        with torch.no_grad():
            y_tf_sup, trig_tf_sup = self._gate_sup_target_from_stats(
                dE_x, dP_f_x, _sup_de, _sup_dp
            )
            # 1.6: New diagnostics — pos_rate, mixed_rate, y_stdF
            self._dbg_sup_pos_rate = float(trig_tf_sup.float().mean().item())
            _mixed = trig_tf_sup.any(dim=2) & (~trig_tf_sup.all(dim=2))  # (B,C,T-1)
            self._dbg_sup_mixed_rate = float(_mixed.float().mean().item())
            self._dbg_sup_y_stdF = float(y_tf_sup.std(dim=2).mean().item())

        # Gate supervision loss
        _bce_scalar = float("nan")
        L_sup = torch.tensor(0.0, device=x_tf.device)
        if self.gate_sup_enable:
            _g_sup = g_raw if self.gate_sup_on == "raw" else g_eff
            _g_sup_clip = _g_sup[..., 1:].clamp(1e-6, 1.0 - 1e-6)  # (B,C,F,T-1)
            if self.gate_sup_loss == "bce":
                _bce = self._weighted_bce(_g_sup_clip, y_tf_sup, self.gate_sup_pos_weight)
            else:  # "focal"
                _bce = self._focal_bce(
                    _g_sup_clip, y_tf_sup, 0.25, 2.0, self.gate_sup_pos_weight
                )
            _bce_scalar = _bce.item()
            L_sup = _bce * self.gate_sup_weight
        self._last_L_sup_tensor = L_sup  # stored with grad for loss() aggregation

        # keep effective gate for debugging (reflects actual decomposition used)
        self._last_gate = g_eff
        n_tf = g_eff * x_tf
        r_tf = x_tf - n_tf
        self._last_r_tf = r_tf
        length = batch_x.shape[1]
        residual = self.stft.inverse(r_tf, length=length)
        n_time = self.stft.inverse(n_tf, length=length)

        # Compute gate statistics on effective gate
        self._compute_gate_stats(g_eff)

        # Compute gate quality (GQ) diagnostics on raw g_local (pre-mask)
        _mask_t_for_gq = self._last_trigger_mask_t if self.trigger_mask else None
        self._compute_gq_stats(g_local, gate_magnitude, _mask_t_for_gq)

        pred_n_time = None
        pred_n_tf = None
        if self.n_tf_predictor is not None:
            # Prepare predictor input features
            if self.pred_input == "n_tf_x_tf":
                # Concatenate n_tf and x_tf as real/imag features
                # n_tf: (B, C, F, T) complex -> (B, C, F, T, 2) real
                # x_tf: (B, C, F, T) complex -> (B, C, F, T, 2) real
                # concat: (B, C, F, T, 4) = [n_real, n_imag, x_real, x_imag]
                n_ri = torch.view_as_real(n_tf)
                x_ri = torch.view_as_real(x_tf)
                features = torch.cat([n_ri, x_ri], dim=-1)  # (B, C, F, T, 4)
            else:
                # Only n_tf: (B, C, F, T, 2)
                features = torch.view_as_real(n_tf)
            
            # Pass features to predictor: (B, C, F, T, feat_dim) -> complex (B, C, F, out_T)
            pred_n_tf = self.n_tf_predictor(features)
            pred_n_time = self.stft.inverse(pred_n_tf, length=self.pred_len)

        state = LocalTFNormState(
            x_tf=x_tf,
            n_tf=n_tf,
            g_local=g_local,
            n_time=n_time,
            length=length,
            pred_n_time=pred_n_time,
            pred_n_tf=pred_n_tf,
            mean=inst_mean,
            std=inst_std,
        )
        self._last_state = state
        self._last_residual = residual

        # Extended diagnostic caches (all detached)
        self._last_x_time = batch_x.detach()
        self._last_x_tf = x_tf.detach()
        self._last_n_tf = n_tf.detach()
        self._last_n_time = n_time.detach()
        self._last_r_time = residual.detach()
        self._last_inst_mean = inst_mean.detach() if inst_mean is not None else None
        self._last_inst_std = inst_std.detach() if inst_std is not None else None
        self._last_pred_n_tf = pred_n_tf.detach() if pred_n_tf is not None else None
        self._last_pred_n_time = pred_n_time.detach() if pred_n_time is not None else None

        # 1.8: Update gate supervision diagnostic caches
        with torch.no_grad():
            _sr, _se, _sm, _st = self._freq_stats(g_raw)
            self._dbg_graw_stdF = _sr
            self._dbg_graw_entF = _se
            self._dbg_graw_maxF_meanF = _sm
            self._dbg_graw_topk_mass = _st

            _sr, _se, _sm, _st = self._freq_stats(g_eff)
            self._dbg_geff_stdF = _sr
            self._dbg_geff_entF = _se
            self._dbg_geff_maxF_meanF = _sm
            self._dbg_geff_topk_mass = _st

            _g_sup_d = (g_raw if self.gate_sup_on == "raw" else g_eff).detach().float()
            _g_sup_d_clip = _g_sup_d[..., 1:]  # (B,C,F,T-1) — aligned with g_sup[...,1:]
            _n_pos = int(trig_tf_sup.sum().item())
            _n_neg = int((~trig_tf_sup).sum().item())
            if _n_pos == 0 or _n_neg == 0:
                self._dbg_sup_pos_mean = float("nan")
                self._dbg_sup_neg_mean = float("nan")
                self._dbg_sup_pos_neg_ratio = float("nan")
            else:
                _pos = _g_sup_d_clip[trig_tf_sup]
                _neg = _g_sup_d_clip[~trig_tf_sup]
                self._dbg_sup_pos_mean = float(_pos.mean().item())
                self._dbg_sup_neg_mean = float(_neg.mean().item())
                self._dbg_sup_pos_neg_ratio = self._dbg_sup_pos_mean / (
                    self._dbg_sup_neg_mean + 1e-8
                )
            self._dbg_sup_bce = _bce_scalar

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
        if state.pred_n_time is not None and target_len == state.pred_n_time.shape[1]:
            self._denorm_used_pred += 1
            result = batch_x + state.pred_n_time
        elif target_len == state.length:
            self._denorm_used_input += 1
            result = batch_x + state.n_time
        else:
            self._denorm_used_extrap += 1
            n_time = self._extrapolate_n_time(state.n_time, target_len)
            result = batch_x + n_time
        # Reverse RevIN: restore instance mean and std
        if state.mean is not None and state.std is not None:
            result = result * state.std + state.mean
        return result

    def loss(self, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Stationarity loss computed in the TF domain from the last normalize() call.

        L_stat = lambda_E * L_E + lambda_P * L_P

          P    = |r_tf|^2                      (B, C, F, T)  power spectrum of TF residual
          E    = mean_F(P)                     (B, C, T)     per-frame energy
          logE = log(E + eps_E)                (B, C, T)     log-energy
          p    = P / (sum_F(P) + eps)          (B, C, F, T)  normalised spectral shape
          dE   = |logE[...,1:] - logE[...,:-1]|              (B, C, T-1)  log-energy step
          dP   = mean_F(|p[...,:,1:] - p[...,:,:-1]|)        (B, C, T-1)  shape step
          L_E  = mean(relu(dE - delta_E))      margin-triggered log-energy TV
          L_P  = mean(relu(dP - delta_P))      margin-triggered shape TV
        """
        device = self._device()
        zero = torch.tensor(0.0, device=device)

        if self._last_r_tf is None:
            self._last_L_E = 0.0
            self._last_L_P = 0.0
            self._last_stationarity_loss = 0.0
            return zero

        r_tf = self._last_r_tf  # (B, C, F, T) complex
        eps = 1e-8

        P = r_tf.abs() ** 2  # (B, C, F, T)

        # Per-frame energy: mean over frequency bins
        E = P.mean(dim=2)  # (B, C, T)

        # Normalised spectral shape
        p = P / (P.sum(dim=2, keepdim=True) + eps)  # (B, C, F, T)

        # Log-energy margin-triggered TV along time
        logE = torch.log(E + self.eps_E)
        dE = torch.abs(logE[..., 1:] - logE[..., :-1])          # (B, C, T-1)
        L_E = torch.mean(torch.relu(dE - self.delta_E))

        # Spectral-shape margin-triggered TV: branch on shape_loss_mode
        dP_f = torch.abs(p[..., :, 1:] - p[..., :, :-1])        # (B, C, F, T-1)
        dP_mean = dP_f.mean(dim=2)                               # (B, C, T-1)
        if self.shape_loss_mode == "perF":
            # Per-frequency margin loss: relu applied per (B,C,F,T-1) then mean
            L_P = torch.mean(torch.relu(dP_f - self.delta_P))
            # Cache per-freq tensor for diagnostics (flatten over F for STAT printing)
            self._last_dP = dP_f.detach()
        else:  # "meanF" — original behaviour
            L_P = torch.mean(torch.relu(dP_mean - self.delta_P))
            self._last_dP = dP_mean.detach()

        # Cache dE for diagnostics
        self._last_dE = dE.detach()

        L_stat = self.lambda_E * L_E + self.lambda_P * L_P

        # Cache scalar values for logging (detached; stationarity only, not including L_sup)
        self._last_L_E = float(L_E.detach().cpu().item())
        self._last_L_P = float(L_P.detach().cpu().item())
        self._last_stationarity_loss = float(L_stat.detach().cpu().item())

        # 1.7.7 / 1.9: Add gate supervision loss (computed in normalize(), stored with grad)
        L_total = L_stat
        if self._last_L_sup_tensor is not None:
            L_total = L_total + self._last_L_sup_tensor

        return L_total

    def loss_with_target(self, true: torch.Tensor) -> torch.Tensor:
        """Stationarity loss only — no predictor supervision."""
        return self.loss()

    def pred_supervision_loss(self, true: torch.Tensor) -> torch.Tensor:
        """Prediction supervision loss: MSE(pred_n_time, oracle_n_future).

        The oracle is derived via teacher extraction: concatenate the cached
        pre-RevIN input window with `true`, apply the same normalization and
        gate+trigger_mask pipeline via _extract_n_time, and take the last
        pred_len frames as the oracle target.

        Args:
            true: (B, pred_len, C) ground truth in the same scale as the
                  original input batch_x (pre-RevIN).

        Returns:
            Scalar MSE tensor (for backprop); also caches the float value in
            self._last_pred_sup_loss.
        """
        state = self._last_state
        if state is None or state.pred_n_time is None:
            self._last_pred_sup_loss = 0.0
            return torch.tensor(0.0, device=self._device())

        x_raw = self._last_x_raw  # (B, T, C) pre-RevIN
        if x_raw is None:
            self._last_pred_sup_loss = 0.0
            return torch.tensor(0.0, device=self._device())

        dev = self._device()
        true_dev = true.to(dev)
        x_full = torch.cat([x_raw, true_dev], dim=1)  # (B, T+pred_len, C)

        # Save and restore gate._last_logits to avoid polluting debug state
        gate = getattr(self, "gate", None)
        saved_logits = gate._last_logits if gate is not None else None

        with torch.no_grad():
            _, oracle_n_full = self._extract_n_time(
                x_full, mean=state.mean, std=state.std
            )
            oracle_n_future = oracle_n_full[:, -self.pred_len:, :]  # (B, pred_len, C)

        if gate is not None and saved_logits is not None:
            gate._last_logits = saved_logits

        pred_n_time = state.pred_n_time  # (B, pred_len, C)
        if pred_n_time.shape != oracle_n_future.shape:
            self._last_pred_sup_loss = 0.0
            return torch.tensor(0.0, device=dev)

        loss = torch.nn.functional.mse_loss(pred_n_time, oracle_n_future)
        self._last_pred_sup_loss = float(loss.detach().cpu().item())
        return loss

    def get_last_mask_coverage_stats(self) -> dict:
        """Return last computed trigger-mask coverage statistics."""
        return {
            "cov": self._last_mask_cov,
            "min_rate_cov95": self._last_mask_min_rate_cov95,
            "min_rate_cov99": self._last_mask_min_rate_cov99,
            "min_rate_cov995": self._last_mask_min_rate_cov995,
            "p_table": self._last_mask_p_table,
        }

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _extract_n_time(
        self,
        batch_x: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract n_tf and n_time with the same gate+trigger_mask logic as normalize().

        In teacher path (called from pred_supervision_loss) no diagnostic caches
        are updated; gate._last_logits is saved and restored by the caller.
        """
        if mean is not None and std is not None:
            batch_x = (batch_x - mean) / std
        x_tf = self.stft(batch_x)
        g_local = self._make_gate(x_tf)

        # Apply trigger mask if enabled (consistent with normalize(), no cache updates)
        if self.trigger_mask:
            with torch.no_grad():
                dE_x, dP_mean_x = self._dE_dP_from_tf(x_tf)
                if self.trigger_mask_mode == "tf":
                    dP_f_x, _ = self._dP_mean_and_perF(x_tf)
            de = self.delta_E_mask if self.delta_E_mask > 0.0 else self.delta_E
            dp = self.delta_P_mask if self.delta_P_mask > 0.0 else self.delta_P
            if de > 0.0 and dp > 0.0:
                T = x_tf.shape[-1]
                if self.trigger_mask_mode == "tf":
                    trig_tf = (dE_x.unsqueeze(2) > de) | (dP_f_x > dp)
                    mask_tf = torch.zeros(
                        x_tf.shape[0], x_tf.shape[1], x_tf.shape[2], T,
                        dtype=torch.bool, device=x_tf.device,
                    )
                    mask_tf[..., :-1] |= trig_tf
                    mask_tf[..., 1:] |= trig_tf
                    g_eff = g_local * mask_tf.float()
                else:  # "time"
                    trig = (dE_x > de) | (dP_mean_x > dp)
                    mask_t = torch.zeros(
                        x_tf.shape[0], x_tf.shape[1], T,
                        dtype=torch.bool, device=x_tf.device,
                    )
                    mask_t[..., :-1] |= trig
                    mask_t[..., 1:] |= trig
                    mask_f = mask_t.unsqueeze(2).float()
                    g_eff = g_local * mask_f
            else:
                g_eff = g_local
        else:
            g_eff = g_local

        n_tf = g_eff * x_tf
        n_time = self.stft.inverse(n_tf, length=batch_x.shape[1])
        return n_tf, n_time

    @torch.no_grad()
    def extract_n_time_only(self, x_time: torch.Tensor) -> torch.Tensor:
        """Apply RevIN + STFT + gate + iSTFT and return n_time.

        Does NOT write any _last_* cache fields.  Safe to call during debug
        collection without disturbing the training-batch state.
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
        # n_time: (B, T, C)
        source_len = n_time.shape[1]
        if target_len <= source_len:
            return n_time[:, :target_len, :]

        if self.future_mode == "repeat_last":
            last = n_time[:, -1:, :]
            repeat = last.expand(-1, target_len, -1)
            return repeat

        if self.future_mode == "linear":
            if source_len < 2:
                last = n_time[:, -1:, :]
                return last.expand(-1, target_len, -1)
            last = n_time[:, -1:, :]
            prev = n_time[:, -2:-1, :]
            slope = last - prev
            steps = (
                torch.arange(target_len, device=n_time.device, dtype=n_time.dtype)
                .view(1, -1, 1)
            )
            return last + slope * steps

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
