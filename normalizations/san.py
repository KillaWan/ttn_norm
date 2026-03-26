# Adapted from FAN/torch_timeseries/normalizations/SAN.py
# Changes vs FAN original:
#   - normalize() stores pred_stats internally (self._pred_stats) and returns
#     only the normalized tensor (so TTNModel.normalize() sees a single tensor).
#   - denormalize() reads from self._pred_stats when station_pred is None.
#   - loss(true) added: computes mean/std supervision loss using stored pred_stats.
# The FAN experiment used a tuple return + external storage in Model; here the
# stats are stored inside SAN itself so train.py/TTNModel stay generic.
#
# spike_stats extension (vs original adaptation):
#   - When spike_stats=True, detected spike positions are replaced by the
#     per-channel median BEFORE normalization (x_used).  Both the statistics
#     (mean/std sent to MLP) and the actual backbone input come from x_used,
#     so the two are always consistent.
#   - sigma_min clamp prevents variance collapse after inpainting flat windows.
#   - z_clip guards against remaining large z-scores after normalization.
#   - r_max fail-safe: if spike_rate > r_max the inpainting is skipped and
#     the module falls back to vanilla SAN behaviour for that sample.
#   - All stat computation uses x_used.detach() so gradients flow only through
#     the normalization of x_used itself, not through the statistics.
#
# TTN extension (learned future-stats refinement):
#   - Replaces the old eval-only EMA/source-blending TTN.
#   - _TTNRefiner is a GRU-based module that takes the historical patch-stat
#     sequence and the base predictor's future stats, and outputs refined
#     future stats via a correction + direct-proposal dual-path fusion.
#   - When ttn_enabled=True, loss() supervises BOTH the base predictor and the
#     refined stats against the oracle (true future) patch stats.
#   - ttn_detach_base_stats controls whether the refined-stats gradient flows
#     back through the base predictor.
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
class _TTNRefiner(nn.Module):
    """Learned future-stats refinement module for SAN-TTN.

    Inputs
    ------
    hist_stats_seq    : (B, P_hist, C, D_hist)
        Historical patch statistics (mu, log_sigma, delta_mu, delta_log_sigma,
        and optionally delta2_mu, delta2_log_sigma).
    base_future_stats : (B, P_pred, C, 2)
        Base predictor's future stats, last dim = [mu, log_sigma].

    Outputs
    -------
    refined_future_stats : (B, P_pred, C, 2)
    diag                 : dict with alpha_mean, corr_abs_mean, direct_abs_mean
    """

    def __init__(
        self,
        d_hist: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_direct_head: bool,
        gate_hidden_dim: int,
        logsigma_min: float,
        logsigma_max: float,
    ):
        super().__init__()
        self.use_direct_head = use_direct_head
        self.logsigma_min = logsigma_min
        self.logsigma_max = logsigma_max

        self.gru = nn.GRU(
            input_size=d_hist,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Correction head: input = hist_ctx (H) + base_stats (2)
        self.corr_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 2),
        )

        if use_direct_head:
            # Direct proposal head: input = hist_ctx (H) only
            self.direct_head = nn.Sequential(
                nn.Linear(hidden_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 2),
            )
            # Fusion gate head: same input as correction head, output [0,1]
            self.gate_head = nn.Sequential(
                nn.Linear(hidden_dim + 2, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 2),
            )

    def forward(
        self,
        hist_stats_seq: torch.Tensor,
        base_future_stats: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        B, P_hist, C, D = hist_stats_seq.shape
        P_pred = base_future_stats.shape[1]

        # ---- GRU encoding ----
        # (B, P_hist, C, D) → (B*C, P_hist, D)
        h_in = hist_stats_seq.permute(0, 2, 1, 3).reshape(B * C, P_hist, D)
        _, h_n = self.gru(h_in)                  # h_n: (num_layers, B*C, H)
        hist_ctx = h_n[-1].reshape(B, C, -1)     # (B, C, H)

        # ---- Broadcast context to future positions ----
        H = hist_ctx.shape[-1]
        # (B, 1, C, H) → (B, P_pred, C, H)
        hist_ctx_exp = hist_ctx.unsqueeze(1).expand(B, P_pred, C, H)

        # ---- Correction path ----
        # cond: (B, P_pred, C, H+2)
        cond = torch.cat([hist_ctx_exp, base_future_stats], dim=-1)
        delta = self.corr_head(cond)             # (B, P_pred, C, 2)
        theta_A = base_future_stats + delta
        # clamp log_sigma dimension
        theta_A = torch.stack([
            theta_A[..., 0],
            theta_A[..., 1].clamp(self.logsigma_min, self.logsigma_max),
        ], dim=-1)

        if self.use_direct_head:
            # ---- Direct proposal path ----
            theta_B = self.direct_head(hist_ctx_exp)    # (B, P_pred, C, 2)
            theta_B = torch.stack([
                theta_B[..., 0],
                theta_B[..., 1].clamp(self.logsigma_min, self.logsigma_max),
            ], dim=-1)

            # ---- Fusion gate ----
            alpha = torch.sigmoid(self.gate_head(cond))  # (B, P_pred, C, 2)
            refined = alpha * theta_A + (1.0 - alpha) * theta_B
        else:
            refined = theta_A
            alpha = torch.ones_like(theta_A)
            theta_B = theta_A  # placeholder for diag

        # Final log_sigma clamp
        refined = torch.stack([
            refined[..., 0],
            refined[..., 1].clamp(self.logsigma_min, self.logsigma_max),
        ], dim=-1)

        with torch.no_grad():
            diag = {
                "alpha_mean": float(alpha.mean().item()),
                "corr_abs_mean": float(delta.abs().mean().item()),
                "direct_abs_mean": float(
                    (theta_B - base_future_stats).abs().mean().item()
                ) if self.use_direct_head else 0.0,
            }

        return refined, diag


# ======================================================================
class SAN(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        period_len,
        enc_in,
        station_type: str = 'adaptive',
        # ---- spike-robust options (all off by default → original SAN) ----
        spike_stats: bool = False,
        spike_q: float = 0.99,
        spike_dilate: int = 1,
        spike_mode: str = "mad",
        spike_eps: float = 1e-6,
        # ---- numerical guards ----
        sigma_min: float = 1e-3,
        z_clip: float = 8.0,
        r_max: float = 0.08,
        # ---- TTN (learned future-stats refinement) ----
        ttn_enabled: bool = False,
        ttn_hidden_dim: int = 64,
        ttn_num_layers: int = 2,
        ttn_dropout: float = 0.0,
        ttn_use_direct_head: bool = True,
        ttn_use_delta2: bool = True,
        ttn_gate_hidden_dim: int = 64,
        ttn_stats_loss_weight: float = 0.5,
        ttn_base_stats_loss_weight: float = 0.25,
        ttn_detach_base_stats: bool = False,
        ttn_logsigma_min: float = -6.0,
        ttn_logsigma_max: float = 6.0,
        # ---- old TTN params accepted for CLI/config compat, not used ----
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.channels = enc_in
        self.enc_in = enc_in
        self.station_type = station_type

        # Spike-robust options
        self.spike_stats = spike_stats
        self.spike_q = spike_q
        self.spike_dilate = spike_dilate
        self.spike_mode = spike_mode
        self.spike_eps = spike_eps

        # Numerical guards
        self.sigma_min = sigma_min
        self.z_clip = z_clip
        self.r_max = r_max

        # New learned TTN options
        self.ttn_enabled = bool(ttn_enabled)
        self.ttn_hidden_dim = int(ttn_hidden_dim)
        self.ttn_num_layers = int(ttn_num_layers)
        self.ttn_dropout = float(ttn_dropout)
        self.ttn_use_direct_head = bool(ttn_use_direct_head)
        self.ttn_use_delta2 = bool(ttn_use_delta2)
        self.ttn_gate_hidden_dim = int(ttn_gate_hidden_dim)
        self.ttn_stats_loss_weight = float(ttn_stats_loss_weight)
        self.ttn_base_stats_loss_weight = float(ttn_base_stats_loss_weight)
        self.ttn_detach_base_stats = bool(ttn_detach_base_stats)
        self.ttn_logsigma_min = float(ttn_logsigma_min)
        self.ttn_logsigma_max = float(ttn_logsigma_max)

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

        # Build TTN refiner if enabled
        if self.ttn_enabled:
            d_hist = 6 if self.ttn_use_delta2 else 4
            self.ttn_refiner = _TTNRefiner(
                d_hist=d_hist,
                hidden_dim=self.ttn_hidden_dim,
                num_layers=self.ttn_num_layers,
                dropout=self.ttn_dropout,
                use_direct_head=self.ttn_use_direct_head,
                gate_hidden_dim=self.ttn_gate_hidden_dim,
                logsigma_min=self.ttn_logsigma_min,
                logsigma_max=self.ttn_logsigma_max,
            )

        # Internal storage of predicted statistics (set during normalize)
        self._pred_stats:         Optional[torch.Tensor] = None
        self._base_pred_stats:    Optional[torch.Tensor] = None
        self._refined_pred_stats: Optional[torch.Tensor] = None

        # Spike diagnostic caches
        self._last_spike_rate:      float = 0.0
        self._last_spike_thr_mean:  float = 0.0
        self._last_clip_frac:       float = 0.0
        self._last_sigma_min_frac:  float = 0.0

        # Learned TTN diagnostic caches
        self._ttn_last_diag:                     dict  = {}
        self._last_ttn_base_mu_abs_mean:         float = 0.0
        self._last_ttn_base_sigma_mean:          float = 0.0
        self._last_ttn_refined_mu_abs_mean:      float = 0.0
        self._last_ttn_refined_sigma_mean:       float = 0.0
        self._last_ttn_refine_delta_mu_abs_mean:    float = 0.0
        self._last_ttn_refine_delta_sigma_abs_mean: float = 0.0

    # ------------------------------------------------------------------
    # Deprecated / compat stubs (old EMA-TTN interface, now no-ops)
    # ------------------------------------------------------------------

    def extract_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-period mean and std from x (B, T, C).

        Kept for backward compatibility.  Not used by the learned TTN path.
        Returns:
            mu:    (P, 1, C)
            sigma: (P, 1, C)
        """
        bs, length, dim = x.shape
        x_s = x.reshape(bs, -1, self.period_len, dim)
        mu    = x_s.mean(dim=-2)
        sigma = x_s.std(dim=-2).clamp(min=self.sigma_min)
        return mu.mean(dim=0, keepdim=False).unsqueeze(1), \
               sigma.mean(dim=0, keepdim=False).unsqueeze(1)

    def set_ttn_source_state(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        """Deprecated: no-op. The learned TTN no longer uses source calibration."""

    def reset_ttn_state(self) -> None:
        """Deprecated: no-op. The learned TTN has no EMA memory to reset."""

    # ------------------------------------------------------------------
    def _spike_inpaint(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect spikes and replace them with the per-channel median."""
        B, T, C = x.shape

        center = torch.quantile(x, 0.5, dim=1, keepdim=True)

        if self.spike_mode == "mad":
            score = (x - center).abs()
        elif self.spike_mode == "diff":
            diff = (x[:, 1:, :] - x[:, :-1, :]).abs()
            score = F.pad(diff, (0, 0, 1, 0))
        else:
            raise ValueError(f"Unknown spike_mode: {self.spike_mode!r}")

        thr  = torch.quantile(score, self.spike_q, dim=1, keepdim=True)
        mask = score >= thr

        if self.spike_dilate > 0:
            d = self.spike_dilate
            mask_f = mask.float().permute(0, 2, 1).reshape(B * C, 1, T)
            mask_f = F.max_pool1d(
                mask_f, kernel_size=2 * d + 1, stride=1, padding=d
            )
            mask = mask_f.reshape(B, C, T).permute(0, 2, 1).bool()

        x_used = torch.where(mask, center.expand_as(x), x)
        return x_used, mask.float().mean(), thr.mean()

    # ------------------------------------------------------------------
    def _build_model(self):
        seq_len  = self.seq_len // self.period_len
        enc_in   = self.enc_in
        pred_len = self.pred_len_new
        self.model     = _MLP(seq_len, pred_len, enc_in, self.period_len, mode='mean').float()
        self.model_std = _MLP(seq_len, pred_len, enc_in, self.period_len, mode='std').float()

    # ------------------------------------------------------------------
    def _build_hist_stats_seq(self, x_used: torch.Tensor) -> torch.Tensor:
        """Construct TTN historical patch-statistics sequence.

        Args:
            x_used: (B, T, C) — detached, spike-inpainted input.
        Returns:
            hist_stats_seq: (B, P_hist, C, D_hist)
            D_hist = 4 (mu, log_sigma, delta_mu, delta_log_sigma)
                or  6 (+ delta2_mu, delta2_log_sigma) when ttn_use_delta2=True.
        """
        B, T, C = x_used.shape
        P_hist = T // self.period_len
        x_r = x_used[:, :P_hist * self.period_len, :].reshape(
            B, P_hist, self.period_len, C
        )

        mu        = x_r.mean(dim=2)                              # (B, P_hist, C)
        sigma     = x_r.std(dim=2).clamp(min=self.sigma_min)    # (B, P_hist, C)
        log_sigma = torch.log(sigma)

        # First-order differences (pad first step with 0)
        delta_mu = F.pad(torch.diff(mu,        dim=1), (0, 0, 1, 0))
        delta_ls = F.pad(torch.diff(log_sigma, dim=1), (0, 0, 1, 0))

        features = [mu, log_sigma, delta_mu, delta_ls]

        if self.ttn_use_delta2:
            delta2_mu = F.pad(torch.diff(delta_mu, dim=1), (0, 0, 1, 0))
            delta2_ls = F.pad(torch.diff(delta_ls, dim=1), (0, 0, 1, 0))
            features.extend([delta2_mu, delta2_ls])

        return torch.stack(features, dim=-1)   # (B, P_hist, C, D_hist)

    # ------------------------------------------------------------------
    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """(B, T, N) → (B, T, N)  [stats stored in self._pred_stats]"""
        if self.spike_stats:
            x_det = input.detach()
            x_inpainted, spike_rate, thr_mean = self._spike_inpaint(x_det)
            spike_rate_val = float(spike_rate.item())
            self._last_spike_rate     = spike_rate_val
            self._last_spike_thr_mean = float(thr_mean.item())

            if spike_rate_val > self.r_max:
                x_used = x_det
            else:
                x_used = x_inpainted
        else:
            x_used = input
            self._last_spike_rate     = 0.0
            self._last_spike_thr_mean = 0.0

        if self.station_type == 'adaptive':
            bs, length, dim = input.shape

            # ---- Statistics from detached x_used ----
            x_for_stats = x_used.detach()
            x_s  = x_for_stats.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(x_s, dim=-2, keepdim=True)   # (B, P, 1, C)
            std  = torch.std(x_s,  dim=-2, keepdim=True)   # (B, P, 1, C)

            if self.spike_stats:
                std_clamped = torch.clamp(std, min=self.sigma_min)
                self._last_sigma_min_frac = float(
                    (std <= self.sigma_min).float().mean().item()
                )
            else:
                std_clamped = std
                self._last_sigma_min_frac = 0.0

            # ---- Normalize backbone input ----
            x_used_r   = x_used.reshape(bs, -1, self.period_len, dim)
            norm_input = (x_used_r - mean) / (std_clamped + self.epsilon)

            if self.spike_stats and self.z_clip > 0.0:
                norm_input = torch.clamp(norm_input, -self.z_clip, self.z_clip)
                self._last_clip_frac = float(
                    (norm_input.abs() >= self.z_clip - 1e-6).float().mean().item()
                )
            else:
                self._last_clip_frac = 0.0

            # ---- Base MLP stat predictors ----
            x_flat   = x_for_stats
            mean_all = torch.mean(x_flat, dim=1, keepdim=True)
            outputs_mean = (
                self.model(mean.squeeze(2) - mean_all, x_flat - mean_all) * self.weight[0]
                + mean_all * self.weight[1]
            )                                                       # (B, P_pred, C)
            outputs_std = self.model_std(std_clamped.squeeze(2), x_flat)  # (B, P_pred, C)

            if not self.ttn_enabled:
                # Original SAN behavior
                outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
                self._pred_stats         = outputs[:, -self.pred_len_new:, :]
                self._base_pred_stats    = self._pred_stats
                self._refined_pred_stats = self._pred_stats
                self._ttn_last_diag      = {}
            else:
                # ---- Learned TTN refinement ----
                P_pred = self.pred_len_new

                # Base future stats in (B, P_pred, C, 2) format: [mu, log_sigma]
                base_mean      = outputs_mean[:, -P_pred:, :]               # (B, P_pred, C)
                base_std       = outputs_std[:, -P_pred:, :].clamp(
                    min=self.sigma_min
                )                                                            # (B, P_pred, C)
                base_log_sigma = torch.log(base_std)                        # (B, P_pred, C)
                base_future_stats = torch.stack(
                    [base_mean, base_log_sigma], dim=-1
                )                                                            # (B, P_pred, C, 2)

                # Optionally detach so refined loss doesn't flow to base predictor
                ttn_input = (
                    base_future_stats.detach()
                    if self.ttn_detach_base_stats
                    else base_future_stats
                )

                # Historical patch-stat sequence
                hist_stats_seq = self._build_hist_stats_seq(x_for_stats)    # (B, P_hist, C, D)

                # Refine
                refined_future_stats, diag = self.ttn_refiner(
                    hist_stats_seq, ttn_input
                )                                                            # (B, P_pred, C, 2)

                refined_mean      = refined_future_stats[..., 0]            # (B, P_pred, C)
                refined_log_sigma = refined_future_stats[..., 1]
                refined_std       = torch.exp(refined_log_sigma).clamp(
                    min=self.sigma_min
                )                                                            # (B, P_pred, C)

                # Pack into (B, P_pred, 2*C) — format expected by denormalize()
                self._base_pred_stats = torch.cat(
                    [base_mean, base_std], dim=-1
                )                                                            # (B, P_pred, 2C)
                self._refined_pred_stats = torch.cat(
                    [refined_mean, refined_std], dim=-1
                )                                                            # (B, P_pred, 2C)
                self._pred_stats    = self._refined_pred_stats
                self._ttn_last_diag = diag

                # Update TTN diagnostic caches
                with torch.no_grad():
                    rd_mu    = (refined_mean - base_mean).abs()
                    rd_sigma = (refined_std  - base_std).abs()
                    self._last_ttn_base_mu_abs_mean    = float(base_mean.abs().mean().item())
                    self._last_ttn_base_sigma_mean     = float(base_std.mean().item())
                    self._last_ttn_refined_mu_abs_mean = float(refined_mean.abs().mean().item())
                    self._last_ttn_refined_sigma_mean  = float(refined_std.mean().item())
                    self._last_ttn_refine_delta_mu_abs_mean    = float(rd_mu.mean().item())
                    self._last_ttn_refine_delta_sigma_abs_mean = float(rd_sigma.mean().item())

            return norm_input.reshape(bs, length, dim)
        else:
            self._pred_stats          = None
            self._base_pred_stats     = None
            self._refined_pred_stats  = None
            self._last_clip_frac      = 0.0
            self._last_sigma_min_frac = 0.0
            return input

    # ------------------------------------------------------------------
    def denormalize(self, input: torch.Tensor, station_pred=None) -> torch.Tensor:
        """(B, O, N) → (B, O, N).  Uses self._pred_stats if station_pred is None."""
        if station_pred is None:
            station_pred = self._pred_stats
        if self.station_type == 'adaptive' and station_pred is not None:
            bs, length, dim = input.shape
            x    = input.reshape(bs, -1, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std  = station_pred[:, :, self.channels:].unsqueeze(2)
            output = x * (std + self.epsilon) + mean
            return output.reshape(bs, length, dim)
        else:
            return input

    # ------------------------------------------------------------------
    def loss(self, true: torch.Tensor) -> torch.Tensor:
        """Supervision loss for future patch mean/std prediction."""
        if self._pred_stats is None or self.station_type != 'adaptive':
            return torch.tensor(0.0, device=true.device)

        bs, pred_len, n = true.shape
        true_r = true.reshape(bs, -1, self.period_len, n)

        if not self.ttn_enabled:
            # Original SAN behavior: MSE in raw sigma space
            mean_pred = self._pred_stats[:, :, :n]
            std_pred  = self._pred_stats[:, :, n:]
            return (
                F.mse_loss(mean_pred, true_r.mean(dim=2))
                + F.mse_loss(std_pred, true_r.std(dim=2))
            )

        # TTN path: supervise both base and refined stats in log_sigma space
        oracle_mean      = true_r.mean(dim=2)                           # (B, P_pred, C)
        oracle_std       = true_r.std(dim=2).clamp(min=self.sigma_min)  # (B, P_pred, C)
        oracle_log_sigma = torch.log(oracle_std)

        # Base stats loss
        base_mean      = self._base_pred_stats[:, :, :n]
        base_std       = self._base_pred_stats[:, :, n:].clamp(min=self.sigma_min)
        base_log_sigma = torch.log(base_std)
        base_loss = (
            F.mse_loss(base_mean, oracle_mean)
            + F.mse_loss(base_log_sigma, oracle_log_sigma)
        )

        # Refined stats loss
        ref_mean      = self._refined_pred_stats[:, :, :n]
        ref_std       = self._refined_pred_stats[:, :, n:].clamp(min=self.sigma_min)
        ref_log_sigma = torch.log(ref_std)
        refined_loss = (
            F.mse_loss(ref_mean, oracle_mean)
            + F.mse_loss(ref_log_sigma, oracle_log_sigma)
        )

        return (
            self.ttn_base_stats_loss_weight * base_loss
            + self.ttn_stats_loss_weight * refined_loss
        )

    # ------------------------------------------------------------------
    def get_last_spike_stats(self) -> dict:
        return {
            "spike_rate":      self._last_spike_rate,
            "spike_thr_mean":  self._last_spike_thr_mean,
            "clip_frac":       self._last_clip_frac,
            "sigma_min_frac":  self._last_sigma_min_frac,
        }

    # ------------------------------------------------------------------
    def get_last_ttn_stats(self) -> dict:
        """Return learned TTN diagnostics from the most recent normalize() call."""
        if not self.ttn_enabled:
            return {
                "enabled":                       False,
                "base_mu_abs_mean":              0.0,
                "base_sigma_mean":               0.0,
                "refined_mu_abs_mean":           0.0,
                "refined_sigma_mean":            0.0,
                "alpha_mean":                    0.0,
                "corr_abs_mean":                 0.0,
                "direct_abs_mean":               0.0,
                "refine_delta_mu_abs_mean":      0.0,
                "refine_delta_sigma_abs_mean":   0.0,
            }
        return {
            "enabled":                       True,
            "base_mu_abs_mean":              self._last_ttn_base_mu_abs_mean,
            "base_sigma_mean":               self._last_ttn_base_sigma_mean,
            "refined_mu_abs_mean":           self._last_ttn_refined_mu_abs_mean,
            "refined_sigma_mean":            self._last_ttn_refined_sigma_mean,
            "alpha_mean":                    self._ttn_last_diag.get("alpha_mean", 0.0),
            "corr_abs_mean":                 self._ttn_last_diag.get("corr_abs_mean", 0.0),
            "direct_abs_mean":               self._ttn_last_diag.get("direct_abs_mean", 0.0),
            "refine_delta_mu_abs_mean":      self._last_ttn_refine_delta_mu_abs_mean,
            "refine_delta_sigma_abs_mean":   self._last_ttn_refine_delta_sigma_abs_mean,
        }

    # ------------------------------------------------------------------
    def forward(self, batch_x, mode='n', station_pred=None):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x, station_pred)


# ======================================================================
class _MLP(nn.Module):
    """Internal MLP for SAN (verbatim from FAN)."""

    def __init__(self, seq_len, pred_len, enc_in, period_len, mode):
        super(_MLP, self).__init__()
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.channels   = enc_in
        self.period_len = period_len
        self.mode       = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input     = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output     = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x     = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
