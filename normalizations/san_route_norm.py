"""SANRouteNorm — base SAN + output-side route framework.

Design principles
-----------------
- This is a normalization-centric framework.  Route modules do NOT modify the
  backbone operator and do NOT add a second forecasting branch.
- normalize() prepares and caches all state features needed by denormalize() and loss().
- The base predictor (nm.predictor) is always trained in Stage 1.

Route state normalisation policy:
- By default, route-state features are normalised by _normalize_patch_state()
  at the framework layer (nu, dlogsigma).
- omega_spec is exempt: it keeps its raw bounded physical meaning.
- lp_state is handled entirely via a dedicated lp_state_predictor — it bypasses
  the generic RouteStatePredictor and _normalize_patch_state entirely.

lp_state + lp_state_correction — structured mean-state denorm:
  This combination implements a residual-domain SAN paired with a predicted
  future low-pass structured mean-state.  It belongs to the Base family and
  does NOT add a second forecasting branch.

  normalize() special-case:
    1. Compute hist_lowpass = centered_moving_average(x_hist).
    2. Compute hist_residual = x_hist - hist_lowpass.
    3. All base SAN window stats (mu_hist, std_hist) are computed on hist_residual.
    4. z = SAN-normalised hist_residual (residual-domain normalised input).
    5. base predictor learns residual-domain future patch mu/std.
    6. lp_state_predictor encodes hist_lowpass (lowpass branch) and fuses with
       raw x_hist to predict pred_lowpass_fut (B, pred_len, C).
       The backbone does NOT directly observe the low-pass state; only
       lp_state_predictor produces pred_lowpass_fut.

  Stage roles for lp combo:
    Stage 1:     nm.predictor only — residual-domain base aux loss.
    lp_pretrain: nm.lp_state_predictor only — route_state_loss (lp sequence MSE).
    Stage 2:     fm + nm.lp_state_predictor jointly; nm.predictor frozen; task loss only.
                 lp_state_predictor and backbone co-adapt to minimise task loss.
    Joint:       fm + nm.predictor + nm.lp_state_predictor; all trainable;
                 loss = task_loss + route_state_loss_scale * route_state_loss.

  denormalize() special-case:
    y_out = pred_lowpass_fut + mu_res_time + sigma_res_time * y_norm
    where:
      pred_lowpass_fut  predicted future low-pass sequence (structured mean state)
      mu_res_time       residual-domain future mean broadcast to time
      sigma_res_time    residual-domain future std  broadcast to time
    No gate, no correction, no route_path_impl call.

  compute_route_state_loss():
    MSE(pred_lowpass_fut, oracle_lowpass_fut)   [sequence-level]

Stage roles for lp combo (enforced by train_state_routes.py):
  Stage 1:     nm.predictor only — residual-domain base aux loss.
  lp_pretrain: nm.lp_state_predictor only — route_state_loss.
  Stage 2:     backbone only — task loss; nm.predictor + lp_state_predictor frozen.
  Joint:       backbone + nm.predictor + lp_state_predictor — task + route_state_loss.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .san_route_paths import build_route_path
from .san_route_states import build_route_state
from .san_route_states.lp_state import LPState, _centered_moving_average

# Valid route_path values (including backward-compat alias)
_VALID_PATHS = {
    "none",
    "local_transport",
    "residual_content",
    "alignment",
    "gating",
    "lp_state_correction",
    "local_value_parameter",   # alias → local_transport
}
_VALID_STATES = {"none", "nu", "dlogsigma", "omega_spec", "lp_state"}


def _normalize_patch_state(feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Normalise patch features per-batch, per-channel across the patch dimension.

    feat: (B, P, C)
    Returns (feat - mean) / std where mean and std are computed over dim=1 (patches).
    Applied to nu and dlogsigma by default.  omega_spec and lp_state are exempt.
    """
    mean = feat.mean(dim=1, keepdim=True)
    std = feat.std(dim=1, keepdim=True).clamp_min(eps)
    return (feat - mean) / std


class _LPStatePredictor(nn.Module):
    """Dual-branch channel-wise MLP predicting future low-pass sequence.

    Architecture:
      lowpass branch : Linear(seq_len -> lp_hidden) + ReLU
      fusion         : Linear(lp_hidden + seq_len -> hidden) + ReLU + Linear(hidden -> pred_len)

    The hist_lowpass is encoded first; the resulting lowpass_feat is then
    concatenated with raw x_hist before the fusion MLP.  This mirrors the
    FAN design where the primary-component predictor receives both the
    filtered component and the original input.

    lp_hidden = 64, hidden = 128.  Default PyTorch initialisation (no zero-init).
    All parameters are shared across channels (per-channel linear).

    Input:  x_hist      (B, T, C)
            hist_lowpass (B, T, C)
    Output: pred_lowpass_fut  (B, pred_len, C)
    """

    def __init__(self, seq_len: int, pred_len: int, lp_hidden: int = 64, hidden: int = 128):
        super().__init__()
        self.lp_enc = nn.Linear(seq_len, lp_hidden)
        self.lp_act = nn.ReLU()
        self.fusion = nn.Linear(lp_hidden + seq_len, hidden)
        self.fusion_act = nn.ReLU()
        self.out = nn.Linear(hidden, pred_len)

    def forward(
        self,
        x_hist: torch.Tensor,        # (B, T, C)
        hist_lowpass: torch.Tensor,  # (B, T, C)
    ) -> torch.Tensor:
        # permute to (B, C, T) for per-channel shared linear
        lp = hist_lowpass.permute(0, 2, 1)           # (B, C, T)
        xh = x_hist.permute(0, 2, 1)                 # (B, C, T)
        lp_feat = self.lp_act(self.lp_enc(lp))       # (B, C, lp_hidden)
        fused = torch.cat([lp_feat, xh], dim=-1)     # (B, C, lp_hidden + T)
        out = self.out(self.fusion_act(self.fusion(fused)))  # (B, C, pred_len)
        return out.permute(0, 2, 1)                  # (B, pred_len, C)


class SANRouteNorm(nn.Module):
    """Single-level SAN base for state-path routing experiments.

    Args:
        seq_len:               Length of the historical input window.
        pred_len:              Length of the prediction horizon.
        period_len:            Window (patch/slice) length.
        enc_in:                Number of input channels.
        stride:                Stride between windows. Defaults to period_len.
        sigma_min:             Minimum allowed std (clamp floor).
        w_mu:                  Weight for mu component of base aux loss.
        w_std:                 Weight for std component of base aux loss.
        route_path:            "none" | "local_transport" | "residual_content"
                               | "alignment" | "gating" | "lp_state_correction"
                               | "local_value_parameter" (alias → local_transport)
        route_state:           "none" | "nu" | "dlogsigma" | "omega_spec" | "lp_state"
        route_state_loss_scale: Weight for route state loss in compute_total_aux_loss().
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        enc_in: int,
        stride: int = 0,
        sigma_min: float = 1e-3,
        w_mu: float = 1.0,
        w_std: float = 1.0,
        route_path: str = "none",
        route_state: str = "none",
        route_state_loss_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.window_len = int(period_len)
        self.stride = int(stride) if int(stride) > 0 else self.window_len
        self.enc_in = enc_in
        self.channels = enc_in
        self.sigma_min = float(sigma_min)
        self.epsilon = 1e-5
        self.w_mu = float(w_mu)
        self.w_std = float(w_std)
        self.route_path = route_path
        self.route_state = route_state
        self.route_state_loss_scale = float(route_state_loss_scale)

        self._validate_config()

        self.hist_stat_len = self._compute_n_windows(self.seq_len)
        self.pred_stat_len = self._compute_n_windows(self.pred_len)
        raw_hist_len = self.hist_stat_len * self.window_len

        self.predictor = _SANBasePredictor(
            hist_stat_len=self.hist_stat_len,
            raw_hist_len=raw_hist_len,
            pred_stat_len=self.pred_stat_len,
            enc_in=enc_in,
            sigma_min=sigma_min,
        )

        # lp_state uses a dedicated predictor; other states use generic RouteStatePredictor
        if route_path == "lp_state_correction":
            self.lp_state_predictor = _LPStatePredictor(self.seq_len, self.pred_len)
            self.route_state_predictor = None
            self.route_path_impl = build_route_path(
                route_path, pred_len=self.pred_len, enc_in=enc_in, sigma_min=sigma_min
            )
            self.route_state_impl = LPState(period_len=self.window_len)
        elif route_path != "none":
            self.lp_state_predictor = None
            self.route_state_predictor = RouteStatePredictor(
                hist_stat_len=self.hist_stat_len,
                raw_hist_len=raw_hist_len,
                pred_stat_len=self.pred_stat_len,
                enc_in=enc_in,
            )
            self.route_path_impl = build_route_path(
                route_path, pred_len=self.pred_len, enc_in=enc_in, sigma_min=sigma_min
            )
            self.route_state_impl = build_route_state(route_state)
        else:
            self.lp_state_predictor = None
            self.route_state_predictor = None
            self.route_path_impl = None
            self.route_state_impl = None

        self._reset_cache()

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if self.route_path not in _VALID_PATHS:
            raise ValueError(
                f"route_path must be one of {_VALID_PATHS}, got '{self.route_path}'."
            )
        if self.route_state not in _VALID_STATES:
            raise ValueError(
                f"route_state must be one of {_VALID_STATES}, got '{self.route_state}'."
            )
        if self.route_path == "none" and self.route_state != "none":
            raise ValueError("When route_path='none', route_state must also be 'none'.")
        if self.route_path != "none" and self.route_state == "none":
            raise ValueError(
                f"When route_path='{self.route_path}', route_state must not be 'none'."
            )
        # lp_state and lp_state_correction must always be paired together
        if self.route_path == "lp_state_correction" and self.route_state != "lp_state":
            raise ValueError(
                "route_path='lp_state_correction' requires route_state='lp_state'."
            )
        if self.route_state == "lp_state" and self.route_path != "lp_state_correction":
            raise ValueError(
                f"route_state='lp_state' requires route_path='lp_state_correction',"
                f" got route_path='{self.route_path}'."
            )
        if self.seq_len < self.window_len:
            raise ValueError(f"seq_len={self.seq_len} < window_len={self.window_len}.")
        if self.pred_len < self.window_len:
            raise ValueError(f"pred_len={self.pred_len} < window_len={self.window_len}.")
        if (self.seq_len - self.window_len) % self.stride != 0:
            raise ValueError(
                f"(seq_len - window_len) % stride != 0: "
                f"seq_len={self.seq_len}, window_len={self.window_len}, stride={self.stride}."
            )
        if (self.pred_len - self.window_len) % self.stride != 0:
            raise ValueError(
                f"(pred_len - window_len) % stride != 0: "
                f"pred_len={self.pred_len}, window_len={self.window_len}, stride={self.stride}."
            )

    # ------------------------------------------------------------------
    # Window utilities
    # ------------------------------------------------------------------

    def _compute_n_windows(self, length: int) -> int:
        return (length - self.window_len) // self.stride + 1

    def _extract_windows(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, N, window_len, C)"""
        windows = x.unfold(dimension=1, size=self.window_len, step=self.stride)
        return windows.permute(0, 1, 3, 2).contiguous()

    def _compute_window_stats(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """windows: (B, N, window_len, C) -> (mean, std) each (B, N, C)"""
        mean = windows.mean(dim=2)
        std = windows.std(dim=2).clamp(min=self.sigma_min)
        return mean, std

    def _window_stats_to_time_stats(
        self, mean: torch.Tensor, std: torch.Tensor, total_length: int
    ) -> torch.Tensor:
        """Overlap-add window stats to per-timestep stats.
        Returns (B, total_length, 2*C) = cat([mu_t, sigma_t], dim=-1).
        """
        B, N, C = mean.shape
        mu_t = torch.zeros(B, total_length, C, device=mean.device, dtype=mean.dtype)
        e2_t = torch.zeros_like(mu_t)
        counts = torch.zeros_like(mu_t)
        e2 = std.pow(2) + mean.pow(2)
        for i in range(N):
            s = i * self.stride
            e = s + self.window_len
            mu_t[:, s:e, :] += mean[:, i : i + 1, :]
            e2_t[:, s:e, :] += e2[:, i : i + 1, :]
            counts[:, s:e, :] += 1.0
        mu_t = mu_t / counts.clamp_min(1.0)
        e2_t = e2_t / counts.clamp_min(1.0)
        sigma_t = torch.sqrt(torch.clamp(e2_t - mu_t.pow(2), min=self.sigma_min ** 2))
        return torch.cat([mu_t, sigma_t], dim=-1)

    def _patch_features_to_time(self, feat: torch.Tensor, total_length: int) -> torch.Tensor:
        """Overlap-add patch features (B, P, C) to per-timestep (B, total_length, C)."""
        B, P, C = feat.shape
        out = torch.zeros(B, total_length, C, device=feat.device, dtype=feat.dtype)
        counts = torch.zeros(B, total_length, C, device=feat.device, dtype=feat.dtype)
        for i in range(P):
            s = i * self.stride
            e = s + self.window_len
            out[:, s:e, :] += feat[:, i : i + 1, :]
            counts[:, s:e, :] += 1.0
        return out / counts.clamp_min(1.0)

    def _split_stats(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return stats[:, :, : self.channels], stats[:, :, self.channels :]

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _reset_cache(self) -> None:
        self._mu_hist: Optional[torch.Tensor] = None
        self._std_hist: Optional[torch.Tensor] = None
        # Base predictor outputs (patch-level)
        self._base_mu_fut_hat: Optional[torch.Tensor] = None
        self._base_std_fut_hat: Optional[torch.Tensor] = None
        # Base per-timestep stats (time-domain)
        self._pred_time_stats: Optional[torch.Tensor] = None
        self._base_time_mean: Optional[torch.Tensor] = None
        self._base_time_std: Optional[torch.Tensor] = None
        # Generic route state features (nu / dlogsigma / omega_spec)
        self._route_hist_state: Optional[torch.Tensor] = None
        self._route_future_state_hat: Optional[torch.Tensor] = None
        self._route_future_state_time: Optional[torch.Tensor] = None
        self._route_future_oracle_state: Optional[torch.Tensor] = None
        self._route_hist_state_raw: Optional[torch.Tensor] = None
        self._route_future_state_hat_raw: Optional[torch.Tensor] = None
        # lp_state specific caches
        self._hist_lowpass: Optional[torch.Tensor] = None
        self._pred_lowpass_fut: Optional[torch.Tensor] = None
        self._oracle_lowpass_fut: Optional[torch.Tensor] = None
        # Residual-domain future patch-level stats (what base predictor learns)
        self._res_mu_fut_hat: Optional[torch.Tensor] = None    # (B, P_fut, C)
        self._res_std_fut_hat: Optional[torch.Tensor] = None   # (B, P_fut, C)
        # Residual-domain future time-level stats (broadcast for denorm)
        self._res_mu_time: Optional[torch.Tensor] = None       # (B, pred_len, C)
        self._res_sigma_time: Optional[torch.Tensor] = None    # (B, pred_len, C)
        # Cached names
        self._route_path_name: str = self.route_path
        self._route_state_name: str = self.route_state
        # Scalar loss trackers
        self._last_mu_loss: float = 0.0
        self._last_std_loss: float = 0.0
        self._last_base_aux_loss: float = 0.0
        self._last_route_state_loss: float = 0.0
        self._last_lp_aux_loss: float = 0.0
        self._last_aux_total: float = 0.0

    # ------------------------------------------------------------------
    # Stage-aware parameter group accessors
    # ------------------------------------------------------------------

    def parameters_base_predictor(self) -> list:
        """Return parameter list for the base predictor (Stage 1 optimizer target)."""
        return list(self.predictor.parameters())

    def parameters_route_modules(self) -> list:
        """Return parameter list for route modules (Stage 3 optimizer target).

        For lp_state_correction: includes lp_state_predictor + route_path_impl params.
        For all other paths: includes route_state_predictor + route_path_impl params.
        Returns an empty list when route_path='none'.
        """
        params: list = []
        if self.lp_state_predictor is not None:
            params.extend(self.lp_state_predictor.parameters())
        if self.route_state_predictor is not None:
            params.extend(self.route_state_predictor.parameters())
        if self.route_path_impl is not None:
            params.extend(self.route_path_impl.parameters())
        return params

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_base_predictor(self) -> None:
        for p in self.predictor.parameters():
            p.requires_grad_(False)

    def unfreeze_base_predictor(self) -> None:
        for p in self.predictor.parameters():
            p.requires_grad_(True)

    def freeze_route_modules(self) -> None:
        """Freeze all route modules (lp_state_predictor, route_state_predictor, route_path_impl)."""
        if self.lp_state_predictor is not None:
            for p in self.lp_state_predictor.parameters():
                p.requires_grad_(False)
        if self.route_state_predictor is not None:
            for p in self.route_state_predictor.parameters():
                p.requires_grad_(False)
        if self.route_path_impl is not None:
            for p in self.route_path_impl.parameters():
                p.requires_grad_(False)

    def unfreeze_route_modules(self) -> None:
        """Unfreeze all route modules."""
        if self.lp_state_predictor is not None:
            for p in self.lp_state_predictor.parameters():
                p.requires_grad_(True)
        if self.route_state_predictor is not None:
            for p in self.route_state_predictor.parameters():
                p.requires_grad_(True)
        if self.route_path_impl is not None:
            for p in self.route_path_impl.parameters():
                p.requires_grad_(True)

    def _should_normalize_route_state(self) -> bool:
        """Return True if framework _normalize_patch_state() should be applied.

        omega_spec and lp_state are exempt (both keep their raw physical meaning).
        """
        return self.route_state not in ("omega_spec", "lp_state")

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input and prepare all caches.

        lp_state_correction special-case:
          - Computes hist_lowpass via centered_moving_average.
          - SAN stats (mu_hist, std_hist, z) computed on hist_residual = x - hist_lowpass.
          - base predictor learns residual-domain future mu/std.
          - lp_state_predictor(x_hist, hist_lowpass) → pred_lowpass_fut.
        Generic route paths: unchanged base SAN + RouteStatePredictor.
        """
        self._reset_cache()
        B, T, C = x.shape
        if T != self.seq_len:
            raise ValueError(
                f"SANRouteNorm expected seq_len={self.seq_len}, got {T}."
            )

        # --- lp_state special case: residual-domain SAN ---
        if self.route_path == "lp_state_correction":
            hist_lowpass = _centered_moving_average(x.detach(), self.route_state_impl._period_len)
            self._hist_lowpass = hist_lowpass
            # Residual = raw - lowpass; SAN operates on residual
            x_input = x - hist_lowpass
        else:
            x_input = x

        # --- Base SAN: history stats (on residual for lp, raw for others) ---
        hist_windows = self._extract_windows(x_input.detach())
        mu_hist, std_hist = self._compute_window_stats(hist_windows)
        self._mu_hist = mu_hist
        self._std_hist = std_hist

        # --- Per-timestep normalization ---
        hist_time_stats = self._window_stats_to_time_stats(mu_hist, std_hist, T)
        hist_time_mean, hist_time_std = self._split_stats(hist_time_stats)
        z = (x_input - hist_time_mean) / (hist_time_std + self.epsilon)

        # --- xbar: flattened normalized windows ---
        norm_windows = self._extract_windows(z.detach())
        xbar = norm_windows.reshape(B, -1, C)

        # --- Base predictor ---
        anchor = mu_hist.mean(dim=1, keepdim=True)
        mu_base_fut, std_base_fut = self.predictor.predict(
            mu_hist, std_hist, xbar, anchor
        )
        self._base_mu_fut_hat = mu_base_fut
        self._base_std_fut_hat = std_base_fut

        self._pred_time_stats = self._window_stats_to_time_stats(
            mu_base_fut, std_base_fut, self.pred_len
        )
        base_time_mean, base_time_std = self._split_stats(self._pred_time_stats)
        self._base_time_mean = base_time_mean
        self._base_time_std = base_time_std

        # --- Route state (if active) ---
        if self.route_path == "lp_state_correction":
            # lp_state_predictor predicts future low-pass sequence
            pred_lowpass_fut = self.lp_state_predictor(x.detach(), hist_lowpass)
            self._pred_lowpass_fut = pred_lowpass_fut
            # Cache residual-domain future stats for denormalize()
            self._res_mu_fut_hat = mu_base_fut    # already residual-domain
            self._res_std_fut_hat = std_base_fut  # already residual-domain
            self._res_mu_time = base_time_mean
            self._res_sigma_time = base_time_std

        elif self.route_path != "none":
            # Generic route state: RouteStatePredictor + optional _normalize_patch_state
            should_normalize = self._should_normalize_route_state()
            hist_state_raw = self.route_state_impl.extract_hist_state(
                x.detach(), hist_windows, mu_hist, std_hist, self.sigma_min
            )
            self._route_hist_state_raw = hist_state_raw
            hist_state = (
                _normalize_patch_state(hist_state_raw) if should_normalize
                else hist_state_raw
            )
            self._route_hist_state = hist_state

            future_state_hat_raw_pred = self.route_state_predictor(hist_state, xbar)
            future_state_hat_adapted = self.route_state_impl.adapt_future_state(
                future_state_hat_raw_pred
            )
            self._route_future_state_hat_raw = future_state_hat_adapted
            future_state_hat = (
                _normalize_patch_state(future_state_hat_adapted) if should_normalize
                else future_state_hat_adapted
            )
            self._route_future_state_hat = future_state_hat
            self._route_future_state_time = self._patch_features_to_time(
                future_state_hat, self.pred_len
            )

        return z

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Restore y_norm to original scale, applying the route path if active.

        lp_state_correction (structured mean-state denorm):
            y_out = pred_lowpass_fut + mu_res_time + sigma_res_time * y_norm
            where mu_res_time / sigma_res_time are residual-domain future stats.
        All other paths: y_base = base_time_mean + (base_time_std + eps) * y_norm,
            then route_path_impl transforms y_base.
        """
        if self._pred_time_stats is None:
            return y_norm

        T = y_norm.shape[1]

        # lp_state_correction: structured mean-state denorm (no gate, no correction)
        if (
            self.route_path == "lp_state_correction"
            and self._pred_lowpass_fut is not None
            and self._res_mu_time is not None
            and self._res_sigma_time is not None
            and T == self.pred_len
        ):
            return (
                self._pred_lowpass_fut
                + self._res_mu_time
                + (self._res_sigma_time + self.epsilon) * y_norm
            ).contiguous()

        # Standard base denorm
        if T == self._pred_time_stats.shape[1]:
            time_stats = self._pred_time_stats
        else:
            time_stats = self._window_stats_to_time_stats(
                self._base_mu_fut_hat, self._base_std_fut_hat, T
            )
        mean, std = self._split_stats(time_stats)
        y_base = mean + (std + self.epsilon) * y_norm

        # Generic route path transformation
        if (
            self.route_path not in ("none", "lp_state_correction")
            and self._route_future_state_hat is not None
            and self._route_future_state_time is not None
            and self._base_time_mean is not None
            and T == self.pred_len
        ):
            y_base = self.route_path_impl(
                y_norm=y_norm,
                y_base=y_base,
                future_state_hat=self._route_future_state_hat,
                future_state_time=self._route_future_state_time,
                mu_base_fut=self._base_mu_fut_hat,
                std_base_fut=self._base_std_fut_hat,
                base_time_mean=self._base_time_mean,
                base_time_std=self._base_time_std,
            )

        return y_base

    # ------------------------------------------------------------------
    # Loss computation (split by stage role)
    # ------------------------------------------------------------------

    def _compute_oracle_stats(
        self, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        true_windows = self._extract_windows(y_true)
        oracle_mu, oracle_std = self._compute_window_stats(true_windows)
        return true_windows, oracle_mu, oracle_std

    def compute_base_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Base SAN stats loss — Stage 1 uses this.

        lp_state_correction: oracle stats are computed on residual-domain y_true
        so the predictor learns residual-domain future mu/std.
        """
        if self._base_mu_fut_hat is None or self._base_std_fut_hat is None:
            return torch.tensor(0.0, device=y_true.device)

        if self.route_path == "lp_state_correction" and self._oracle_lowpass_fut is None:
            # Compute oracle lowpass for residual oracle (detached, no grad needed here)
            oracle_lowpass = _centered_moving_average(
                y_true.detach(), self.route_state_impl._period_len
            )
            y_oracle_input = y_true - oracle_lowpass
        elif self.route_path == "lp_state_correction" and self._oracle_lowpass_fut is not None:
            y_oracle_input = y_true - self._oracle_lowpass_fut
        else:
            y_oracle_input = y_true

        _, oracle_mu, oracle_std = self._compute_oracle_stats(y_oracle_input)
        mu_loss = F.mse_loss(self._base_mu_fut_hat, oracle_mu)
        std_loss = F.mse_loss(self._base_std_fut_hat, oracle_std)
        base_aux = self.w_mu * mu_loss + self.w_std * std_loss

        self._last_mu_loss = float(mu_loss.detach().item())
        self._last_std_loss = float(std_loss.detach().item())
        self._last_base_aux_loss = float(base_aux.detach().item())
        self._last_aux_total = self._last_base_aux_loss
        return base_aux

    def compute_route_state_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Route state prediction loss — Stage 3 uses this.

        lp_state: MSE(pred_lowpass_fut, oracle_lowpass_fut) over the full future sequence.
        Others:   MSE(future_state_hat, oracle_state) at patch level.
        Returns 0 when route_path='none'.
        """
        if self.route_path == "none":
            return torch.tensor(0.0, device=y_true.device)

        # lp_state special case
        if self.route_state == "lp_state":
            if self._pred_lowpass_fut is None:
                return torch.tensor(0.0, device=y_true.device)
            oracle_lowpass_fut = _centered_moving_average(y_true, self.route_state_impl._period_len)
            self._oracle_lowpass_fut = oracle_lowpass_fut
            lp_aux_loss = F.mse_loss(self._pred_lowpass_fut, oracle_lowpass_fut)
            self._last_route_state_loss = float(lp_aux_loss.detach().item())
            self._last_lp_aux_loss = float(lp_aux_loss.detach().item())
            # Push diagnostics to route_path_impl
            if self.route_path_impl is not None and hasattr(
                self.route_path_impl, "update_diagnostics"
            ):
                self.route_path_impl.update_diagnostics(self._pred_lowpass_fut, oracle_lowpass_fut)
            return lp_aux_loss

        # Generic route state
        if self._route_future_state_hat is None:
            return torch.tensor(0.0, device=y_true.device)

        true_windows, oracle_mu, oracle_std = self._compute_oracle_stats(y_true)
        oracle_raw = self.route_state_impl.build_future_oracle_state(
            y_true, true_windows, oracle_mu, oracle_std, self.sigma_min
        )
        should_normalize = self._should_normalize_route_state()
        oracle_state = (
            _normalize_patch_state(oracle_raw) if should_normalize else oracle_raw
        )
        self._route_future_oracle_state = oracle_state

        route_state_loss = F.mse_loss(self._route_future_state_hat, oracle_state)
        self._last_route_state_loss = float(route_state_loss.detach().item())
        return route_state_loss

    def compute_total_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Combined base aux + weighted route state loss.

        For lp_state_correction: compute route_state_loss first so that
        _oracle_lowpass_fut is cached before compute_base_aux_loss uses it.
        """
        if self.route_path == "lp_state_correction":
            route_loss = self.compute_route_state_loss(y_true)
            base_aux = self.compute_base_aux_loss(y_true)
        else:
            base_aux = self.compute_base_aux_loss(y_true)
            route_loss = self.compute_route_state_loss(y_true)
        total = base_aux + self.route_state_loss_scale * route_loss
        self._last_aux_total = float(total.detach().item())
        return total

    def loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Backward-compat alias for compute_total_aux_loss."""
        return self.compute_total_aux_loss(y_true)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_last_aux_stats(self) -> dict:
        """Scalar stats from most recent loss computation.
        base_aux and route_state_loss are kept separate.
        """
        return {
            "aux_total": self._last_aux_total,
            "base_aux_loss": self._last_base_aux_loss,
            "mu_loss": self._last_mu_loss,
            "std_loss": self._last_std_loss,
            "route_state_loss": self._last_route_state_loss,
            "lp_aux_loss": self._last_lp_aux_loss,
            "mu_hist_mean": (
                float(self._mu_hist.mean().item()) if self._mu_hist is not None else 0.0
            ),
            "std_hist_mean": (
                float(self._std_hist.mean().item()) if self._std_hist is not None else 0.0
            ),
            "base_mu_fut_mean": (
                float(self._base_mu_fut_hat.mean().item())
                if self._base_mu_fut_hat is not None else 0.0
            ),
            "base_std_fut_mean": (
                float(self._base_std_fut_hat.mean().item())
                if self._base_std_fut_hat is not None else 0.0
            ),
        }

    def get_route_diagnostics(self) -> dict:
        """Cached diagnostics from the route path.
        Empty dict when route_path='none'.
        Aux stats are NOT mixed in here — use get_last_aux_stats() for those.
        """
        if self.route_path == "none" or self.route_path_impl is None:
            return {}
        diag: dict = {
            "route_path": self._route_path_name,
            "route_state": self._route_state_name,
        }
        if self.route_path == "lp_state_correction":
            # lp-specific diagnostics
            diag["lp_aux_loss"] = self._last_lp_aux_loss
            diag["hist_lowpass_mean"] = (
                float(self._hist_lowpass.mean().item())
                if self._hist_lowpass is not None else 0.0
            )
            diag["pred_lowpass_mean"] = (
                float(self._pred_lowpass_fut.mean().item())
                if self._pred_lowpass_fut is not None else 0.0
            )
            diag["oracle_lowpass_mean"] = (
                float(self._oracle_lowpass_fut.mean().item())
                if self._oracle_lowpass_fut is not None else 0.0
            )
            if self._pred_lowpass_fut is not None and self._oracle_lowpass_fut is not None:
                diag["lp_abs_error"] = float(
                    (self._pred_lowpass_fut - self._oracle_lowpass_fut).abs().mean().item()
                )
            else:
                diag["lp_abs_error"] = 0.0
            diag["res_mu_mean"] = (
                float(self._res_mu_fut_hat.mean().item())
                if self._res_mu_fut_hat is not None else 0.0
            )
            diag["res_sigma_mean"] = (
                float(self._res_std_fut_hat.mean().item())
                if self._res_std_fut_hat is not None else 0.0
            )
            # Also include route_path_impl diagnostics (lp_pred_mean, etc.)
            if hasattr(self.route_path_impl, "get_route_diagnostics"):
                diag.update(self.route_path_impl.get_route_diagnostics())
        elif hasattr(self.route_path_impl, "get_route_diagnostics"):
            diag.update(self.route_path_impl.get_route_diagnostics())
        return diag

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "n",
        station_pred=None,
    ) -> torch.Tensor:
        if mode == "n":
            return self.normalize(x)
        if mode == "d":
            return self.denormalize(x)
        return x


# ---------------------------------------------------------------------------
# Internal MLP and predictor classes
# ---------------------------------------------------------------------------


class _SANBaseMLP(nn.Module):
    """Lightweight MLP for future-stats prediction.

    mode='mu':    anchor + residual with Tanh + learnable blend weight.
    mode='sigma': direct sigma prediction with GELU + Softplus.
    """

    def __init__(
        self,
        hist_stat_len: int,
        raw_hist_len: int,
        pred_stat_len: int,
        enc_in: int,
        mode: str,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.pred_stat_len = pred_stat_len
        self.mode = mode
        self.stats_input = nn.Linear(hist_stat_len, hidden_dim)
        self.raw_input = nn.Linear(raw_hist_len, hidden_dim)
        self.output = nn.Linear(2 * hidden_dim, pred_stat_len)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        if mode == "mu":
            self.activation = nn.Tanh()
            self.final_activation = nn.Identity()
            self.weight = nn.Parameter(torch.ones(2, enc_in))
        else:
            self.activation = nn.GELU()
            self.final_activation = nn.Softplus()
            self.weight = None

    def forward(
        self,
        x_stats: torch.Tensor,
        x_raw: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xs = x_stats.permute(0, 2, 1)
        xr = x_raw.permute(0, 2, 1)
        feat = torch.cat([self.stats_input(xs), self.raw_input(xr)], dim=-1)
        pred = self.final_activation(
            self.output(self.activation(feat))
        ).permute(0, 2, 1)

        if self.mode == "mu" and anchor is not None:
            anch = anchor.expand(-1, self.pred_stat_len, -1)
            pred = (
                pred * self.weight[0].view(1, 1, -1)
                + anch * self.weight[1].view(1, 1, -1)
            )
        return pred


class _SANBasePredictor(nn.Module):
    """Predicts future (mean, std) from historical window stats and normalized input."""

    def __init__(
        self,
        hist_stat_len: int,
        raw_hist_len: int,
        pred_stat_len: int,
        enc_in: int,
        sigma_min: float = 1e-3,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.mean_head = _SANBaseMLP(
            hist_stat_len=hist_stat_len,
            raw_hist_len=raw_hist_len,
            pred_stat_len=pred_stat_len,
            enc_in=enc_in,
            mode="mu",
            hidden_dim=hidden_dim,
        )
        self.std_head = _SANBaseMLP(
            hist_stat_len=hist_stat_len,
            raw_hist_len=raw_hist_len,
            pred_stat_len=pred_stat_len,
            enc_in=enc_in,
            mode="sigma",
            hidden_dim=hidden_dim,
        )

    def predict(
        self,
        mu_hist: torch.Tensor,
        std_hist: torch.Tensor,
        xbar: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_fut = self.mean_head(
            mu_hist - anchor.expand(-1, mu_hist.shape[1], -1), xbar, anchor
        )
        std_fut = self.std_head(std_hist, xbar, None).clamp(min=self.sigma_min)
        return mu_fut, std_fut


class RouteStatePredictor(nn.Module):
    """Shared predictor for future route state — state-agnostic.

    Input:
        hist_state: (B, P_hist, C)  — normalised by framework if applicable
        xbar:       (B, P_hist * window_len, C)
    Output:
        future_state_hat_raw: (B, P_fut, C)
    """

    def __init__(
        self,
        hist_stat_len: int,
        raw_hist_len: int,
        pred_stat_len: int,
        enc_in: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.pred_stat_len = pred_stat_len
        self.state_input = nn.Linear(hist_stat_len, hidden_dim)
        self.raw_input = nn.Linear(raw_hist_len, hidden_dim)
        self.activation = nn.Tanh()
        self.output = nn.Linear(2 * hidden_dim, pred_stat_len)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        hist_state: torch.Tensor,
        xbar: torch.Tensor,
    ) -> torch.Tensor:
        hs = hist_state.permute(0, 2, 1)
        xr = xbar.permute(0, 2, 1)
        feat = torch.cat([self.state_input(hs), self.raw_input(xr)], dim=-1)
        out = self.output(self.activation(feat))
        return out.permute(0, 2, 1)
