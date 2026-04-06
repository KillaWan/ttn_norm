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

lp_state + lp_state_correction — "Base SAN + predicted future low-pass patch mean":
  This combination is a minimal extension of pure SAN.  It does NOT add a second
  forecasting branch and does NOT implement residual-domain SAN.

  The lp slow state is defined by:
    mu_lp = patch_mean(lowpass_time(raw_series))
  i.e. first apply a fixed time-domain lowpass filter to the raw time series,
  then extract patch means.  This guarantees that history and oracle share
  the identical operator.

  normalize() semantics:
    1. Raw SAN: mu_hist, std_hist computed on raw x (not residual).
    2. z = standard SAN-normalised input.
    3. base predictor predicts raw future patch std (std branch) and mean (unused in denorm).
    4. mu_hist_lp = patch_mean(lowpass_time(x_hist))   <- lp slow state
       lp_state_predictor(mu_hist_lp) -> lp_mu_fut_hat (B, P_fut, C).
       Trained to match oracle future lp slow state.

  denormalize():
    lp_time_stats = _window_stats_to_time_stats(lp_mu_fut_hat, base_std_fut_hat, pred_len)
    y_hat = mu_lp_time + (sigma_base_time + eps) * y_norm
    (standard affine denorm; only the mean source is replaced)

  compute_route_state_loss():
    oracle_lp_mu_fut = patch_mean(lowpass_time(y_true))   [same operator as history side]
    loss = MSE(lp_mu_fut_hat, oracle_lp_mu_fut)

Stage roles for lp combo (enforced by train_state_routes.py):
  Stage 1:     nm.predictor only — base std aux loss (mu loss inactive).
  lp_pretrain: nm.lp_state_predictor only — route_state_loss (patch-level lp mean MSE).
  Stage 2:     fm only — task loss; nm.predictor + lp_state_predictor both frozen.
  joint/stage3: forbidden for lp combo.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .san_route_paths import build_route_path
from .san_route_states import build_route_state

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
    """Channel-wise MLP predicting future low-pass patch mean.

    Input:  mu_hist_raw   (B, P_hist, C) — raw historical patch means
    Output: mu_lp_fut_hat (B, P_fut, C)  — predicted future low-pass patch mean

    Architecture: Linear(P_hist → hidden) + ReLU + Linear(hidden → P_fut)
    Parameters are shared across channels (applied on permuted (B, C, P) input).
    """

    def __init__(self, hist_stat_len: int, pred_stat_len: int, hidden: int = 128):
        super().__init__()
        self.enc = nn.Linear(hist_stat_len, hidden)
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden, pred_stat_len)

    def forward(self, mu_hist: torch.Tensor) -> torch.Tensor:
        # mu_hist: (B, P, C) -> permute to (B, C, P) for shared per-channel linear
        x = mu_hist.permute(0, 2, 1)          # (B, C, P)
        out = self.out(self.act(self.enc(x))) # (B, C, P_fut)
        return out.permute(0, 2, 1)           # (B, P_fut, C)


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
            # lp combo: future mean source replacement only — no route_path_impl
            self.lp_state_predictor = _LPStatePredictor(
                hist_stat_len=self.hist_stat_len,
                pred_stat_len=self.pred_stat_len,
            )
            self.route_state_predictor = None
            self.route_path_impl = None
            self.route_state_impl = None
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

    def _time_to_patch_mean(self, x_time: torch.Tensor) -> torch.Tensor:
        """Extract patch means from a time-domain tensor.
        x_time: (B, T, C) → (B, P, C) where P = _compute_n_windows(T).
        """
        wins = self._extract_windows(x_time)
        mu, _ = self._compute_window_stats(wins)
        return mu

    def _patch_values_to_time(self, feat: torch.Tensor, total_length: int) -> torch.Tensor:
        """Overlap-add patch-level values (B, P, C) to time domain (B, total_length, C).

        Two-step mean-preserving lift:
          1. Linear interpolation as smooth baseline.
          2. Exact patch-mean correction.
        """
        x = feat.permute(0, 2, 1)                                             # (B, C, P)
        c0 = F.interpolate(x, size=total_length, mode="linear", align_corners=True)
        c0 = c0.permute(0, 2, 1)                                              # (B, T, C)
        m0 = self._time_to_patch_mean(c0)                                     # (B, P, C)
        res = feat - m0
        return c0 + self._patch_features_to_time(res, total_length)

    def _lowpass_time_series(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fixed [1, 2, 1]/4 FIR lowpass along the time axis.

        x:      (B, T, C)
        Returns (B, T, C) smoothed along the T dimension.
        T == 1: returns x unchanged.
        Kernel [1, 2, 1] / 4; reflect padding; no learnable parameters.
        Does NOT mix channels.
        """
        B, T, C = x.shape
        if T == 1:
            return x
        k = x.new_tensor([1.0, 2.0, 1.0]) / 4.0         # (3,)
        xt = x.permute(0, 2, 1).reshape(B * C, 1, T)    # (B*C, 1, T)
        xt_padded = F.pad(xt, (1, 1), mode="reflect")    # (B*C, 1, T+2)
        w = k.view(1, 1, 3)                               # (1, 1, 3)
        out = F.conv1d(xt_padded, w, padding=0)          # (B*C, 1, T)
        return out.reshape(B, C, T).permute(0, 2, 1)     # (B, T, C)

    def _lowpass_time_to_patch_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Compute lp slow state: patch_mean(lowpass_time(x)).

        x:      (B, T, C) — raw time series (hist or future)
        Returns (B, P, C) where P = _compute_n_windows(T).
        Uses current window_len and stride for patch extraction.
        This is the sole lp slow-state extractor for the lp combo.
        """
        lp = self._lowpass_time_series(x)
        wins = self._extract_windows(lp)
        mu, _ = self._compute_window_stats(wins)
        return mu

    def _lowpass_patch_mean(self, feat: torch.Tensor) -> torch.Tensor:
        """[Retained for non-lp use; not used by lp combo.]

        Apply fixed [1, 2, 1]/4 FIR lowpass along the patch axis.
        feat:   (B, P, C) — patch-level feature.
        Returns (B, P, C) smoothed along the P dimension.
        """
        B, P, C = feat.shape
        if P == 1:
            return feat
        k = feat.new_tensor([1.0, 2.0, 1.0]) / 4.0         # (3,)
        xp = feat.permute(0, 2, 1).reshape(B * C, 1, P)    # (B*C, 1, P)
        xp_padded = F.pad(xp, (1, 1), mode="reflect")       # (B*C, 1, P+2)
        w = k.view(1, 1, 3)                                  # (1, 1, 3)
        out = F.conv1d(xp_padded, w, padding=0)             # (B*C, 1, P)
        return out.reshape(B, C, P).permute(0, 2, 1)        # (B, P, C)

    # ------------------------------------------------------------------
    # Stage-aware parameter group accessors
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
        # lp_state specific caches (patch-level)
        self._lp_mu_hist: Optional[torch.Tensor] = None      # (B, P_hist, C) mu_hist_lp
        self._lp_mu_fut_hat: Optional[torch.Tensor] = None   # (B, P_fut, C)  predicted lp mean
        self._oracle_lp_mu_fut: Optional[torch.Tensor] = None  # (B, P_fut, C) oracle lp mean
        # Cached names
        self._route_path_name: str = self.route_path
        self._route_state_name: str = self.route_state
        # Scalar loss trackers
        self._last_mu_loss: float = 0.0
        self._last_std_loss: float = 0.0
        self._last_base_aux_loss: float = 0.0
        self._last_route_state_loss: float = 0.0
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

        lp_state_correction: raw SAN stats on x (not residual), then
          lp_state_predictor(mu_hist_raw) → lp_mu_fut_hat (B, P_fut, C).
        Generic route paths: unchanged base SAN + RouteStatePredictor.
        """
        self._reset_cache()
        B, T, C = x.shape
        if T != self.seq_len:
            raise ValueError(
                f"SANRouteNorm expected seq_len={self.seq_len}, got {T}."
            )

        # --- Base SAN: history stats on raw input ---
        x_input = x
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
            # lp combo: mu_lp = patch_mean(lowpass_time(x_hist))
            mu_hist_lp = self._lowpass_time_to_patch_mean(x.detach())
            self._lp_mu_hist = mu_hist_lp
            lp_mu_fut_hat = self.lp_state_predictor(mu_hist_lp)
            self._lp_mu_fut_hat = lp_mu_fut_hat

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

    def denormalize(
        self,
        y_norm: torch.Tensor,
        y_true_oracle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Restore y_norm to original scale, applying the route path if active.

        lp_state_correction (mean-only replacement):
            Uses predicted future low-pass patch mean as the denorm mean;
            base std for the scale.  Standard affine denorm.
        All other paths: standard affine denorm then optional route_path_impl.
        """
        if self._pred_time_stats is None:
            return y_norm

        T = y_norm.shape[1]

        # lp_state_correction: future mean from lp predictor, std from base predictor
        if (
            self.route_path == "lp_state_correction"
            and self._lp_mu_fut_hat is not None
            and self._base_std_fut_hat is not None
            and T == self.pred_len
        ):
            lp_time_stats = self._window_stats_to_time_stats(
                self._lp_mu_fut_hat, self._base_std_fut_hat, T
            )
            mu_lp_time, sigma_base_time = self._split_stats(lp_time_stats)
            return (mu_lp_time + (sigma_base_time + self.epsilon) * y_norm).contiguous()

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

        lp_state_correction: oracle stats from raw y_true; only std_loss active;
          mu_loss is inactive (base mean prediction is not used in lp denorm).
        All other paths: standard mu + std loss on raw y_true.
        """
        if self._base_mu_fut_hat is None or self._base_std_fut_hat is None:
            return torch.tensor(0.0, device=y_true.device)

        _, oracle_mu, oracle_std = self._compute_oracle_stats(y_true)

        if self.route_path == "lp_state_correction":
            # Only std loss; mu predictor is not used in lp denorm
            mu_loss = torch.tensor(0.0, device=y_true.device)
            std_loss = F.mse_loss(self._base_std_fut_hat, oracle_std)
            base_aux = self.w_std * std_loss
        else:
            mu_loss = F.mse_loss(self._base_mu_fut_hat, oracle_mu)
            std_loss = F.mse_loss(self._base_std_fut_hat, oracle_std)
            base_aux = self.w_mu * mu_loss + self.w_std * std_loss

        self._last_mu_loss = float(mu_loss.detach().item())
        self._last_std_loss = float(std_loss.detach().item())
        self._last_base_aux_loss = float(base_aux.detach().item())
        self._last_aux_total = self._last_base_aux_loss
        return base_aux

    def compute_route_state_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Route state prediction loss — lp_pretrain / Stage 3 uses this.

        lp_state: patch-level MSE(lp_mu_fut_hat, oracle_lp_mu_fut).
                  oracle_lp_mu_fut = patch_mean(lowpass_time(y_true)).
        Others:   MSE(future_state_hat, oracle_state) at patch level.
        Returns 0 when route_path='none'.
        """
        if self.route_path == "none":
            return torch.tensor(0.0, device=y_true.device)

        # lp_state special case: patch-level MSE against oracle slow state
        if self.route_state == "lp_state":
            if self._lp_mu_fut_hat is None:
                return torch.tensor(0.0, device=y_true.device)
            # oracle: same operator as history side — patch_mean(lowpass_time(y_true))
            oracle_lp_mu_fut = self._lowpass_time_to_patch_mean(y_true)
            self._oracle_lp_mu_fut = oracle_lp_mu_fut
            lp_loss = F.mse_loss(self._lp_mu_fut_hat, oracle_lp_mu_fut)
            self._last_route_state_loss = float(lp_loss.detach().item())
            return lp_loss

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
        """Combined base aux + weighted route state loss."""
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
        lp_* fields are non-zero only when route_path='lp_state_correction'.
        lp_mu_hist_mean     : mean of patch_mean(lowpass_time(x_hist))
        oracle_lp_mu_fut_mean: mean of patch_mean(lowpass_time(y_true))
        """
        return {
            "aux_total": self._last_aux_total,
            "base_aux_loss": self._last_base_aux_loss,
            "mu_loss": self._last_mu_loss,
            "std_loss": self._last_std_loss,
            "route_state_loss": self._last_route_state_loss,
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
            "lp_mu_hist_mean": (
                float(self._lp_mu_hist.mean().item()) if self._lp_mu_hist is not None else 0.0
            ),
            "lp_mu_fut_hat_mean": (
                float(self._lp_mu_fut_hat.mean().item())
                if self._lp_mu_fut_hat is not None else 0.0
            ),
            "oracle_lp_mu_fut_mean": (
                float(self._oracle_lp_mu_fut.mean().item())
                if self._oracle_lp_mu_fut is not None else 0.0
            ),
            "lp_mu_abs_error": (
                float(
                    (self._lp_mu_fut_hat - self._oracle_lp_mu_fut).abs().mean().item()
                )
                if self._lp_mu_fut_hat is not None and self._oracle_lp_mu_fut is not None
                else 0.0
            ),
        }

    def get_route_diagnostics(self) -> dict:
        """Cached diagnostics from the route path.
        Empty dict when route_path='none'.
        Aux stats are NOT mixed in here — use get_last_aux_stats() for those.
        lp combo fields:
          lp_mu_hist_mean      : mean of patch_mean(lowpass_time(x_hist))
          lp_mu_fut_hat_mean   : mean of predicted future lp slow state
          oracle_lp_mu_fut_mean: mean of patch_mean(lowpass_time(y_true))
          lp_mu_abs_error      : |lp_mu_fut_hat - oracle_lp_mu_fut|.mean()
        """
        if self.route_path == "none":
            return {}
        diag: dict = {
            "route_path": self._route_path_name,
            "route_state": self._route_state_name,
        }
        if self.route_path == "lp_state_correction":
            # lp-specific diagnostics
            diag["lp_mu_hist_mean"] = (
                float(self._lp_mu_hist.mean().item())
                if self._lp_mu_hist is not None else 0.0
            )
            diag["lp_mu_fut_hat_mean"] = (
                float(self._lp_mu_fut_hat.mean().item())
                if self._lp_mu_fut_hat is not None else 0.0
            )
            diag["oracle_lp_mu_fut_mean"] = (
                float(self._oracle_lp_mu_fut.mean().item())
                if self._oracle_lp_mu_fut is not None else 0.0
            )
            diag["lp_mu_abs_error"] = (
                float(
                    (self._lp_mu_fut_hat - self._oracle_lp_mu_fut).abs().mean().item()
                )
                if self._lp_mu_fut_hat is not None and self._oracle_lp_mu_fut is not None
                else 0.0
            )
            diag["base_std_fut_mean"] = (
                float(self._base_std_fut_hat.mean().item())
                if self._base_std_fut_hat is not None else 0.0
            )
        elif self.route_path_impl is not None and hasattr(
            self.route_path_impl, "get_route_diagnostics"
        ):
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
