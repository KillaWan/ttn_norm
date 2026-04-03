"""SANRouteNorm — base SAN + output-side route framework.

Design principles
-----------------
- This is a normalization-centric framework.  Route modules do NOT modify the
    backbone operator and do NOT modify the base mu/std predictor.
- Route operates only on the output side: inside denormalize(), route_path_impl
    transforms y_base after base denormalization.
- normalize() prepares and caches all state features needed by denormalize() and loss().
- The base predictor (nm.predictor) is always trained in Stage 1.  Route modules
    (nm.route_state_predictor, nm.route_path_impl) are trained in Stage 3 only.
- By default, route-state features are normalised by _normalize_patch_state()
    at the framework layer, so individual state implementations need not standardise.
- omega_spec is the exception: it keeps its raw bounded physical meaning and skips
    framework-level patch z-score normalisation, because omega_spec represents absolute
    spectral concentration / predictability and sample-wise standardisation would
    destroy that control signal.

Stage roles (enforced by train_state_routes.py):
  Stage 1: nm.predictor only — base aux loss.
  Stage 2: backbone only — task loss; nm fully frozen.
  Stage 3: nm.route_state_predictor + nm.route_path_impl — task + route_state_loss.
  Joint finetune (optional): backbone + route modules — task loss; nm.predictor frozen.
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
    "local_value_parameter",   # alias → local_transport
}
_VALID_STATES = {"none", "nu", "dlogsigma", "omega_spec"}


def _normalize_patch_state(feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Normalise patch features per-batch, per-channel across the patch dimension.

    feat: (B, P, C)
    Returns (feat - mean) / std where mean and std are computed over dim=1 (patches).
    This is the default framework-level normalisation applied to route state features,
    ensuring the route_state_predictor always operates on a consistent scale.
    omega_spec is intentionally exempt so its bounded physical semantics are preserved.
    """
    mean = feat.mean(dim=1, keepdim=True)
    std = feat.std(dim=1, keepdim=True).clamp_min(eps)
    return (feat - mean) / std


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
                               | "alignment" | "gating"
                               | "local_value_parameter" (alias → local_transport)
        route_state:           "none" | "nu" | "dlogsigma"
        route_state_loss_scale: Weight for route state prediction loss in
                                compute_total_aux_loss().
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

        if route_path != "none":
            self.route_state_predictor = RouteStatePredictor(
                hist_stat_len=self.hist_stat_len,
                raw_hist_len=raw_hist_len,
                pred_stat_len=self.pred_stat_len,
                enc_in=enc_in,
            )
            self.route_path_impl = build_route_path(
                route_path,
                pred_len=self.pred_len,
                enc_in=enc_in,
                sigma_min=sigma_min,
            )
            self.route_state_impl = build_route_state(route_state)
        else:
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
            raise ValueError(
                "When route_path='none', route_state must also be 'none'."
            )
        if self.route_path != "none" and self.route_state == "none":
            raise ValueError(
                f"When route_path='{self.route_path}', route_state must not be 'none'."
            )
        if self.seq_len < self.window_len:
            raise ValueError(
                f"seq_len={self.seq_len} < window_len={self.window_len}."
            )
        if self.pred_len < self.window_len:
            raise ValueError(
                f"pred_len={self.pred_len} < window_len={self.window_len}."
            )
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
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        total_length: int,
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
        sigma_t = torch.sqrt(
            torch.clamp(e2_t - mu_t.pow(2), min=self.sigma_min ** 2)
        )
        return torch.cat([mu_t, sigma_t], dim=-1)

    def _patch_features_to_time(
        self, feat: torch.Tensor, total_length: int
    ) -> torch.Tensor:
        """Overlap-add patch features to per-timestep features.

        feat: (B, P, C) — one feature vector per patch.
        Returns (B, total_length, C).
        """
        B, P, C = feat.shape
        out = torch.zeros(B, total_length, C, device=feat.device, dtype=feat.dtype)
        counts = torch.zeros(B, total_length, C, device=feat.device, dtype=feat.dtype)
        for i in range(P):
            s = i * self.stride
            e = s + self.window_len
            out[:, s:e, :] += feat[:, i : i + 1, :]
            counts[:, s:e, :] += 1.0
        return out / counts.clamp_min(1.0)

    def _split_stats(
        self, stats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        # Route state features — normalised versions (fed to predictor / loss)
        self._route_hist_state: Optional[torch.Tensor] = None        # normalised
        self._route_future_state_hat: Optional[torch.Tensor] = None  # normalised
        self._route_future_state_time: Optional[torch.Tensor] = None
        self._route_future_oracle_state: Optional[torch.Tensor] = None
        # Raw (pre-normalisation) route state features — for diagnostics only
        self._route_hist_state_raw: Optional[torch.Tensor] = None
        self._route_future_state_hat_raw: Optional[torch.Tensor] = None
        # Cached names (for stats reporting)
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

        Returns an empty list when route_path='none'.
        """
        params: list = []
        if self.route_state_predictor is not None:
            params.extend(self.route_state_predictor.parameters())
        if self.route_path_impl is not None:
            params.extend(self.route_path_impl.parameters())
        return params

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers (used by train_state_routes.py)
    # ------------------------------------------------------------------

    def freeze_base_predictor(self) -> None:
        """Freeze nm.predictor parameters (no gradient in Stage 2 and Stage 3)."""
        for p in self.predictor.parameters():
            p.requires_grad_(False)

    def unfreeze_base_predictor(self) -> None:
        """Unfreeze nm.predictor parameters."""
        for p in self.predictor.parameters():
            p.requires_grad_(True)

    def freeze_route_modules(self) -> None:
        """Freeze route_state_predictor and route_path_impl (safe when route_path='none')."""
        if self.route_state_predictor is not None:
            for p in self.route_state_predictor.parameters():
                p.requires_grad_(False)
        if self.route_path_impl is not None:
            for p in self.route_path_impl.parameters():
                p.requires_grad_(False)

    def unfreeze_route_modules(self) -> None:
        """Unfreeze route_state_predictor and route_path_impl (safe when route_path='none')."""
        if self.route_state_predictor is not None:
            for p in self.route_state_predictor.parameters():
                p.requires_grad_(True)
        if self.route_path_impl is not None:
            for p in self.route_path_impl.parameters():
                p.requires_grad_(True)

    def _should_normalize_route_state(self) -> bool:
        """Return whether framework patch-state normalisation should be applied.

        Default behavior is to normalise route states via _normalize_patch_state().
        omega_spec is excluded so predictor, oracle, and path use its raw bounded
        physical quantity instead of a sample-wise patch z-score.
        """
        return self.route_state != "omega_spec"

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input and prepare all caches needed by denormalize() and loss().

        Flow:
          1. Compute base window stats (mu_hist, std_hist).
          2. Build per-timestep normalization and compute z.
          3. Run base predictor → mu_base_fut, std_base_fut.
          4. Build and cache per-timestep base stats (base_time_mean, base_time_std).
             5. If routing active:
                 - extract hist_state_raw, apply default framework normalisation unless
                    the route state opts out (omega_spec)
                 - predict future_state_hat_raw, adapt, apply the same default/opt-out rule
             - cache raw versions for diagnostics
                 - convert the predictor/path representation to time domain

        mu/std tensors are NEVER modified by the route path here; routing
        happens entirely in denormalize().
        """
        self._reset_cache()
        B, T, C = x.shape
        if T != self.seq_len:
            raise ValueError(
                f"SANRouteNorm expected seq_len={self.seq_len}, got {T}."
            )

        # --- Base SAN: history stats ---
        hist_windows = self._extract_windows(x.detach())      # (B, P_hist, W, C)
        mu_hist, std_hist = self._compute_window_stats(hist_windows)
        self._mu_hist = mu_hist
        self._std_hist = std_hist

        # --- Per-timestep normalization ---
        hist_time_stats = self._window_stats_to_time_stats(mu_hist, std_hist, T)
        hist_time_mean, hist_time_std = self._split_stats(hist_time_stats)
        z = (x - hist_time_mean) / (hist_time_std + self.epsilon)

        # --- xbar: flattened normalized windows ---
        norm_windows = self._extract_windows(z.detach())
        xbar = norm_windows.reshape(B, -1, C)  # (B, P_hist * W, C)

        # --- Base predictor ---
        anchor = mu_hist.mean(dim=1, keepdim=True)  # (B, 1, C)
        mu_base_fut, std_base_fut = self.predictor.predict(
            mu_hist, std_hist, xbar, anchor
        )

        # Cache base outputs
        self._base_mu_fut_hat = mu_base_fut
        self._base_std_fut_hat = std_base_fut

        # Build per-timestep base stats and cache for denormalize()
        self._pred_time_stats = self._window_stats_to_time_stats(
            mu_base_fut, std_base_fut, self.pred_len
        )
        base_time_mean, base_time_std = self._split_stats(self._pred_time_stats)
        self._base_time_mean = base_time_mean
        self._base_time_std = base_time_std

        # --- Route state (if active) ---
        if self.route_path != "none":
            should_normalize_state = self._should_normalize_route_state()
            hist_state_raw = self.route_state_impl.extract_hist_state(
                x.detach(), hist_windows, mu_hist, std_hist, self.sigma_min
            )
            self._route_hist_state_raw = hist_state_raw
            if should_normalize_state:
                hist_state = _normalize_patch_state(hist_state_raw)
            else:
                hist_state = hist_state_raw
            self._route_hist_state = hist_state

            future_state_hat_raw_pred = self.route_state_predictor(hist_state, xbar)
            future_state_hat_adapted = self.route_state_impl.adapt_future_state(
                future_state_hat_raw_pred
            )
            self._route_future_state_hat_raw = future_state_hat_adapted
            if should_normalize_state:
                future_state_hat = _normalize_patch_state(future_state_hat_adapted)
            else:
                future_state_hat = future_state_hat_adapted
            self._route_future_state_hat = future_state_hat

            # Convert the predictor/path representation to time domain
            self._route_future_state_time = self._patch_features_to_time(
                future_state_hat, self.pred_len
            )

        return z

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Restore y_norm to original scale, applying the route path if active.

        Base denorm:  y_base = base_time_mean + (base_time_std + eps) * y_norm
        Route output: y_route = route_path_impl(y_norm, y_base, …)

        Routing is skipped (falls back to y_base) when T != pred_len.
        """
        if self._pred_time_stats is None:
            return y_norm

        T = y_norm.shape[1]
        if T == self._pred_time_stats.shape[1]:
            time_stats = self._pred_time_stats
        else:
            time_stats = self._window_stats_to_time_stats(
                self._base_mu_fut_hat, self._base_std_fut_hat, T
            )
        mean, std = self._split_stats(time_stats)
        y_base = mean + (std + self.epsilon) * y_norm

        if (
            self.route_path != "none"
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
        """Helper: extract future windows and oracle (mu, std) from y_true."""
        true_windows = self._extract_windows(y_true)
        oracle_mu, oracle_std = self._compute_window_stats(true_windows)
        return true_windows, oracle_mu, oracle_std

    def compute_base_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Base SAN stats loss only — Stage 1 uses this.

        loss = w_mu * MSE(base_mu_fut_hat, oracle_mu)
             + w_std * MSE(base_std_fut_hat, oracle_std)

        Updates _last_mu_loss, _last_std_loss, _last_base_aux_loss.
        Does NOT touch _last_route_state_loss.
        """
        if self._base_mu_fut_hat is None or self._base_std_fut_hat is None:
            return torch.tensor(0.0, device=y_true.device)

        _, oracle_mu, oracle_std = self._compute_oracle_stats(y_true)
        mu_loss = F.mse_loss(self._base_mu_fut_hat, oracle_mu)
        std_loss = F.mse_loss(self._base_std_fut_hat, oracle_std)
        base_aux = self.w_mu * mu_loss + self.w_std * std_loss

        self._last_mu_loss = float(mu_loss.detach().item())
        self._last_std_loss = float(std_loss.detach().item())
        self._last_base_aux_loss = float(base_aux.detach().item())
        self._last_aux_total = self._last_base_aux_loss
        return base_aux

    def compute_route_state_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Route state prediction loss only — Stage 3 uses this.

        loss = MSE(future_state_hat_norm, oracle_state_norm)

        By default, both sides are normalised by _normalize_patch_state to ensure
        a consistent predictor scale independent of the physical quantity.
        omega_spec is the exception and keeps its raw bounded physical meaning.

        Updates _last_route_state_loss.  Returns 0 when route_path='none'.
        """
        if self.route_path == "none" or self._route_future_state_hat is None:
            return torch.tensor(0.0, device=y_true.device)

        true_windows, oracle_mu, oracle_std = self._compute_oracle_stats(y_true)
        oracle_raw = self.route_state_impl.build_future_oracle_state(
            y_true, true_windows, oracle_mu, oracle_std, self.sigma_min
        )
        if self._should_normalize_route_state():
            oracle_state = _normalize_patch_state(oracle_raw)
        else:
            oracle_state = oracle_raw
        self._route_future_oracle_state = oracle_state

        route_state_loss = F.mse_loss(self._route_future_state_hat, oracle_state)
        self._last_route_state_loss = float(route_state_loss.detach().item())
        return route_state_loss

    def compute_total_aux_loss(self, y_true: torch.Tensor) -> torch.Tensor:
        """Combined base aux + weighted route state loss.

        = compute_base_aux_loss + route_state_loss_scale * compute_route_state_loss

        Updates all _last_* tracking fields.
        """
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
        """Return cached scalar stats from the most recent loss computation.

        Keys are separated: base_aux stats and route_state_loss are distinct.
        Do not mix route path diagnostics here — use get_route_diagnostics().
        """
        return {
            "aux_total": self._last_aux_total,
            "base_aux_loss": self._last_base_aux_loss,
            "mu_loss": self._last_mu_loss,
            "std_loss": self._last_std_loss,
            "route_state_loss": self._last_route_state_loss,
            "route_state_framework_normalized": self._should_normalize_route_state(),
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
        """Return cached diagnostics from the route path implementation.

        Returns an empty dict when route_path='none'.
        Aux stats (mu_loss, route_state_loss, …) are NOT included here;
        use get_last_aux_stats() for those.
        """
        if self.route_path == "none" or self.route_path_impl is None:
            return {}
        diag: dict = {
            "route_path": self._route_path_name,
            "route_state": self._route_state_name,
            "route_state_name": self._route_state_name,
            "route_state_framework_normalized": self._should_normalize_route_state(),
        }
        if hasattr(self.route_path_impl, "get_route_diagnostics"):
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
        ).permute(0, 2, 1)  # (B, pred_stat_len, C)

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
            mu_hist - anchor.expand(-1, mu_hist.shape[1], -1),
            xbar,
            anchor,
        )
        std_fut = self.std_head(std_hist, xbar, None).clamp(min=self.sigma_min)
        return mu_fut, std_fut


class RouteStatePredictor(nn.Module):
    """Shared predictor for future route state — state-agnostic.

    Input:
        hist_state: (B, P_hist, C)  — normalised by _normalize_patch_state
        xbar:       (B, P_hist * window_len, C)
    Output:
        future_state_hat_raw: (B, P_fut, C)  — before adapt_future_state
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
        hist_state: torch.Tensor,  # (B, P_hist, C) — normalised
        xbar: torch.Tensor,        # (B, P_hist * window_len, C)
    ) -> torch.Tensor:
        hs = hist_state.permute(0, 2, 1)   # (B, C, P_hist)
        xr = xbar.permute(0, 2, 1)         # (B, C, raw_hist_len)
        feat = torch.cat([self.state_input(hs), self.raw_input(xr)], dim=-1)
        out = self.output(self.activation(feat))  # (B, C, P_fut)
        return out.permute(0, 2, 1)               # (B, P_fut, C)
