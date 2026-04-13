"""train_state_routes.py — Training entry point for SANRouteNorm experiments.

Stage protocol
--------------
Stages are executed in the order specified by `stage_order` (comma-separated).
Valid tokens: stage1, lp_pretrain, timeapn_pretrain, stage2, stage3, joint.

Default stage_order when left empty:
  route_path="none", route_state="none" (pure SAN):
                                   : stage1, stage2
  lp_state_correction + lp_state   : stage1, lp_pretrain, stage2
  timeapn_correction + timeapn_state: timeapn_pretrain, stage2
  all other route combinations      : stage1, stage2, stage3, joint

Stage semantics
---------------
  stage1:
      Optimizer: nm.predictor only
      Loss:      nm.compute_base_aux_loss
      lp combo:  std-only aux loss; mu loss is inactive
      Eval:      val base_aux_loss (patience-based early stop)

  lp_pretrain  [lp_state_correction + lp_state only]:
      Optimizer: nm.lp_state_predictor only
      Frozen:    fm, nm.predictor
      Loss:      nm.compute_route_state_loss (patch-level lp mean MSE)
      Eval:      val route_state_loss (patience-based early stop)
      Max epochs: route_stage_epochs

  timeapn_pretrain  [timeapn_correction + timeapn_state only]:
      Optimizer: nm.apn_module only (station_optim, lr=timeapn_station_lr)
      Frozen:    fm
      Loss:      nm.compute_route_state_loss (official sliding_loss_P:
                   MSE(recon, y_true) + MSE(pred_mean, true_mean) +
                   MSE(combined_phase, true_phase))
      Eval:      val route_state_loss (patience-based early stop)
      Max epochs: timeapn_pre_epoch

  stage2:
      All combos:  fm only; nm fully frozen
      Eval:        val MSE (patience-based early stop)
      TimeAPN: fm only throughout; APN stays frozen.
               Only if timeapn_enable_late_merge=True AND epoch==timeapn_twice_epoch,
               APN params are added to the fm optimizer (late merge).

  stage3  [not valid for lp_state_correction + lp_state]:
      nm.route_modules only; fm + nm.predictor frozen
      Loss: route_task_loss_scale * task + route_state_loss_scale * route_state_loss
      Eval: val MSE (patience-based early stop)
      Max epochs: route_stage_epochs

  joint  [not valid for lp_state_correction + lp_state]:
      fm + nm.route_modules; nm.predictor frozen; task loss

Each stage starts from the previous stage's best checkpoint.
best_stage records the stage that produced the best overall val MSE.

Result directory:
  results/state_routes/<exp_name>/<dataset>/<backbone>/pred_len_<pred_len>/
      <route_path>/<route_state>/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ttn_norm.experiments.train import (
    TrainConfig,
    _build_backbone_kwargs,
    _build_dataloader,
    _build_metrics,
    _make_dec_inputs,
    _move_model_to_device,
    _set_seed,
)
from ttn_norm.models import TTNModel
from ttn_norm.normalizations import SANRouteNorm


def _forward(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_x_enc: torch.Tensor,
    batch_y_enc: torch.Tensor,
    cfg: "StateRouteConfig",
) -> torch.Tensor:
    """Forward pass that constructs dec_inp for former backbones (Informer etc.)."""
    dec_inp, dec_inp_enc = None, None
    if model.is_former:
        label_len = min(cfg.label_len, batch_x.size(1))
        dec_inp, dec_inp_enc = _make_dec_inputs(batch_x, batch_y_enc, batch_x_enc, label_len)
    return model(batch_x, batch_x_enc, dec_inp, dec_inp_enc)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StateRouteConfig:
    """Configuration for SANRouteNorm stage-order experiments."""

    # Experiment identity
    exp_name: str = "san_route_base"
    dataset: str = "ETTh1"
    backbone: str = "DLinear"

    # Data
    data_path: str = "./data"
    split_type: str = "ratio"
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    freq: str = "h"
    scaler_type: str = "StandarScaler"
    scale_in_train: bool = False
    batch_size: int = 32
    num_worker: int = 4

    # Model geometry
    window: int = 96
    pred_len: int = 96
    horizon: int = 1
    label_len: int = 48

    # Training (shared across stages unless overridden per-stage)
    device: str = "cuda:0"
    cuda_oom_cpu_fallback: bool = False
    seeds: str = "1"
    epochs: int = 1000       # Stage 2 max epochs
    lr: float = 1.5e-4       # Stage 1 and Stage 2 lr
    weight_decay: float = 5e-4
    max_grad_norm: float = 5.0

    # Early stopping (Stage 1 and Stage 2 share these)
    early_stop: bool = True
    early_stop_patience: int = 5
    early_stop_min_epochs: int = 1
    early_stop_delta: float = 0.0

    # Aux loss schedule (Stage 1 only; Stage 2 uses task loss only)
    aux_loss_scale: float = 0.2
    aux_loss_min_scale: float = 0.05
    aux_loss_schedule: str = "cosine"
    aux_loss_decay_start_epoch: int = 4
    aux_loss_decay_epochs: int = 16

    # SANRouteNorm geometry
    san_period_len: int = 12
    san_stride: int = 0
    san_sigma_min: float = 1e-3
    san_w_mu: float = 1.0
    san_w_std: float = 1.0

    # Stage 1 epochs (nm.predictor pretrain)
    san_pretrain_epochs: int = 5

    # Route path / state
    route_path: str = "none"
    route_state: str = "none"
    route_state_loss_scale: float = 0.1

    # Stage 3 / lp_pretrain config (route_stage_epochs serves both)
    route_stage_epochs: int = 0           # 0 = skip stage3 / lp_pretrain
    route_stage_lr: float = 1.5e-4
    route_stage_weight_decay: float = 5e-4
    route_stage_patience: int = 5
    route_stage_min_epochs: int = 1
    route_task_loss_scale: float = 1.0    # weight of task loss in stage3

    # Joint finetune
    joint_finetune_epochs: int = 0        # 0 = skip
    joint_finetune_lr: float = 1e-5

    # Stage order control
    stage_order: str = ""                 # "" = use default for the route combo

    # Non-overlap enforcement for route experiments
    route_require_nonoverlap: bool = True

    # Checkpoint persistence & loading
    save_stage2_ckpt: bool = False
    baseline_ckpt_dir: str = ""

    # Results
    result_dir: str = "./results/state_routes"

    # ------------------------------------------------------------------
    # TimeAPN-specific parameters (official APN replication)
    # ------------------------------------------------------------------
    # APN module construction
    timeapn_j: int = 0                  # DWT decomposition levels (0 = no DWT)
    timeapn_learnable: bool = True      # learnable DWT filter coefficients
    timeapn_wavelet: str = "bior3.5"    # wavelet name for DWT1D
    timeapn_dr: float = 0.05            # dropout rate for Statics_MLP
    timeapn_kernel_len: int = 7         # sliding-window kernel
    timeapn_hkernel_len: int = 5        # DWT-band sliding-window kernel
    timeapn_pd_model: int = 256         # hidden dimension d_model for Statics_MLP
    timeapn_pd_ff: int = 512            # FFN hidden dimension d_ff
    timeapn_pe_layers: int = 0          # FFN layers in mean_ffn

    # Training schedule
    timeapn_pre_epoch: int = 5          # APN-only pretrain epochs (0 = skip)
    timeapn_twice_epoch: int = -1       # epoch in stage2 to add APN to fm optim (-1 = never)
    timeapn_enable_late_merge: bool = False  # must be True to activate twice_epoch late merge
    timeapn_station_lr: float = 1e-3    # APN pretrain optimizer lr

    # ------------------------------------------------------------------
    # Inference-time update (B2SC) parameters
    # ------------------------------------------------------------------
    b2sc_enable: bool = False
    b2sc_recent_weight: float = 0.75
    b2sc_prev_weight: float = 0.25
    b2sc_second_slice_scale: float = 0.5
    b2sc_apply_on_val: bool = True
    b2sc_apply_on_test: bool = True

    # ------------------------------------------------------------------
    # Inference-time streaming state calibration (SSC) parameters
    # ------------------------------------------------------------------
    ssc_enable: bool = False
    ssc_lambda: float = 0.9
    ssc_decay_rho: float = 0.5
    ssc_scale_beta: float = 0.5
    ssc_trend_scale: float = 0.1
    ssc_apply_on_val: bool = True
    ssc_apply_on_test: bool = True
    ssc_sample_stride: int = 1

    # ------------------------------------------------------------------
    # Post-training plot parameters
    # ------------------------------------------------------------------
    plot_comparison: bool = True    # save oracle vs pred line plots after training
    plot_n_samples: int = 8         # number of consecutive test samples to plot
    plot_channel: int = 0           # which feature channel to visualise

    def seed_list(self) -> list[int]:
        return [int(s.strip()) for s in self.seeds.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Stage order resolution
# ---------------------------------------------------------------------------

_VALID_STAGE_TOKENS = {
    "stage1",
    "lp_pretrain",
    "timeapn_pretrain",
    "stage2",
    "stage3",
    "joint",
}


def _parse_stage_order(cfg: StateRouteConfig) -> list[str]:
    """Return ordered list of stage tokens to execute.

    Empty stage_order → default for the route combo:
      pure SAN (route_path="none", route_state="none"):
                                       ["stage1", "stage2"]
      lp_state_correction + lp_state : ["stage1", "lp_pretrain", "stage2"]
      timeapn_correction + timeapn_state:
                                       ["timeapn_pretrain", "stage2"]
      others                          : ["stage1", "stage2", "stage3", "joint"]

    Raises ValueError for unknown tokens, duplicates, or invalid combos.
    lp combo: joint and stage3 are forbidden.
    pure SAN baseline: only ["stage1", "stage2"] is valid.
    """
    is_lp      = (cfg.route_path == "lp_state_correction" and cfg.route_state == "lp_state")
    is_pure_san = _is_pure_san_baseline(cfg)
    is_timeapn = (cfg.route_path == "timeapn_correction" and cfg.route_state == "timeapn_state")

    if cfg.stage_order == "":
        if is_pure_san:
            return ["stage1", "stage2"]
        elif is_lp:
            return ["stage1", "lp_pretrain", "stage2"]
        elif is_timeapn:
            return ["timeapn_pretrain", "stage2"]
        else:
            return ["stage1", "stage2", "stage3", "joint"]

    if is_pure_san:
        tokens = [t.strip() for t in cfg.stage_order.split(",") if t.strip()]
        if tokens != ["stage1", "stage2"]:
            raise ValueError(
                f"pure SAN baseline only allows stage_order='stage1,stage2', "
                f"got {tokens}"
            )
        return tokens

    tokens = [t.strip() for t in cfg.stage_order.split(",") if t.strip()]
    for tok in tokens:
        if tok not in _VALID_STAGE_TOKENS:
            raise ValueError(
                f"Unknown stage token '{tok}'. Valid: {sorted(_VALID_STAGE_TOKENS)}"
            )
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate tokens in stage_order: {tokens}")
    if is_lp and "stage3" in tokens:
        raise ValueError(
            "stage3 is not valid for lp_state_correction + lp_state. "
            "Use lp_pretrain instead."
        )
    if is_lp and "joint" in tokens:
        raise ValueError(
            "joint is not valid for lp_state_correction + lp_state. "
            "Stage 2 trains fm only with both nm modules frozen."
        )
    if is_lp and "timeapn_pretrain" in tokens:
        raise ValueError(
            "timeapn_pretrain is not valid for lp_state_correction + lp_state."
        )
    if is_timeapn and "lp_pretrain" in tokens:
        raise ValueError(
            "lp_pretrain is not valid for timeapn_correction + timeapn_state. "
            "Use timeapn_pretrain instead."
        )
    return tokens


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _is_pure_san_baseline(cfg: StateRouteConfig) -> bool:
    """True iff this run is the pure SAN baseline (no routing)."""
    return cfg.route_path == "none" and cfg.route_state == "none"


def _build_san_route_model(cfg: StateRouteConfig, num_features: int) -> TTNModel:
    # SSC hard constraints (must be checked before pure/non-pure split)
    if cfg.ssc_enable and not _is_pure_san_baseline(cfg):
        raise ValueError(
            "ssc_enable requires pure SAN baseline "
            "(route_path='none', route_state='none')."
        )
    if cfg.ssc_enable and cfg.b2sc_enable:
        raise ValueError(
            "ssc_enable and b2sc_enable cannot be enabled together."
        )
    if cfg.ssc_enable and cfg.ssc_sample_stride != 1:
        raise ValueError("ssc_sample_stride must be 1 when ssc_enable=True.")

    if _is_pure_san_baseline(cfg):
        # Hard divisibility checks for pure SAN
        if cfg.window % cfg.san_period_len != 0:
            raise ValueError(
                f"pure SAN baseline requires window % san_period_len == 0, "
                f"got window={cfg.window}, san_period_len={cfg.san_period_len}"
            )
        if cfg.pred_len % cfg.san_period_len != 0:
            raise ValueError(
                f"pure SAN baseline requires pred_len % san_period_len == 0, "
                f"got pred_len={cfg.pred_len}, san_period_len={cfg.san_period_len}"
            )
        # stride=0 → SANRouteNorm internally sets stride=period_len (non-overlapping)
        norm = SANRouteNorm(
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            period_len=cfg.san_period_len,
            enc_in=num_features,
            stride=0,
            sigma_min=cfg.san_sigma_min,
            w_mu=cfg.san_w_mu,
            w_std=cfg.san_w_std,
            route_path="none",
            route_state="none",
            route_state_loss_scale=0.0,
            b2sc_enable=cfg.b2sc_enable,
            b2sc_recent_weight=cfg.b2sc_recent_weight,
            b2sc_prev_weight=cfg.b2sc_prev_weight,
            b2sc_second_slice_scale=cfg.b2sc_second_slice_scale,
            ssc_enable=cfg.ssc_enable,
            ssc_lambda=cfg.ssc_lambda,
            ssc_decay_rho=cfg.ssc_decay_rho,
            ssc_scale_beta=cfg.ssc_scale_beta,
            ssc_trend_scale=cfg.ssc_trend_scale,
        )
    else:
        norm = SANRouteNorm(
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            period_len=cfg.san_period_len,
            enc_in=num_features,
            stride=cfg.san_stride,
            sigma_min=cfg.san_sigma_min,
            w_mu=cfg.san_w_mu,
            w_std=cfg.san_w_std,
            route_path=cfg.route_path,
            route_state=cfg.route_state,
            route_state_loss_scale=cfg.route_state_loss_scale,
            timeapn_j=cfg.timeapn_j,
            timeapn_learnable=cfg.timeapn_learnable,
            timeapn_wavelet=cfg.timeapn_wavelet,
            timeapn_dr=cfg.timeapn_dr,
            timeapn_kernel_len=cfg.timeapn_kernel_len,
            timeapn_hkernel_len=cfg.timeapn_hkernel_len,
            timeapn_pd_model=cfg.timeapn_pd_model,
            timeapn_pd_ff=cfg.timeapn_pd_ff,
            timeapn_pe_layers=cfg.timeapn_pe_layers,
            b2sc_enable=cfg.b2sc_enable,
            b2sc_recent_weight=cfg.b2sc_recent_weight,
            b2sc_prev_weight=cfg.b2sc_prev_weight,
            b2sc_second_slice_scale=cfg.b2sc_second_slice_scale,
        )
        if cfg.route_require_nonoverlap:
            if norm.stride != norm.window_len:
                raise ValueError(
                    f"route_require_nonoverlap=True requires stride == period_len "
                    f"(got stride={norm.stride}, period_len={norm.window_len}). "
                    f"Use san_stride=0 or san_stride=san_period_len."
                )
    tc = TrainConfig(
        backbone_type=cfg.backbone,
        window=cfg.window,
        pred_len=cfg.pred_len,
        horizon=cfg.horizon,
        label_len=cfg.label_len,
        freq=cfg.freq,
    )
    label_len_v = min(cfg.label_len, cfg.window // 2)
    backbone_kwargs = _build_backbone_kwargs(tc, num_features, label_len_v)
    return TTNModel(
        backbone_type=cfg.backbone,
        backbone_kwargs=backbone_kwargs,
        norm_model=norm,
    )



# ---------------------------------------------------------------------------
# Per-stage training functions
# ---------------------------------------------------------------------------

def _train_epoch_stage1(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    cfg: StateRouteConfig,
) -> dict[str, float]:
    """Stage 1: nm.predictor only, base aux loss."""
    model.train()
    base_aux_list: list[float] = []
    mu_list: list[float] = []
    std_list: list[float] = []

    for batch_x, batch_y, _origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x     = batch_x.to(cfg.device).float()
        batch_y     = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()

        if hasattr(model.nm, "set_b2sc_mode"):
            model.nm.set_b2sc_mode(False)
        optim.zero_grad()
        _forward(model, batch_x, batch_x_enc, batch_y_enc, cfg)
        base_aux = model.nm.compute_base_aux_loss(batch_y)
        loss = base_aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.nm.parameters_base_predictor(), cfg.max_grad_norm
        )
        optim.step()

        base_aux_list.append(float(base_aux.item()))
        aux_stats = model.nm.get_last_aux_stats()
        mu_list.append(aux_stats["mu_loss"])
        std_list.append(aux_stats["std_loss"])

    return {
        "base_aux_loss": float(np.mean(base_aux_list)) if base_aux_list else 0.0,
        "mu_loss":       float(np.mean(mu_list))       if mu_list       else 0.0,
        "std_loss":      float(np.mean(std_list))      if std_list      else 0.0,
    }


def _train_epoch_lp_pretrain(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    cfg: StateRouteConfig,
) -> dict[str, float]:
    """lp_pretrain: nm.lp_state_predictor only.
    Loss: patch-level MSE where lp mean is defined by patch_mean(lowpass_time(raw_series)).
    """
    model.train()
    route_state_list: list[float] = []

    for batch_x, batch_y, _origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x     = batch_x.to(cfg.device).float()
        batch_y     = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()

        if hasattr(model.nm, "set_b2sc_mode"):
            model.nm.set_b2sc_mode(False)
        optim.zero_grad()
        _forward(model, batch_x, batch_x_enc, batch_y_enc, cfg)
        rs_loss = model.nm.compute_route_state_loss(batch_y)
        rs_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.nm.lp_state_predictor.parameters()), cfg.max_grad_norm
        )
        optim.step()

        route_state_list.append(float(rs_loss.item()))

    return {
        "route_state_loss": float(np.mean(route_state_list)) if route_state_list else 0.0,
    }


def _train_epoch_timeapn_pretrain(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    cfg: StateRouteConfig,
) -> dict[str, float]:
    """timeapn_pretrain: APN module only (station_optim).

    fm is frozen (requires_grad=False for all fm params).
    Loss: official sliding_loss_P via nm.compute_route_state_loss.
    Early stop on total route_state_loss.
    """
    model.train()
    route_state_list: list[float] = []
    am_list:    list[float] = []
    phase_list: list[float] = []
    recon_list: list[float] = []

    for batch_x, batch_y, _origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x     = batch_x.to(cfg.device).float()
        batch_y     = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()

        if hasattr(model.nm, "set_b2sc_mode"):
            model.nm.set_b2sc_mode(False)
        optim.zero_grad()
        _forward(model, batch_x, batch_x_enc, batch_y_enc, cfg)
        rs_loss = model.nm.compute_route_state_loss(batch_y)
        rs_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.nm.parameters_route_modules(), cfg.max_grad_norm
        )
        optim.step()

        route_state_list.append(float(rs_loss.item()))
        aux_stats = model.nm.get_last_aux_stats()
        am_list.append(aux_stats.get("apn_loss_mean",  0.0))
        phase_list.append(aux_stats.get("apn_loss_phase", 0.0))
        recon_list.append(aux_stats.get("apn_loss_recon",  0.0))

    return {
        "route_state_loss": float(np.mean(route_state_list)) if route_state_list else 0.0,
        "apn_loss_mean":    float(np.mean(am_list))    if am_list    else 0.0,
        "apn_loss_phase":   float(np.mean(phase_list)) if phase_list else 0.0,
        "apn_loss_recon":   float(np.mean(recon_list)) if recon_list else 0.0,
    }


def _train_epoch_stage2(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    cfg: StateRouteConfig,
) -> dict[str, float]:
    """Stage 2 (standard): backbone only, task loss. nm is frozen; aux monitored."""
    model.train()
    loss_fn = nn.MSELoss()
    task_list: list[float] = []
    base_aux_list: list[float] = []
    route_state_list: list[float] = []

    for batch_x, batch_y, _origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x     = batch_x.to(cfg.device).float()
        batch_y     = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()

        if hasattr(model.nm, "set_b2sc_mode"):
            model.nm.set_b2sc_mode(False)
        optim.zero_grad()
        pred = _forward(model, batch_x, batch_x_enc, batch_y_enc, cfg)
        task_loss = loss_fn(pred, batch_y)
        task_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.fm.parameters(), cfg.max_grad_norm)
        optim.step()

        task_list.append(float(task_loss.item()))

        with torch.no_grad():
            model.nm.compute_total_aux_loss(batch_y)
        aux_stats = model.nm.get_last_aux_stats()
        base_aux_list.append(aux_stats["base_aux_loss"])
        route_state_list.append(aux_stats["route_state_loss"])

    return {
        "task_loss":        float(np.mean(task_list))        if task_list        else 0.0,
        "base_aux_loss":    float(np.mean(base_aux_list))    if base_aux_list    else 0.0,
        "route_state_loss": float(np.mean(route_state_list)) if route_state_list else 0.0,
    }


def _train_epoch_stage2_ssc(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    cfg: StateRouteConfig,
) -> dict[str, float]:
    """Stage 2 (SSC-aware): sequential sample-by-sample training with live SSC update."""
    model.train()
    loss_fn = nn.MSELoss()
    task_list: list[float] = []
    base_aux_list: list[float] = []
    route_state_list: list[float] = []

    model.nm.reset_ssc_state()
    model.nm.set_ssc_mode(True)

    optim.zero_grad()
    sample_counter = 0
    batch_loss_accum = torch.tensor(0.0, device=cfg.device)

    for batch_x, batch_y, _origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x     = batch_x.to(cfg.device).float()
        batch_y     = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()
        batch_size  = batch_x.shape[0]

        batch_base_aux_vals: list[float] = []
        batch_route_state_vals: list[float] = []

        for i in range(batch_size):
            x_i   = batch_x[i:i+1]
            y_i   = batch_y[i:i+1]
            xe_i  = batch_x_enc[i:i+1]
            ye_i  = batch_y_enc[i:i+1]

            sample_start = sample_counter * cfg.ssc_sample_stride
            model.nm.ssc_update_from_current_lookback(x_i, sample_start)

            pred_i    = _forward(model, x_i, xe_i, ye_i, cfg)
            samp_loss = loss_fn(pred_i, y_i)
            batch_loss_accum = batch_loss_accum + samp_loss

            model.nm.ssc_cache_current_base_prediction(sample_start + cfg.window)
            sample_counter += 1

            with torch.no_grad():
                model.nm.compute_total_aux_loss(y_i)
            aux_stats = model.nm.get_last_aux_stats()
            batch_base_aux_vals.append(aux_stats["base_aux_loss"])
            batch_route_state_vals.append(aux_stats["route_state_loss"])

        # backward once per original batch
        avg_loss = batch_loss_accum / batch_size
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.fm.parameters(), cfg.max_grad_norm)
        optim.step()
        optim.zero_grad()
        batch_loss_accum = torch.tensor(0.0, device=cfg.device)

        task_list.append(float(avg_loss.item()))
        base_aux_list.append(float(np.mean(batch_base_aux_vals)) if batch_base_aux_vals else 0.0)
        route_state_list.append(float(np.mean(batch_route_state_vals)) if batch_route_state_vals else 0.0)

    model.nm.set_ssc_mode(False)

    return {
        "task_loss":        float(np.mean(task_list))        if task_list        else 0.0,
        "base_aux_loss":    float(np.mean(base_aux_list))    if base_aux_list    else 0.0,
        "route_state_loss": float(np.mean(route_state_list)) if route_state_list else 0.0,
    }


def _train_epoch_stage3(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    cfg: StateRouteConfig,
    stage3_params: list,
) -> dict[str, float]:
    """Stage 3: route modules only. fm + predictor frozen.
    Loss = route_task_loss_scale * task + route_state_loss_scale * route_state_loss.
    """
    model.train()
    loss_fn = nn.MSELoss()
    task_list: list[float] = []
    route_state_list: list[float] = []
    total_list: list[float] = []

    for batch_x, batch_y, _origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x     = batch_x.to(cfg.device).float()
        batch_y     = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()

        if hasattr(model.nm, "set_b2sc_mode"):
            model.nm.set_b2sc_mode(False)
        optim.zero_grad()
        pred = _forward(model, batch_x, batch_x_enc, batch_y_enc, cfg)
        task_loss        = loss_fn(pred, batch_y)
        route_state_loss = model.nm.compute_route_state_loss(batch_y)
        loss = (
            cfg.route_task_loss_scale * task_loss
            + model.nm.route_state_loss_scale * route_state_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stage3_params, cfg.max_grad_norm)
        optim.step()

        task_list.append(float(task_loss.item()))
        route_state_list.append(float(route_state_loss.item()))
        total_list.append(float(loss.item()))

    return {
        "task_loss":        float(np.mean(task_list))        if task_list        else 0.0,
        "route_state_loss": float(np.mean(route_state_list)) if route_state_list else 0.0,
        "total_loss":       float(np.mean(total_list))       if total_list       else 0.0,
    }


def _train_epoch_joint(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    cfg: StateRouteConfig,
    joint_params: list,
) -> dict[str, float]:
    """Joint (standard): fm + route modules, task loss only. nm.predictor frozen."""
    model.train()
    loss_fn = nn.MSELoss()
    task_list: list[float] = []

    for batch_x, batch_y, _origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x     = batch_x.to(cfg.device).float()
        batch_y     = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()

        if hasattr(model.nm, "set_b2sc_mode"):
            model.nm.set_b2sc_mode(False)
        optim.zero_grad()
        pred = _forward(model, batch_x, batch_x_enc, batch_y_enc, cfg)
        task_loss = loss_fn(pred, batch_y)
        task_loss.backward()
        torch.nn.utils.clip_grad_norm_(joint_params, cfg.max_grad_norm)
        optim.step()

        task_list.append(float(task_loss.item()))

    return {
        "task_loss": float(np.mean(task_list)) if task_list else 0.0,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_loader(
    model: nn.Module,
    loader,
    cfg: StateRouteConfig,
    metrics: dict,
    enable_b2sc_eval: bool = False,
    enable_ssc_eval: bool = False,
    reset_ssc: bool = True,
) -> dict[str, float]:
    """Evaluate *loader*.

    ``reset_ssc``: when True (the default and the only value used for test
    loaders) SSC state is reset before eval so that test always cold-starts,
    never inheriting val's end-of-pass memory.
    """
    model.eval()
    for metric in metrics.values():
        metric.reset()

    loss_fn = nn.MSELoss()
    task_list: list[float] = []
    base_aux_list: list[float] = []
    route_state_list: list[float] = []

    # ------------------------------------------------------------------
    # SSC sequential eval path
    # ------------------------------------------------------------------
    if enable_ssc_eval and hasattr(model.nm, "set_ssc_mode"):
        if reset_ssc:
            model.nm.reset_ssc_state()
        model.nm.set_ssc_mode(True)
        sample_counter = 0

        for batch_x, batch_y, _origin_y, batch_x_enc, batch_y_enc in loader:
            batch_x     = batch_x.to(cfg.device).float()
            batch_y     = batch_y.to(cfg.device).float()
            batch_x_enc = batch_x_enc.to(cfg.device).float()
            batch_y_enc = batch_y_enc.to(cfg.device).float()

            for i in range(batch_x.shape[0]):
                x_i     = batch_x[i:i+1]
                y_i     = batch_y[i:i+1]
                xenc_i  = batch_x_enc[i:i+1]
                yenc_i  = batch_y_enc[i:i+1]

                sample_start = sample_counter * cfg.ssc_sample_stride
                model.nm.ssc_update_from_current_lookback(x_i, sample_start)
                pred_i = _forward(model, x_i, xenc_i, yenc_i, cfg)
                model.nm.ssc_cache_current_base_prediction(
                    sample_start + cfg.window
                )
                sample_counter += 1

                task_loss = loss_fn(pred_i, y_i)
                task_list.append(float(task_loss.item()))

                model.nm.compute_total_aux_loss(y_i)
                aux_stats = model.nm.get_last_aux_stats()
                base_aux_list.append(aux_stats["base_aux_loss"])
                route_state_list.append(aux_stats["route_state_loss"])

                pred_i = pred_i.contiguous()
                y_i    = y_i.contiguous()
                if cfg.pred_len == 1:
                    pred_i = pred_i.view(1, -1)
                    y_i    = y_i.view(1, -1)
                for metric in metrics.values():
                    metric.update(pred_i, y_i)

        model.nm.set_ssc_mode(False)
        results = {name: float(m.compute()) for name, m in metrics.items()}
        results.update({
            "task_loss":        float(np.mean(task_list))        if task_list        else 0.0,
            "base_aux_loss":    float(np.mean(base_aux_list))    if base_aux_list    else 0.0,
            "route_state_loss": float(np.mean(route_state_list)) if route_state_list else 0.0,
        })
        return results

    # ------------------------------------------------------------------
    # Standard batched eval path
    # ------------------------------------------------------------------
    for batch_x, batch_y, _origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x     = batch_x.to(cfg.device).float()
        batch_y     = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()

        if hasattr(model.nm, "set_b2sc_mode"):
            model.nm.set_b2sc_mode(enable_b2sc_eval)

        pred = _forward(model, batch_x, batch_x_enc, batch_y_enc, cfg)
        task_loss = loss_fn(pred, batch_y)
        task_list.append(float(task_loss.item()))

        model.nm.compute_total_aux_loss(batch_y)
        aux_stats = model.nm.get_last_aux_stats()
        base_aux_list.append(aux_stats["base_aux_loss"])
        route_state_list.append(aux_stats["route_state_loss"])

        pred    = pred.contiguous()
        batch_y = batch_y.contiguous()
        if cfg.pred_len == 1:
            B = pred.shape[0]
            pred    = pred.view(B, -1)
            batch_y = batch_y.view(B, -1)
        for metric in metrics.values():
            metric.update(pred, batch_y)

    results = {name: float(m.compute()) for name, m in metrics.items()}
    results.update({
        "task_loss":        float(np.mean(task_list))        if task_list        else 0.0,
        "base_aux_loss":    float(np.mean(base_aux_list))    if base_aux_list    else 0.0,
        "route_state_loss": float(np.mean(route_state_list)) if route_state_list else 0.0,
    })
    if hasattr(model.nm, "set_b2sc_mode"):
        model.nm.set_b2sc_mode(False)
    return results


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def _run_trial(cfg: StateRouteConfig, seed: int, tmp_dir: str) -> dict[str, Any]:
    _set_seed(seed)
    pure_baseline = _is_pure_san_baseline(cfg)
    is_lp      = (cfg.route_path == "lp_state_correction" and cfg.route_state == "lp_state")
    is_timeapn = (cfg.route_path == "timeapn_correction"  and cfg.route_state == "timeapn_state")

    enable_val_b2sc  = bool(cfg.b2sc_enable and cfg.b2sc_apply_on_val)
    enable_test_b2sc = bool(cfg.b2sc_enable and cfg.b2sc_apply_on_test)
    enable_val_ssc   = bool(cfg.ssc_enable and cfg.ssc_apply_on_val)
    enable_test_ssc  = bool(cfg.ssc_enable and cfg.ssc_apply_on_test)

    if pure_baseline and cfg.baseline_ckpt_dir:
        raise ValueError(
            "pure SAN baseline run cannot use baseline_ckpt_dir"
        )

    tc = TrainConfig(
        dataset_type=cfg.dataset,
        data_path=cfg.data_path,
        device=cfg.device,
        num_worker=cfg.num_worker,
        seed=seed,
        backbone_type=cfg.backbone,
        window=cfg.window,
        pred_len=cfg.pred_len,
        horizon=cfg.horizon,
        batch_size=cfg.batch_size,
        scaler_type=cfg.scaler_type,
        scale_in_train=cfg.scale_in_train,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        split_type=cfg.split_type,
        freq=cfg.freq,
        norm_type="san_route",
        wav_ctx_patches=0,
    )
    dataloader, dataset, scaler, split_info = _build_dataloader(tc)
    num_features = dataset.num_features

    # Ordered (non-shuffled) replay loader for SSC-aware stage2 training
    _ssc_train_loader = DataLoader(
        dataloader.train_loader.dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_worker,
    ) if cfg.ssc_enable else None

    print(
        f"[Split] train={dataloader.train_size}, val={dataloader.val_size},"
        f" test={dataloader.test_size}"
    )

    model   = _move_model_to_device(_build_san_route_model(cfg, num_features), cfg)
    metrics = _build_metrics(torch.device(cfg.device))

    stage_order = _parse_stage_order(cfg)

    # Checkpoint paths
    norm_ckpt             = os.path.join(tmp_dir, f"norm_s{seed}.pt")
    lp_pretrain_ckpt      = os.path.join(tmp_dir, f"lp_pretrain_s{seed}.pt")
    timeapn_pretrain_ckpt = os.path.join(tmp_dir, f"timeapn_pretrain_s{seed}.pt")
    stage2_ckpt           = os.path.join(tmp_dir, f"stage2_s{seed}.pt")
    stage3_ckpt           = os.path.join(tmp_dir, f"stage3_s{seed}.pt")
    joint_ckpt            = os.path.join(tmp_dir, f"joint_s{seed}.pt")

    t0 = time.time()
    best_val_mse  = float("inf")
    best_val_mae  = float("inf")
    best_test_mse = float("inf")
    best_test_mae = float("inf")
    best_epoch    = 0
    best_stage    = "none"

    # last_ckpt: most recent full-model checkpoint from a completed stage
    last_ckpt: str | None = None
    # Stages to skip when using baseline_ckpt_dir
    skip_stages: frozenset[str] = frozenset()

    # -------------------------------------------------------------------
    # Baseline checkpoint loading (skip stage1, lp_pretrain, stage2)
    # Only allowed for route experiments (not pure baseline)
    # -------------------------------------------------------------------
    if cfg.baseline_ckpt_dir:
        # pure_baseline guard already raised above; this branch is route-only
        meta_path = os.path.join(cfg.baseline_ckpt_dir, f"stage2_s{seed}_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as _mf:
                _meta = json.load(_mf)
            if _meta.get("route_path") != "none" or _meta.get("route_state") != "none":
                raise ValueError(
                    f"baseline_ckpt_dir must come from a pure SAN baseline "
                    f"(route_path='none', route_state='none'), "
                    f"but meta reports route_path='{_meta.get('route_path')}', "
                    f"route_state='{_meta.get('route_state')}'"
                )
        else:
            # Fall back to config.json if meta not present
            cfg_json_path = os.path.join(cfg.baseline_ckpt_dir, "..", "config.json")
            cfg_json_path = os.path.normpath(cfg_json_path)
            if os.path.isfile(cfg_json_path):
                with open(cfg_json_path) as _cf:
                    _bcfg = json.load(_cf)
                if _bcfg.get("route_path") != "none" or _bcfg.get("route_state") != "none":
                    raise ValueError(
                        f"baseline_ckpt_dir must come from a pure SAN baseline "
                        f"(route_path='none', route_state='none'), "
                        f"but config.json reports route_path='{_bcfg.get('route_path')}', "
                        f"route_state='{_bcfg.get('route_state')}'"
                    )
        ckpt_path = os.path.join(cfg.baseline_ckpt_dir, f"stage2_s{seed}.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Baseline checkpoint not found: {ckpt_path}\n"
                f"Run the baseline experiment with --save-stage2-ckpt first."
            )
        baseline_sd = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
        missing, unexpected = model.load_state_dict(baseline_sd, strict=False)
        print(f"\n--- Loading baseline checkpoint: {ckpt_path} ---")
        if missing:
            print(f"  Missing keys (expected for new route modules): {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        val_m  = _eval_loader(model, dataloader.val_loader,  cfg, metrics, enable_b2sc_eval=enable_val_b2sc,  enable_ssc_eval=enable_val_ssc)
        test_m = _eval_loader(model, dataloader.test_loader, cfg, metrics, enable_b2sc_eval=enable_test_b2sc, enable_ssc_eval=enable_test_ssc, reset_ssc=True)
        best_val_mse  = val_m["mse"]
        best_val_mae  = val_m["mae"]
        best_test_mse = test_m["mse"]
        best_test_mae = test_m["mae"]
        print(
            f"  Baseline eval — val_mse={best_val_mse:.6f}"
            f"  test_mse={best_test_mse:.6f}  test_mae={best_test_mae:.6f}"
        )
        torch.save(model.state_dict(), stage2_ckpt)
        last_ckpt   = stage2_ckpt
        skip_stages = frozenset({"stage1", "lp_pretrain", "timeapn_pretrain", "stage2"})
        print("  Skipping stage1, *_pretrain, stage2 — proceeding with remaining stages.")

    # -------------------------------------------------------------------
    # Execute stages in order
    # -------------------------------------------------------------------
    for stage_name in stage_order:
        if stage_name in skip_stages:
            continue

        # Load from the most recent full-model checkpoint before each stage
        # (stage1 is exempt: it only trains nm.predictor and doesn't need a prior checkpoint)
        if last_ckpt is not None and stage_name != "stage1":
            model.load_state_dict(torch.load(last_ckpt, weights_only=True))

        # ==========================================================
        # stage1: nm.predictor pretrain
        # Exact original SAN: train statistics predictor on oracle patch mean/std.
        # ==========================================================
        if stage_name == "stage1":
            if cfg.san_pretrain_epochs <= 0:
                continue
            _s1_label = (
                "Stage 1: pure SAN predictor pretrain"
                if pure_baseline else
                "Stage 1: base predictor pretrain"
            )
            print(
                f"\n--- {_s1_label}"
                f" (max {cfg.san_pretrain_epochs} epochs, patience={cfg.early_stop_patience}) ---"
            )
            for p in model.fm.parameters():
                p.requires_grad_(False)
            model.nm.freeze_route_modules()

            s1_optim = torch.optim.Adam(
                model.nm.parameters_base_predictor(),
                lr=cfg.lr, weight_decay=cfg.weight_decay,
            )
            best_s1_val = float("inf")
            s1_patience = 0

            for epoch in range(cfg.san_pretrain_epochs):
                tr    = _train_epoch_stage1(model, dataloader.train_loader, s1_optim, cfg)
                val_m = _eval_loader(model, dataloader.val_loader, cfg, metrics, enable_b2sc_eval=enable_val_b2sc, enable_ssc_eval=False)
                val_base_aux = val_m["base_aux_loss"]

                improved = val_base_aux < best_s1_val - cfg.early_stop_delta
                marker   = "  *" if improved else ""
                if improved:
                    best_s1_val = val_base_aux
                    s1_patience = 0
                    torch.save(model.nm.predictor.state_dict(), norm_ckpt)
                else:
                    s1_patience += 1

                print(
                    f"[Stage1 epoch {epoch+1:3d}]"
                    f"  train_base_aux={tr['base_aux_loss']:.6f}"
                    f"  train_mu={tr['mu_loss']:.4e}"
                    f"  train_std={tr['std_loss']:.4e}"
                    f"  val_base_aux={val_base_aux:.6f}"
                    + marker
                )

                if (
                    cfg.early_stop
                    and s1_patience >= cfg.early_stop_patience
                    and epoch + 1 >= cfg.early_stop_min_epochs
                ):
                    print(f"  Stage 1 early stopping at epoch {epoch + 1}.")
                    break

            if os.path.exists(norm_ckpt):
                model.nm.predictor.load_state_dict(
                    torch.load(norm_ckpt, weights_only=True)
                )
                print("  Loaded best base predictor checkpoint.")
            # Restore backbone grads; stage1 does not update last_ckpt
            for p in model.fm.parameters():
                p.requires_grad_(True)

        # ==========================================================
        # lp_pretrain: nm.lp_state_predictor only, route_state_loss
        # ==========================================================
        elif stage_name == "lp_pretrain":
            if cfg.route_stage_epochs <= 0:
                continue
            print(
                f"\n--- lp_pretrain: lp_state_predictor only (route_state_loss)"
                f" (max {cfg.route_stage_epochs} epochs, patience={cfg.route_stage_patience}) ---"
            )
            # Freeze everything except nm.lp_state_predictor
            for p in model.fm.parameters():
                p.requires_grad_(False)
            model.nm.freeze_base_predictor()
            for p in model.nm.lp_state_predictor.parameters():
                p.requires_grad_(True)

            lp_optim = torch.optim.Adam(
                list(model.nm.lp_state_predictor.parameters()),
                lr=cfg.route_stage_lr,
                weight_decay=cfg.route_stage_weight_decay,
            )
            best_lp_val = float("inf")
            lp_patience = 0

            for epoch in range(cfg.route_stage_epochs):
                tr    = _train_epoch_lp_pretrain(model, dataloader.train_loader, lp_optim, cfg)
                val_m = _eval_loader(model, dataloader.val_loader, cfg, metrics, enable_b2sc_eval=enable_val_b2sc, enable_ssc_eval=enable_val_ssc)
                val_rs = val_m["route_state_loss"]

                improved_lp = val_rs < best_lp_val - cfg.early_stop_delta
                marker = "  *" if improved_lp else ""
                if improved_lp:
                    best_lp_val = val_rs
                    lp_patience = 0
                    torch.save(model.state_dict(), lp_pretrain_ckpt)
                else:
                    lp_patience += 1

                print(
                    f"[lp_pretrain epoch {epoch+1:4d}]"
                    f"  train_route_state={tr['route_state_loss']:.6f}"
                    f"  val_route_state={val_rs:.6f}"
                    + marker
                )

                if (
                    cfg.early_stop
                    and lp_patience >= cfg.route_stage_patience
                    and epoch + 1 >= cfg.route_stage_min_epochs
                ):
                    print(f"  lp_pretrain early stopping at epoch {epoch + 1}.")
                    break

            if os.path.exists(lp_pretrain_ckpt):
                model.load_state_dict(torch.load(lp_pretrain_ckpt, weights_only=True))
                last_ckpt = lp_pretrain_ckpt

        # ==========================================================
        # timeapn_pretrain: APN module only (station_optim).
        # fm frozen; APN params trained with sliding_loss_P.
        # ==========================================================
        elif stage_name == "timeapn_pretrain":
            if cfg.timeapn_pre_epoch <= 0:
                continue
            print(
                f"\n--- timeapn_pretrain: APN module only"
                f" (max {cfg.timeapn_pre_epoch} epochs,"
                f" patience={cfg.early_stop_patience}) ---"
            )
            # Freeze fm; unfreeze APN only
            for p in model.fm.parameters():
                p.requires_grad_(False)
            model.nm.freeze_base_predictor()
            model.nm.unfreeze_route_modules()  # unfreezes apn_module

            tapn_optim = torch.optim.Adam(
                model.nm.parameters_route_modules(),
                lr=cfg.timeapn_station_lr,
                weight_decay=cfg.weight_decay,
            )
            best_tapn_val = float("inf")
            tapn_patience = 0

            for epoch in range(cfg.timeapn_pre_epoch):
                tr    = _train_epoch_timeapn_pretrain(
                    model, dataloader.train_loader, tapn_optim, cfg
                )
                val_m = _eval_loader(model, dataloader.val_loader, cfg, metrics, enable_b2sc_eval=enable_val_b2sc, enable_ssc_eval=enable_val_ssc)
                val_rs = val_m["route_state_loss"]

                improved_tapn = val_rs < best_tapn_val - cfg.early_stop_delta
                marker = "  *" if improved_tapn else ""
                if improved_tapn:
                    best_tapn_val = val_rs
                    tapn_patience = 0
                    torch.save(model.state_dict(), timeapn_pretrain_ckpt)
                else:
                    tapn_patience += 1

                print(
                    f"[timeapn_pretrain epoch {epoch+1:4d}]"
                    f"  train_total={tr['route_state_loss']:.6f}"
                    f"  apn_mean={tr['apn_loss_mean']:.4e}"
                    f"  apn_phase={tr['apn_loss_phase']:.4e}"
                    f"  apn_recon={tr['apn_loss_recon']:.4e}"
                    f"  val_route_state={val_rs:.6f}"
                    + marker
                )

                if (
                    cfg.early_stop
                    and tapn_patience >= cfg.early_stop_patience
                    and epoch + 1 >= cfg.early_stop_min_epochs
                ):
                    print(f"  timeapn_pretrain early stopping at epoch {epoch + 1}.")
                    break

            if os.path.exists(timeapn_pretrain_ckpt):
                model.load_state_dict(
                    torch.load(timeapn_pretrain_ckpt, weights_only=True)
                )
                last_ckpt = timeapn_pretrain_ckpt

        # ==========================================================
        # stage2: backbone-only plugin training (nm fully frozen)
        # For TimeAPN: alpha gates must stay at 0 → exact SAN baseline.
        # ==========================================================
        elif stage_name == "stage2":
            _s2_label = (
                "Stage 2: pure SAN backbone training"
                if pure_baseline else
                "Stage 2: backbone training"
            )
            print(
                f"\n--- {_s2_label} (max {cfg.epochs} epochs,"
                f" patience={cfg.early_stop_patience}) ---"
            )

            # All combos: nm fully frozen, fm only
            model.nm.freeze_base_predictor()
            model.nm.freeze_route_modules()
            for p in model.fm.parameters():
                p.requires_grad_(True)
            s2_params = list(model.fm.parameters())
            s2_optim  = torch.optim.Adam(
                s2_params, lr=cfg.lr, weight_decay=cfg.weight_decay
            )

            s2_patience = 0
            _s2_apn_added = False  # track whether APN was merged into s2_optim

            for epoch in range(cfg.epochs):
                # TimeAPN late merge: only when explicitly enabled
                if (
                    is_timeapn
                    and cfg.timeapn_enable_late_merge
                    and not _s2_apn_added
                    and cfg.timeapn_twice_epoch >= 0
                    and epoch == cfg.timeapn_twice_epoch
                ):
                    model.nm.unfreeze_route_modules()
                    _current_lr = s2_optim.param_groups[0]["lr"]
                    s2_optim.add_param_group(
                        {"params": model.nm.parameters_route_modules(), "lr": _current_lr}
                    )
                    _s2_apn_added = True
                    print(
                        f"  [TimeAPN] late_merge at epoch={cfg.timeapn_twice_epoch}:"
                        f" APN added to optimizer (lr={_current_lr:.2e})."
                    )

                if cfg.ssc_enable:
                    tr = _train_epoch_stage2_ssc(model, _ssc_train_loader, s2_optim, cfg)
                else:
                    tr = _train_epoch_stage2(model, dataloader.train_loader, s2_optim, cfg)
                val_m  = _eval_loader(model, dataloader.val_loader,  cfg, metrics, enable_b2sc_eval=enable_val_b2sc,  enable_ssc_eval=enable_val_ssc)
                test_m = _eval_loader(model, dataloader.test_loader, cfg, metrics, enable_b2sc_eval=enable_test_b2sc, enable_ssc_eval=enable_test_ssc, reset_ssc=True)
                val_mse = val_m["mse"]
                val_mae = val_m["mae"]

                improved = val_mse < best_val_mse - cfg.early_stop_delta
                marker   = "  *" if improved else ""
                if improved:
                    best_val_mse  = val_mse
                    best_val_mae  = val_mae
                    best_test_mse = test_m["mse"]
                    best_test_mae = test_m["mae"]
                    best_epoch    = epoch + 1
                    best_stage    = "stage2"
                    s2_patience   = 0
                    torch.save(model.state_dict(), stage2_ckpt)
                else:
                    s2_patience += 1

                print(
                    f"[Stage2 epoch {epoch+1:4d}]"
                    f"  train_task={tr['task_loss']:.6f}"
                    f"  train_base_aux={tr['base_aux_loss']:.6f}"
                    f"  train_route_state={tr['route_state_loss']:.6f}"
                    f"  val_mse={val_mse:.6f}  val_mae={val_mae:.6f}"
                    f"  val_base_aux={val_m['base_aux_loss']:.6f}"
                    f"  val_route_state={val_m['route_state_loss']:.6f}"
                    f"  test_mse={test_m['mse']:.6f}  test_mae={test_m['mae']:.6f}"
                    + marker
                )

                if (
                    cfg.early_stop
                    and s2_patience >= cfg.early_stop_patience
                    and epoch + 1 >= cfg.early_stop_min_epochs
                ):
                    print(f"  Stage 2 early stopping at epoch {epoch + 1}.")
                    break

            if cfg.save_stage2_ckpt and os.path.exists(stage2_ckpt):
                ckpt_dir    = os.path.join(_result_dir(cfg), "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                persist_path = os.path.join(ckpt_dir, f"stage2_s{seed}.pt")
                shutil.copy2(stage2_ckpt, persist_path)
                print(f"  Saved stage2 checkpoint to {persist_path}")
                meta = {
                    "exp_name":               cfg.exp_name,
                    "dataset":                cfg.dataset,
                    "backbone":               cfg.backbone,
                    "pred_len":               cfg.pred_len,
                    "route_path":             cfg.route_path,
                    "route_state":            cfg.route_state,
                    "baseline_mode":          "pure_san" if pure_baseline else "san_route",
                    "seed":                   seed,
                    "b2sc_enable":            cfg.b2sc_enable,
                    "b2sc_apply_on_val":      cfg.b2sc_apply_on_val,
                    "b2sc_apply_on_test":     cfg.b2sc_apply_on_test,
                    "b2sc_recent_weight":     cfg.b2sc_recent_weight,
                    "b2sc_prev_weight":       cfg.b2sc_prev_weight,
                    "b2sc_second_slice_scale": cfg.b2sc_second_slice_scale,
                }
                meta_path = os.path.join(ckpt_dir, f"stage2_s{seed}_meta.json")
                with open(meta_path, "w") as _mf:
                    json.dump(meta, _mf, indent=2)
                print(f"  Saved stage2 meta to {meta_path}")

            if os.path.exists(stage2_ckpt):
                model.load_state_dict(torch.load(stage2_ckpt, weights_only=True))
                last_ckpt = stage2_ckpt

        # ==========================================================
        # stage3: route modules only.
        # stage3: nm.route_modules only; fm + nm.predictor frozen.
        # For generic: route_modules only (unchanged behaviour).
        # ==========================================================
        elif stage_name == "stage3":
            if cfg.route_stage_epochs <= 0:
                continue
            print(
                f"\n--- Stage 3: route-only training (max {cfg.route_stage_epochs} epochs,"
                f" patience={cfg.route_stage_patience}) ---"
            )
            model.nm.freeze_base_predictor()
            for p in model.fm.parameters():
                p.requires_grad_(False)
            model.nm.unfreeze_route_modules()

            stage3_params = model.nm.parameters_route_modules()

            s3_optim = torch.optim.Adam(
                stage3_params,
                lr=cfg.route_stage_lr,
                weight_decay=cfg.route_stage_weight_decay,
            )
            s3_patience = 0
            best_s3_val = float("inf")

            for epoch in range(cfg.route_stage_epochs):
                tr    = _train_epoch_stage3(
                    model, dataloader.train_loader, s3_optim, cfg, stage3_params
                )
                val_m  = _eval_loader(model, dataloader.val_loader,  cfg, metrics, enable_b2sc_eval=enable_val_b2sc,  enable_ssc_eval=enable_val_ssc)
                test_m = _eval_loader(model, dataloader.test_loader, cfg, metrics, enable_b2sc_eval=enable_test_b2sc, enable_ssc_eval=enable_test_ssc, reset_ssc=True)
                val_mse = val_m["mse"]
                val_mae = val_m["mae"]
                route_diag = model.nm.get_route_diagnostics()

                improved_s3 = val_mse < best_s3_val - cfg.early_stop_delta
                marker = "  *" if improved_s3 else ""
                if improved_s3:
                    best_s3_val = val_mse
                    s3_patience = 0
                    torch.save(model.state_dict(), stage3_ckpt)
                    if val_mse < best_val_mse:
                        best_val_mse  = val_mse
                        best_val_mae  = val_mae
                        best_test_mse = test_m["mse"]
                        best_test_mae = test_m["mae"]
                        best_epoch    = epoch + 1
                        best_stage    = "stage3"
                else:
                    s3_patience += 1

                diag_str = "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in route_diag.items()
                    if k not in ("route_path", "route_state")
                )
                print(
                    f"[Stage3 epoch {epoch+1:4d}]"
                    f"  train_task={tr['task_loss']:.6f}"
                    f"  train_route_state={tr['route_state_loss']:.6f}"
                    f"  val_mse={val_mse:.6f}  val_mae={val_mae:.6f}"
                    f"  val_route_state={val_m['route_state_loss']:.6f}"
                    f"  test_mse={test_m['mse']:.6f}  test_mae={test_m['mae']:.6f}"
                    + (f"  [{diag_str}]" if diag_str else "")
                    + marker
                )

                if (
                    cfg.early_stop
                    and s3_patience >= cfg.route_stage_patience
                    and epoch + 1 >= cfg.route_stage_min_epochs
                ):
                    print(f"  Stage 3 early stopping at epoch {epoch + 1}.")
                    break

            if os.path.exists(stage3_ckpt):
                model.load_state_dict(torch.load(stage3_ckpt, weights_only=True))
                last_ckpt = stage3_ckpt

        # ==========================================================
        # joint: fm + route modules finetune (not valid for lp combo)
        # joint: fm + nm.route_modules; nm.predictor frozen.
        # ==========================================================
        elif stage_name == "joint":
            if cfg.joint_finetune_epochs <= 0:
                continue
            print(
                f"\n--- Joint finetune ({cfg.joint_finetune_epochs} epochs,"
                f" lr={cfg.joint_finetune_lr}) ---"
            )
            model.nm.freeze_base_predictor()
            for p in model.fm.parameters():
                p.requires_grad_(True)
            model.nm.unfreeze_route_modules()

            joint_params = (
                list(model.fm.parameters()) + model.nm.parameters_route_modules()
            )

            jf_optim = torch.optim.Adam(
                joint_params, lr=cfg.joint_finetune_lr, weight_decay=cfg.weight_decay
            )

            for epoch in range(cfg.joint_finetune_epochs):
                tr    = _train_epoch_joint(
                    model, dataloader.train_loader, jf_optim, cfg, joint_params
                )
                val_m  = _eval_loader(model, dataloader.val_loader,  cfg, metrics, enable_b2sc_eval=enable_val_b2sc,  enable_ssc_eval=enable_val_ssc)
                test_m = _eval_loader(model, dataloader.test_loader, cfg, metrics, enable_b2sc_eval=enable_test_b2sc, enable_ssc_eval=enable_test_ssc, reset_ssc=True)
                val_mse = val_m["mse"]

                improved = val_mse < best_val_mse - cfg.early_stop_delta
                marker   = "  *" if improved else ""
                if improved:
                    best_val_mse  = val_mse
                    best_val_mae  = val_m["mae"]
                    best_test_mse = test_m["mse"]
                    best_test_mae = test_m["mae"]
                    best_epoch    = epoch + 1
                    best_stage    = "joint"
                    torch.save(model.state_dict(), joint_ckpt)

                route_diag = model.nm.get_route_diagnostics()
                diag_str   = "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in route_diag.items()
                    if k not in ("route_path", "route_state")
                )
                print(
                    f"[Joint epoch {epoch+1:4d}]"
                    f"  train_task={tr['task_loss']:.6f}"
                    f"  val_mse={val_mse:.6f}"
                    f"  test_mse={test_m['mse']:.6f}"
                    + (f"  [{diag_str}]" if diag_str else "")
                    + marker
                )

            if os.path.exists(joint_ckpt):
                model.load_state_dict(torch.load(joint_ckpt, weights_only=True))
                last_ckpt = joint_ckpt

    train_time = time.time() - t0

    # Final route diagnostics (from last eval pass)
    final_route_diag = model.nm.get_route_diagnostics()

    result: dict[str, Any] = {
        "exp_name":        cfg.exp_name,
        "dataset":         cfg.dataset,
        "backbone":        cfg.backbone,
        "seed":            seed,
        "window":          cfg.window,
        "pred_len":        cfg.pred_len,
        "best_val_mse":    best_val_mse,
        "best_val_mae":    best_val_mae,
        "best_test_mse":   best_test_mse,
        "best_test_mae":   best_test_mae,
        "best_epoch":      best_epoch,
        "best_stage":      best_stage,
        "train_time_sec":  round(train_time, 1),
        "route_diagnostics": final_route_diag,
    }

    if cfg.plot_comparison:
        _plot_oracle_pred_comparison(
            model=model,
            loader=dataloader.test_loader,
            cfg=cfg,
            scaler=scaler,
            out_dir=_result_dir(cfg),
            seed=seed,
            enable_b2sc=enable_test_b2sc,
            enable_ssc=enable_test_ssc,
        )
        _plot_states_comparison(
            model=model,
            train_loader=dataloader.train_loader,
            val_loader=dataloader.val_loader,
            test_loader=dataloader.test_loader,
            cfg=cfg,
            out_dir=_result_dir(cfg),
            seed=seed,
        )

    del model, metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result



# ---------------------------------------------------------------------------
# Post-training plot helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_preds_and_targets(
    model: nn.Module,
    loader,
    cfg: "StateRouteConfig",
    n_samples: int,
    enable_b2sc: bool = False,
    enable_ssc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect up to *n_samples* (pred, target) pairs from *loader*.

    Returns tensors ``(preds, targets)`` each shaped ``(N, pred_len, C)``.
    """
    model.eval()
    preds_list:   list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    if enable_ssc and hasattr(model.nm, "set_ssc_mode"):
        model.nm.reset_ssc_state()
        model.nm.set_ssc_mode(True)
        sample_counter = 0
        for batch_x, batch_y, _oy, batch_x_enc, batch_y_enc in loader:
            if len(preds_list) >= n_samples:
                break
            batch_x     = batch_x.to(cfg.device).float()
            batch_y     = batch_y.to(cfg.device).float()
            batch_x_enc = batch_x_enc.to(cfg.device).float()
            batch_y_enc = batch_y_enc.to(cfg.device).float()
            for i in range(batch_x.shape[0]):
                if len(preds_list) >= n_samples:
                    break
                x_i    = batch_x[i:i+1]
                y_i    = batch_y[i:i+1]
                xenc_i = batch_x_enc[i:i+1]
                yenc_i = batch_y_enc[i:i+1]
                sample_start = sample_counter * cfg.ssc_sample_stride
                model.nm.ssc_update_from_current_lookback(x_i, sample_start)
                pred_i = _forward(model, x_i, xenc_i, yenc_i, cfg)
                model.nm.ssc_cache_current_base_prediction(sample_start + cfg.window)
                sample_counter += 1
                preds_list.append(pred_i.cpu())
                targets_list.append(y_i.cpu())
        model.nm.set_ssc_mode(False)
    else:
        if hasattr(model.nm, "set_b2sc_mode"):
            model.nm.set_b2sc_mode(enable_b2sc)
        for batch_x, batch_y, _oy, batch_x_enc, batch_y_enc in loader:
            if len(preds_list) >= n_samples:
                break
            batch_x     = batch_x.to(cfg.device).float()
            batch_y     = batch_y.to(cfg.device).float()
            batch_x_enc = batch_x_enc.to(cfg.device).float()
            batch_y_enc = batch_y_enc.to(cfg.device).float()
            pred = _forward(model, batch_x, batch_x_enc, batch_y_enc, cfg)
            for i in range(pred.shape[0]):
                preds_list.append(pred[i:i+1].cpu())
                targets_list.append(batch_y[i:i+1].cpu())
                if len(preds_list) >= n_samples:
                    break
        if hasattr(model.nm, "set_b2sc_mode"):
            model.nm.set_b2sc_mode(False)

    preds   = torch.cat(preds_list[:n_samples],   dim=0)  # (N, pred_len, C)
    targets = torch.cat(targets_list[:n_samples], dim=0)  # (N, pred_len, C)
    return preds, targets


def _plot_oracle_pred_comparison(
    model: nn.Module,
    loader,
    cfg: "StateRouteConfig",
    scaler,
    out_dir: str,
    seed: int,
    enable_b2sc: bool,
    enable_ssc: bool,
) -> None:
    """Collect test predictions, inverse-transform, and save comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n  = max(1, cfg.plot_n_samples)
    ch = cfg.plot_channel

    preds, targets = _collect_preds_and_targets(
        model, loader, cfg, n_samples=n,
        enable_b2sc=enable_b2sc, enable_ssc=enable_ssc,
    )

    # Inverse-transform: move to the scaler's device
    preds_inv   = scaler.inverse_transform(
        preds.to(cfg.device).float()
    ).cpu().numpy()    # (N, pred_len, C)
    targets_inv = scaler.inverse_transform(
        targets.to(cfg.device).float()
    ).cpu().numpy()    # (N, pred_len, C)

    actual_n  = preds_inv.shape[0]
    actual_ch = min(ch, preds_inv.shape[2] - 1)
    t = np.arange(cfg.pred_len)

    fig, axes = plt.subplots(
        actual_n, 1, figsize=(12, 3 * actual_n), squeeze=False
    )
    for i in range(actual_n):
        ax = axes[i, 0]
        ax.plot(
            t, targets_inv[i, :, actual_ch],
            label="Oracle", color="C0", linewidth=1.2,
        )
        ax.plot(
            t, preds_inv[i, :, actual_ch],
            label="Pred", color="C1", linewidth=1.2, linestyle="--",
        )
        ax.set_ylabel("Value")
        ax.set_title(f"Test sample {i + 1}")
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Forecast horizon step")
    fig.suptitle(
        f"Oracle vs Pred  |  {cfg.dataset}  {cfg.backbone}"
        f"  pred_len={cfg.pred_len}  ch={actual_ch}  seed={seed}",
        fontsize=11,
    )
    fig.tight_layout()

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig_path = os.path.join(plot_dir, f"oracle_vs_pred_s{seed}.png")
    fig.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved oracle vs pred comparison → {fig_path}")


@torch.no_grad()
def _collect_states(
    model: nn.Module,
    loader,
    cfg: "StateRouteConfig",
    enable_b2sc: bool = False,
    enable_ssc: bool = False,
    reset_ssc: bool = True,
) -> tuple:
    """Collect patch-level state stats for ALL batches in *loader*.

    Runs full-batch forward passes and records **one point per batch**
    (mean over the B and P dimensions), so the x-axis matches the number
    of batches rather than individual samples — giving smooth curves
    comparable to the reference norm-stats plot.

    Returns ``(mu_base, std_base, mu_cal, mu_oracle, std_oracle)`` each shaped
    ``(N_batches, C)`` — averaged over the B×P dimensions.
    ``mu_cal`` is SSC-corrected mu (equals mu_base when SSC is off).
    """
    model.eval()
    nm = model.nm
    if not hasattr(nm, "_base_mu_fut_hat") or not hasattr(nm, "_compute_oracle_stats"):
        return None, None, None, None, None

    mu_base_list: list[torch.Tensor] = []
    std_base_list: list[torch.Tensor] = []
    mu_cal_list:  list[torch.Tensor] = []
    mu_ora_list:   list[torch.Tensor] = []
    std_ora_list:  list[torch.Tensor] = []

    if hasattr(nm, "set_b2sc_mode"):
        nm.set_b2sc_mode(enable_b2sc)
    if enable_ssc and hasattr(nm, "set_ssc_mode"):
        if reset_ssc:
            nm.reset_ssc_state()
        nm.set_ssc_mode(True)

    sample_counter = 0

    for batch_x, batch_y, _oy, batch_x_enc, batch_y_enc in loader:
        batch_x     = batch_x.to(cfg.device).float()
        batch_y     = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()
        batch_size  = batch_x.shape[0]

        if enable_ssc:
            # True sequential SSC: per-sample update → forward → cache
            # Record stats per sample so oracle/base/cal are correctly aligned.
            for i in range(batch_size):
                x_i  = batch_x[i:i+1]
                y_i  = batch_y[i:i+1]
                xe_i = batch_x_enc[i:i+1]
                ye_i = batch_y_enc[i:i+1]
                sample_start = sample_counter * cfg.ssc_sample_stride
                nm.ssc_update_from_current_lookback(x_i, sample_start)
                with torch.no_grad():
                    _forward(model, x_i, xe_i, ye_i, cfg)
                nm.ssc_cache_current_base_prediction(sample_start + cfg.window)
                sample_counter += 1

                if nm._base_mu_fut_hat is None or nm._base_std_fut_hat is None:
                    continue
                _, oracle_mu_i, oracle_std_i = nm._compute_oracle_stats(y_i)
                mu_cal_i, _ = nm._apply_ssc_to_patch_stats(
                    nm._base_mu_fut_hat, nm._base_std_fut_hat
                )
                # Average over P dim → (1, C)
                mu_base_list.append(nm._base_mu_fut_hat.mean(dim=1).cpu())
                std_base_list.append(nm._base_std_fut_hat.mean(dim=1).cpu())
                mu_cal_list.append(mu_cal_i.mean(dim=1).cpu())
                mu_ora_list.append(oracle_mu_i.mean(dim=1).cpu())
                std_ora_list.append(oracle_std_i.mean(dim=1).cpu())
            continue  # stats already recorded above; skip the batch-level block
        else:
            with torch.no_grad():
                _forward(model, batch_x, batch_x_enc, batch_y_enc, cfg)
            sample_counter += batch_size

        if nm._base_mu_fut_hat is None or nm._base_std_fut_hat is None:
            continue
        _, oracle_mu, oracle_std = nm._compute_oracle_stats(batch_y)

        mu_cal_patch, std_cal_patch = nm._apply_ssc_to_patch_stats(
            nm._base_mu_fut_hat, nm._base_std_fut_hat
        )

        # Average over P (patch) dim → (B, C)
        mu_base_list.append(nm._base_mu_fut_hat.mean(dim=1).cpu())
        std_base_list.append(nm._base_std_fut_hat.mean(dim=1).cpu())
        mu_cal_list.append(mu_cal_patch.mean(dim=1).cpu())
        mu_ora_list.append(oracle_mu.mean(dim=1).cpu())
        std_ora_list.append(oracle_std.mean(dim=1).cpu())

    if enable_ssc and hasattr(nm, "set_ssc_mode"):
        nm.set_ssc_mode(False)
    if hasattr(nm, "set_b2sc_mode"):
        nm.set_b2sc_mode(False)

    if not mu_base_list:
        return None, None, None, None, None

    return (
        torch.cat(mu_base_list, dim=0),   # (N, C)
        torch.cat(std_base_list, dim=0),  # (N, C)
        torch.cat(mu_cal_list,  dim=0),   # (N, C)
        torch.cat(mu_ora_list,  dim=0),   # (N, C)
        torch.cat(std_ora_list, dim=0),   # (N, C)
    )


def _plot_states_comparison(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    cfg: "StateRouteConfig",
    out_dir: str,
    seed: int,
) -> None:
    """2-panel continuous time-series (train → val → test) of oracle vs
    predicted window mean (top) and std (bottom).  Vertical dashed lines
    mark the split boundaries.

    - train: SSC disabled (no calibration during training)
    - val:   SSC enabled if cfg.ssc_apply_on_val (with fresh reset)
    - test:  SSC enabled if cfg.ssc_apply_on_test (with fresh reset, NOT
             inheriting val state)
    Three lines per panel: oracle (C0), base mean (C1 dashed), calibrated (C2 dashed)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_ssc = cfg.ssc_enable

    split_cfgs = [
        # (loader,       enable_ssc,                  reset_ssc)
        (train_loader,  False,                         True),
        (val_loader,    use_ssc and cfg.ssc_apply_on_val,  True),
        (test_loader,   use_ssc and cfg.ssc_apply_on_test, True),  # always reset — no val carry-over
    ]

    results = []
    for loader, en_ssc, rst in split_cfgs:
        r = _collect_states(model, loader, cfg,
                            enable_b2sc=False, enable_ssc=en_ssc, reset_ssc=rst)
        results.append(r)

    if all(r[0] is None for r in results):
        print("  [Plot] States plot skipped: no patch stats available.")
        return

    ch = cfg.plot_channel

    def _to_np(tensor):
        if tensor is None:
            return np.array([])
        actual_ch = min(ch, tensor.shape[1] - 1)
        return tensor[:, actual_ch].numpy()

    # results[i] = (mu_base, std_base, mu_cal, mu_ora, std_ora)
    segs_mu_ora  = [_to_np(r[3]) for r in results]
    segs_mu_base = [_to_np(r[0]) for r in results]
    segs_mu_cal  = [_to_np(r[2]) for r in results]
    segs_std_ora  = [_to_np(r[4]) for r in results]
    segs_std_base = [_to_np(r[1]) for r in results]

    # Concatenate + track split boundaries
    boundaries = []
    offset = 0
    for seg in segs_mu_ora:
        offset += len(seg)
        boundaries.append(offset)
    boundaries = boundaries[:-1]

    mu_ora_np   = np.concatenate(segs_mu_ora)
    mu_base_np  = np.concatenate(segs_mu_base)
    mu_cal_np   = np.concatenate(segs_mu_cal)
    std_ora_np  = np.concatenate(segs_std_ora)
    std_base_np = np.concatenate(segs_std_base)
    x = np.arange(len(mu_ora_np))

    has_cal = use_ssc and (not np.allclose(mu_base_np, mu_cal_np))

    split_labels = ["val", "test"]
    split_colors = ["C1", "C2"]

    fig, (ax_mu, ax_std) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # -- mean panel --
    ax_mu.plot(x, mu_ora_np,  color="C0", linewidth=1.0, label="oracle mean")
    ax_mu.plot(x, mu_base_np, color="C1", linewidth=1.0, linestyle="--", label="base mean")
    if has_cal:
        ax_mu.plot(x, mu_cal_np, color="C2", linewidth=1.0, linestyle="--", label="calibrated mean")
    for b, lbl, col in zip(boundaries, split_labels, split_colors):
        ax_mu.axvline(x=b, color=col, linestyle="--", linewidth=1.0)
        ax_mu.text(b + 0.5, 1.0, lbl,
                   transform=ax_mu.get_xaxis_transform(),
                   color=col, fontsize=9, va="top", ha="left")
    ax_mu.set_ylabel("Window Mean")
    ax_mu.legend(loc="upper right")
    ax_mu.grid(True, alpha=0.3)

    # -- std panel --
    ax_std.plot(x, std_ora_np,  color="C0", linewidth=1.0, label="oracle std")
    ax_std.plot(x, std_base_np, color="C1", linewidth=1.0, linestyle="--", label="base std")
    for b, lbl, col in zip(boundaries, split_labels, split_colors):
        ax_std.axvline(x=b, color=col, linestyle="--", linewidth=1.0)
        ax_std.text(b + 0.5, 1.0, lbl,
                    transform=ax_std.get_xaxis_transform(),
                    color=col, fontsize=9, va="top", ha="left")
    ax_std.set_ylabel("Window Std")
    ax_std.set_xlabel("Batch index")
    ax_std.legend(loc="upper right")
    ax_std.grid(True, alpha=0.3)

    backbone = getattr(cfg, "backbone", "")
    fig.suptitle(
        f"SAN Predicted vs Oracle Window Stats — {backbone} | {cfg.dataset} | pred={cfg.pred_len}",
        fontsize=12,
    )
    fig.tight_layout()

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig_path = os.path.join(plot_dir, f"states_oracle_vs_pred_s{seed}.png")
    fig.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved states comparison → {fig_path}")


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def _result_dir(cfg: StateRouteConfig) -> str:
    return os.path.join(
        cfg.result_dir,
        cfg.exp_name,
        cfg.dataset,
        cfg.backbone,
        f"pred_len_{cfg.pred_len}",
        cfg.route_path,
        cfg.route_state,
    )


def _write_results(cfg: StateRouteConfig, trials: list[dict]) -> None:
    out_dir = _result_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    trials_path = os.path.join(out_dir, "trials.jsonl")
    with open(trials_path, "w") as f:
        for trial in trials:
            f.write(json.dumps(trial) + "\n")

    scalar_keys = [
        "best_val_mse", "best_val_mae",
        "best_test_mse", "best_test_mae",
        "best_epoch", "train_time_sec",
    ]
    summary: dict[str, Any] = {}
    for key in scalar_keys:
        vals = [t[key] for t in trials if key in t]
        if vals:
            summary[key + "_mean"] = float(np.mean(vals))
            summary[key + "_std"]  = float(np.std(vals))

    # Best stage counts
    stage_counts: dict[str, int] = {}
    for t in trials:
        s = t.get("best_stage", "unknown")
        stage_counts[s] = stage_counts.get(s, 0) + 1
    summary["best_stage_counts"] = stage_counts
    summary["n_trials"] = len(trials)

    # Aggregate route diagnostics (mean over trials for numeric fields)
    all_diags = [t.get("route_diagnostics", {}) for t in trials]
    agg_diag: dict[str, Any] = {}
    if all_diags and all_diags[0]:
        for k in all_diags[0]:
            vals_d = [d[k] for d in all_diags if k in d]
            if vals_d and isinstance(vals_d[0], float):
                agg_diag[k] = float(np.mean(vals_d))
            else:
                agg_diag[k] = vals_d[0] if vals_d else None
    summary["route_diagnostics_mean"] = agg_diag

    _bm = "pure_san" if _is_pure_san_baseline(cfg) else "san_route"
    summary["baseline_mode"] = _bm
    summary["b2sc_enable"]             = cfg.b2sc_enable
    summary["b2sc_apply_on_val"]       = cfg.b2sc_apply_on_val
    summary["b2sc_apply_on_test"]      = cfg.b2sc_apply_on_test
    summary["b2sc_recent_weight"]      = cfg.b2sc_recent_weight
    summary["b2sc_prev_weight"]        = cfg.b2sc_prev_weight
    summary["b2sc_second_slice_scale"] = cfg.b2sc_second_slice_scale
    summary["ssc_enable"]              = cfg.ssc_enable
    summary["ssc_apply_on_val"]        = cfg.ssc_apply_on_val
    summary["ssc_apply_on_test"]       = cfg.ssc_apply_on_test
    summary["ssc_lambda"]              = cfg.ssc_lambda
    summary["ssc_decay_rho"]           = cfg.ssc_decay_rho
    summary["ssc_scale_beta_legacy"]   = cfg.ssc_scale_beta  # legacy / unused since trend-SSC
    summary["ssc_trend_scale"]         = cfg.ssc_trend_scale
    summary["ssc_sample_stride"]       = cfg.ssc_sample_stride

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    cfg_dict = asdict(cfg)
    cfg_dict["baseline_mode"] = _bm
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

    print(f"\n[Results] Written to {out_dir}")
    print(
        f"  test_mse = {summary.get('best_test_mse_mean', float('nan')):.6f}"
        f" \u00b1 {summary.get('best_test_mse_std', float('nan')):.6f}"
    )
    print(
        f"  test_mae = {summary.get('best_test_mae_mean', float('nan')):.6f}"
        f" \u00b1 {summary.get('best_test_mae_std', float('nan')):.6f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> StateRouteConfig:
    parser = argparse.ArgumentParser(
        description="Train SANRouteNorm (stage-order protocol)."
    )
    cfg = StateRouteConfig()
    defaults = asdict(cfg)
    for field, value in defaults.items():
        arg = f"--{field.replace('_', '-')}"
        if isinstance(value, bool):
            parser.add_argument(arg, action="store_true", default=value)
            parser.add_argument(
                f"--no-{field.replace('_', '-')}", dest=field, action="store_false"
            )
        else:
            parser.add_argument(
                arg, type=type(value) if value is not None else str, default=value
            )
    args = parser.parse_args(argv)
    return StateRouteConfig(**vars(args))


def main(argv=None):
    cfg = _parse_args(argv)
    seeds = cfg.seed_list()

    is_timeapn = (cfg.route_path == "timeapn_correction" and cfg.route_state == "timeapn_state")

    _parse_stage_order(cfg)   # validate early before launching

    print(
        f"[Config] exp_name={cfg.exp_name}  dataset={cfg.dataset}"
        f"  backbone={cfg.backbone}  window={cfg.window}  pred_len={cfg.pred_len}"
    )
    print(
        f"[Config] seeds={seeds}  device={cfg.device}"
    )

    stage_order = _parse_stage_order(cfg)
    _mode_str = "pure_san_baseline" if _is_pure_san_baseline(cfg) else "san_route_experiment"
    print(f"[Config] mode={_mode_str}")
    print(
        f"[Config] san_period_len={cfg.san_period_len}"
        f"  san_stride={cfg.san_stride}"
    )
    print(f"[Config] san_pretrain_epochs={cfg.san_pretrain_epochs}")
    print(
        f"[Config] route_path={cfg.route_path}  route_state={cfg.route_state}"
        f"  route_state_loss_scale={cfg.route_state_loss_scale}"
    )
    print(
        f"[Config] route_stage_epochs={cfg.route_stage_epochs}"
        f"  route_stage_lr={cfg.route_stage_lr}"
        f"  route_task_loss_scale={cfg.route_task_loss_scale}"
    )
    print(
        f"[Config] joint_finetune_epochs={cfg.joint_finetune_epochs}"
        f"  route_require_nonoverlap={cfg.route_require_nonoverlap}"
    )
    if is_timeapn:
        print(
            f"[Config] timeapn_pre_epoch={cfg.timeapn_pre_epoch}"
            f"  timeapn_enable_late_merge={cfg.timeapn_enable_late_merge}"
            f"  timeapn_twice_epoch={cfg.timeapn_twice_epoch}"
            f"  timeapn_station_lr={cfg.timeapn_station_lr}"
        )
        print(
            f"[Config] timeapn_j={cfg.timeapn_j}"
            f"  timeapn_wavelet={cfg.timeapn_wavelet}"
            f"  timeapn_dr={cfg.timeapn_dr}"
            f"  timeapn_kernel_len={cfg.timeapn_kernel_len}"
            f"  timeapn_hkernel_len={cfg.timeapn_hkernel_len}"
        )
        print(
            f"[Config] timeapn_pd_model={cfg.timeapn_pd_model}"
            f"  timeapn_pd_ff={cfg.timeapn_pd_ff}"
            f"  timeapn_pe_layers={cfg.timeapn_pe_layers}"
        )
    print(f"[Config] stage_order={stage_order}")
    if cfg.baseline_ckpt_dir:
        print(f"[Config] baseline_ckpt_dir={cfg.baseline_ckpt_dir}  (skip stage1+pretrain+stage2)")
    if cfg.save_stage2_ckpt:
        print(f"[Config] save_stage2_ckpt=True")
    _b2sc_upd = cfg.b2sc_enable
    _ssc_upd  = cfg.ssc_enable
    _eval_upd = (
        "b2sc+ssc" if (_b2sc_upd and _ssc_upd) else
        "b2sc" if _b2sc_upd else
        "ssc"  if _ssc_upd  else
        "disabled"
    )
    print(
        f"[Config] b2sc_enable={cfg.b2sc_enable}"
        f"  b2sc_apply_on_val={cfg.b2sc_apply_on_val}"
        f"  b2sc_apply_on_test={cfg.b2sc_apply_on_test}"
        f"  b2sc_recent_weight={cfg.b2sc_recent_weight}"
        f"  b2sc_prev_weight={cfg.b2sc_prev_weight}"
        f"  b2sc_second_slice_scale={cfg.b2sc_second_slice_scale}"
    )
    print(
        f"[Config] ssc_enable={cfg.ssc_enable}"
        f"  ssc_apply_on_val={cfg.ssc_apply_on_val}"
        f"  ssc_apply_on_test={cfg.ssc_apply_on_test}"
        f"  ssc_lambda={cfg.ssc_lambda}"
        f"  ssc_decay_rho={cfg.ssc_decay_rho}"
        f"  ssc_scale_beta={cfg.ssc_scale_beta}(legacy/unused)"
        f"  ssc_trend_scale={cfg.ssc_trend_scale}"
        f"  ssc_sample_stride={cfg.ssc_sample_stride}"
    )
    print(f"[Config] eval_update={_eval_upd}")

    trials: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"[Seed {seed}]")
            trial = _run_trial(cfg, seed, tmp_dir)
            trials.append(trial)
            print(
                f"[Seed {seed}] done — test_mse={trial['best_test_mse']:.6f}"
                f"  test_mae={trial['best_test_mae']:.6f}"
                f"  best_epoch={trial['best_epoch']}"
                f"  best_stage={trial['best_stage']}"
            )

    _write_results(cfg, trials)


if __name__ == "__main__":
    main()
