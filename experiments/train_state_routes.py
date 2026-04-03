"""train_state_routes.py — Training entry point for SANRouteNorm experiments.

Formal three-stage training protocol
-------------------------------------
  Stage 1: base predictor pretrain
      Optimizer: nm.predictor only
      Loss: nm.compute_base_aux_loss (mu/std prediction loss)
      Epochs: san_pretrain_epochs  (patience-based early stop)

  Stage 2: backbone training
      Optimizer: model.fm (backbone) only; nm fully frozen
      Loss: forecasting task loss (MSE)
      Epochs: up to cfg.epochs  (patience-based early stop)

  Stage 3: route-only training  [skipped when route_path='none' or route_stage_epochs=0]
      Starts from Stage 2 best checkpoint
      Optimizer: nm.route_state_predictor + nm.route_path_impl
      Loss: route_task_loss_scale * task_loss + nm.compute_route_state_loss
      Epochs: route_stage_epochs  (patience-based early stop)

Joint finetune (optional, joint_finetune_epochs > 0):
      Optimizer: backbone + route modules; nm.predictor frozen
      Loss: task loss
      Epochs: joint_finetune_epochs (fixed, no early stop)

Result directory:
  results/state_routes/<exp_name>/<dataset>/<backbone>/pred_len_<pred_len>/
      <route_path>/<route_state>/
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ttn_norm.experiments.train import (
    TrainConfig,
    _build_backbone_kwargs,
    _build_dataloader,
    _build_metrics,
    _move_model_to_device,
    _set_seed,
)
from ttn_norm.models import TTNModel
from ttn_norm.normalizations import SANRouteNorm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StateRouteConfig:
    """Configuration for SANRouteNorm three-stage experiments."""

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
    route_state_loss_scale: float = 0.1   # used inside nm for compute_route_state_loss

    # Stage 3: route-only training
    route_stage_epochs: int = 0           # 0 = skip Stage 3
    route_stage_lr: float = 1.5e-4
    route_stage_weight_decay: float = 5e-4
    route_stage_patience: int = 5
    route_stage_min_epochs: int = 1
    route_task_loss_scale: float = 1.0    # weight of task loss in Stage 3

    # Joint finetune (optional post-Stage-3 step)
    joint_finetune_epochs: int = 0        # 0 = skip
    joint_finetune_lr: float = 1e-5

    # Non-overlap enforcement for route experiments
    route_require_nonoverlap: bool = True

    # Results
    result_dir: str = "./results/state_routes"

    def seed_list(self) -> list[int]:
        return [int(s.strip()) for s in self.seeds.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_san_route_model(cfg: StateRouteConfig, num_features: int) -> TTNModel:
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
    )
    # Validate non-overlap constraint before continuing
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

    for batch_x, batch_y, _origin_y, _batch_x_enc, _batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()

        optim.zero_grad()
        model(batch_x)
        base_aux = model.nm.compute_base_aux_loss(batch_y)
        base_aux.backward()
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
        "mu_loss": float(np.mean(mu_list)) if mu_list else 0.0,
        "std_loss": float(np.mean(std_list)) if std_list else 0.0,
    }


def _train_epoch_stage2(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    cfg: StateRouteConfig,
) -> dict[str, float]:
    """Stage 2: backbone only, task loss. nm is frozen; aux monitored under no_grad."""
    model.train()
    loss_fn = nn.MSELoss()
    task_list: list[float] = []
    base_aux_list: list[float] = []
    route_state_list: list[float] = []

    for batch_x, batch_y, _origin_y, _batch_x_enc, _batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()

        optim.zero_grad()
        pred = model(batch_x)
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
        "task_loss": float(np.mean(task_list)) if task_list else 0.0,
        "base_aux_loss": float(np.mean(base_aux_list)) if base_aux_list else 0.0,
        "route_state_loss": float(np.mean(route_state_list)) if route_state_list else 0.0,
    }


def _train_epoch_stage3(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    cfg: StateRouteConfig,
) -> dict[str, float]:
    """Stage 3: route modules only. Loss = route_task_loss_scale * task + route_state_loss."""
    model.train()
    loss_fn = nn.MSELoss()
    task_list: list[float] = []
    route_state_list: list[float] = []
    total_list: list[float] = []

    for batch_x, batch_y, _origin_y, _batch_x_enc, _batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()

        optim.zero_grad()
        pred = model(batch_x)
        task_loss = loss_fn(pred, batch_y)
        route_state_loss = model.nm.compute_route_state_loss(batch_y)
        loss = (
            cfg.route_task_loss_scale * task_loss
            + model.nm.route_state_loss_scale * route_state_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.nm.parameters_route_modules(), cfg.max_grad_norm
        )
        optim.step()

        task_list.append(float(task_loss.item()))
        route_state_list.append(float(route_state_loss.item()))
        total_list.append(float(loss.item()))

    return {
        "task_loss": float(np.mean(task_list)) if task_list else 0.0,
        "route_state_loss": float(np.mean(route_state_list)) if route_state_list else 0.0,
        "total_loss": float(np.mean(total_list)) if total_list else 0.0,
    }


def _train_epoch_joint(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    cfg: StateRouteConfig,
    joint_params: list,
) -> dict[str, float]:
    """Joint finetune: backbone + route modules, task loss only. nm.predictor frozen."""
    model.train()
    loss_fn = nn.MSELoss()
    task_list: list[float] = []

    for batch_x, batch_y, _origin_y, _batch_x_enc, _batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()

        optim.zero_grad()
        pred = model(batch_x)
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
) -> dict[str, float]:
    model.eval()
    for metric in metrics.values():
        metric.reset()

    loss_fn = nn.MSELoss()
    task_list: list[float] = []
    base_aux_list: list[float] = []
    route_state_list: list[float] = []

    for batch_x, batch_y, _origin_y, _batch_x_enc, _batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()
        pred = model(batch_x)
        task_loss = loss_fn(pred, batch_y)
        task_list.append(float(task_loss.item()))

        model.nm.compute_total_aux_loss(batch_y)
        aux_stats = model.nm.get_last_aux_stats()
        base_aux_list.append(aux_stats["base_aux_loss"])
        route_state_list.append(aux_stats["route_state_loss"])

        if cfg.pred_len == 1:
            B = pred.shape[0]
            pred = pred.contiguous().view(B, -1)
            batch_y = batch_y.contiguous().view(B, -1)
        for metric in metrics.values():
            metric.update(pred, batch_y)

    results = {name: float(m.compute()) for name, m in metrics.items()}
    results.update({
        "task_loss": float(np.mean(task_list)) if task_list else 0.0,
        "base_aux_loss": float(np.mean(base_aux_list)) if base_aux_list else 0.0,
        "route_state_loss": float(np.mean(route_state_list)) if route_state_list else 0.0,
    })
    return results


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def _run_trial(cfg: StateRouteConfig, seed: int, tmp_dir: str) -> dict[str, Any]:
    _set_seed(seed)

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

    print(
        f"[Split] train={dataloader.train_size}, val={dataloader.val_size},"
        f" test={dataloader.test_size}"
    )

    model = _move_model_to_device(_build_san_route_model(cfg, num_features), cfg)
    metrics = _build_metrics(torch.device(cfg.device))

    norm_ckpt = os.path.join(tmp_dir, f"norm_s{seed}.pt")
    stage2_ckpt = os.path.join(tmp_dir, f"stage2_s{seed}.pt")
    stage3_ckpt = os.path.join(tmp_dir, f"stage3_s{seed}.pt")

    t0 = time.time()
    best_val_mse = float("inf")
    best_val_mae = float("inf")
    best_test_mse = float("inf")
    best_test_mae = float("inf")
    best_epoch = 0
    best_stage = "stage2"

    # -------------------------------------------------------------------
    # Stage 1: nm.predictor only (san_pretrain_epochs > 0)
    # -------------------------------------------------------------------
    if cfg.san_pretrain_epochs > 0:
        print(
            f"\n--- Stage 1: base predictor pretrain"
            f" (max {cfg.san_pretrain_epochs} epochs, patience={cfg.early_stop_patience}) ---"
        )
        # Freeze backbone and route modules; only nm.predictor trains
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
            tr = _train_epoch_stage1(model, dataloader.train_loader, s1_optim, cfg)
            val_m = _eval_loader(model, dataloader.val_loader, cfg, metrics)
            val_base_aux = val_m["base_aux_loss"]

            improved = val_base_aux < best_s1_val - cfg.early_stop_delta
            marker = "  *" if improved else ""
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
            model.nm.predictor.load_state_dict(torch.load(norm_ckpt, weights_only=True))
            print("Loaded best base predictor checkpoint for Stage 2.")

        # Restore backbone grads for Stage 2
        for p in model.fm.parameters():
            p.requires_grad_(True)

    # -------------------------------------------------------------------
    # Stage 2: backbone only; nm fully frozen
    # -------------------------------------------------------------------
    model.nm.freeze_base_predictor()
    model.nm.freeze_route_modules()

    s2_optim = torch.optim.Adam(
        model.fm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    print(
        f"\n--- Stage 2: backbone only (max {cfg.epochs} epochs,"
        f" patience={cfg.early_stop_patience}) ---"
    )
    s2_patience = 0

    for epoch in range(cfg.epochs):
        tr = _train_epoch_stage2(model, dataloader.train_loader, s2_optim, cfg)
        val_m = _eval_loader(model, dataloader.val_loader, cfg, metrics)
        test_m = _eval_loader(model, dataloader.test_loader, cfg, metrics)
        val_mse = val_m["mse"]
        val_mae = val_m["mae"]

        improved = val_mse < best_val_mse - cfg.early_stop_delta
        marker = "  *" if improved else ""
        if improved:
            best_val_mse = val_mse
            best_val_mae = val_mae
            best_test_mse = test_m["mse"]
            best_test_mae = test_m["mae"]
            best_epoch = epoch + 1
            best_stage = "stage2"
            s2_patience = 0
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
            print(f"  Early stopping at epoch {epoch + 1}.")
            break

    # -------------------------------------------------------------------
    # Stage 3: route modules only (if applicable)
    # -------------------------------------------------------------------
    if cfg.route_path != "none" and cfg.route_stage_epochs > 0:
        print(
            f"\n--- Stage 3: route-only training (max {cfg.route_stage_epochs} epochs,"
            f" patience={cfg.route_stage_patience}) ---"
        )
        # Reload Stage 2 best checkpoint as starting point
        if os.path.exists(stage2_ckpt):
            model.load_state_dict(torch.load(stage2_ckpt, weights_only=True))

        # nm.predictor frozen, backbone frozen, route modules unfrozen
        model.nm.freeze_base_predictor()
        for p in model.fm.parameters():
            p.requires_grad_(False)
        model.nm.unfreeze_route_modules()

        route_params = model.nm.parameters_route_modules()
        s3_optim = torch.optim.Adam(
            route_params,
            lr=cfg.route_stage_lr,
            weight_decay=cfg.route_stage_weight_decay,
        )
        s3_patience = 0
        best_s3_val = float("inf")

        for epoch in range(cfg.route_stage_epochs):
            tr = _train_epoch_stage3(model, dataloader.train_loader, s3_optim, cfg)
            val_m = _eval_loader(model, dataloader.val_loader, cfg, metrics)
            test_m = _eval_loader(model, dataloader.test_loader, cfg, metrics)
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
                    best_val_mse = val_mse
                    best_val_mae = val_mae
                    best_test_mse = test_m["mse"]
                    best_test_mae = test_m["mae"]
                    best_epoch = epoch + 1
                    best_stage = "stage3"
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

        # Reload Stage 3 best
        if os.path.exists(stage3_ckpt):
            model.load_state_dict(torch.load(stage3_ckpt, weights_only=True))

    # -------------------------------------------------------------------
    # Joint finetune (optional)
    # -------------------------------------------------------------------
    if cfg.joint_finetune_epochs > 0:
        print(f"\n--- Joint finetune ({cfg.joint_finetune_epochs} epochs, lr={cfg.joint_finetune_lr}) ---")
        model.nm.freeze_base_predictor()
        for p in model.fm.parameters():
            p.requires_grad_(True)
        model.nm.unfreeze_route_modules()

        joint_params = list(model.fm.parameters()) + model.nm.parameters_route_modules()
        jf_optim = torch.optim.Adam(
            joint_params, lr=cfg.joint_finetune_lr, weight_decay=cfg.weight_decay
        )

        for epoch in range(cfg.joint_finetune_epochs):
            tr = _train_epoch_joint(model, dataloader.train_loader, jf_optim, cfg, joint_params)
            val_m = _eval_loader(model, dataloader.val_loader, cfg, metrics)
            test_m = _eval_loader(model, dataloader.test_loader, cfg, metrics)
            val_mse = val_m["mse"]

            improved = val_mse < best_val_mse - cfg.early_stop_delta
            marker = "  *" if improved else ""
            if improved:
                best_val_mse = val_mse
                best_val_mae = val_m["mae"]
                best_test_mse = test_m["mse"]
                best_test_mae = test_m["mae"]
                best_epoch = epoch + 1
                best_stage = "joint"

            print(
                f"[Joint epoch {epoch+1:4d}]"
                f"  train_task={tr['task_loss']:.6f}"
                f"  val_mse={val_mse:.6f}"
                f"  test_mse={test_m['mse']:.6f}"
                + marker
            )

    train_time = time.time() - t0

    # Final route diagnostics (from last eval pass)
    final_route_diag = model.nm.get_route_diagnostics()

    result: dict[str, Any] = {
        "exp_name": cfg.exp_name,
        "dataset": cfg.dataset,
        "backbone": cfg.backbone,
        "seed": seed,
        "window": cfg.window,
        "pred_len": cfg.pred_len,
        "best_val_mse": best_val_mse,
        "best_val_mae": best_val_mae,
        "best_test_mse": best_test_mse,
        "best_test_mae": best_test_mae,
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "train_time_sec": round(train_time, 1),
        "route_diagnostics": final_route_diag,
    }

    del model, metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


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
            summary[key + "_std"] = float(np.std(vals))

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

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

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
        description="Train SANRouteNorm (three-stage protocol)."
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
    print(
        f"[Config] exp_name={cfg.exp_name}  dataset={cfg.dataset}"
        f"  backbone={cfg.backbone}  window={cfg.window}  pred_len={cfg.pred_len}"
    )
    print(
        f"[Config] seeds={seeds}  san_period_len={cfg.san_period_len}"
        f"  san_stride={cfg.san_stride}  device={cfg.device}"
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
