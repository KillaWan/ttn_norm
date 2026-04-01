"""train_state_routes.py — Training entry point for SANRouteNorm experiments.

First version: single experiment, no state/path routing.
Uses SANRouteNorm as the normalization module (a clean single-level SAN base).

Result directory layout:
    results/state_routes/<exp_name>/<dataset>/<backbone>/pred_len_<pred_len>/base/
        trials.jsonl    — one trial per line
        summary.json    — mean/std over trials
        config.json     — full config snapshot

Usage:
    python -m ttn_norm.experiments.train_state_routes \\
        --dataset ETTh1 --backbone DLinear --window 96 --pred-len 96 \\
        --seeds 1,2,3 --exp-name my_exp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Reuse existing dataloader, backbone, and metric utilities from train.py
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
    """Focused config for SANRouteNorm single-experiment runs."""

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

    # Model
    window: int = 96
    pred_len: int = 96
    horizon: int = 1
    label_len: int = 48

    # Training
    device: str = "cuda:0"
    cuda_oom_cpu_fallback: bool = False
    seeds: str = "1"          # comma-separated list of seeds to run
    epochs: int = 1000
    lr: float = 1.5e-4
    weight_decay: float = 5e-4
    max_grad_norm: float = 5.0

    # Early stopping
    early_stop: bool = True
    early_stop_patience: int = 5
    early_stop_min_epochs: int = 1
    early_stop_delta: float = 0.0

    # Aux loss schedule (cosine annealing from scale to min_scale)
    aux_loss_scale: float = 0.2
    aux_loss_min_scale: float = 0.05
    aux_loss_schedule: str = "cosine"   # "none" | "cosine"
    aux_loss_decay_start_epoch: int = 4
    aux_loss_decay_epochs: int = 16

    # SANRouteNorm
    san_period_len: int = 12
    san_stride: int = 0
    san_sigma_min: float = 1e-3
    san_w_mu: float = 1.0
    san_w_std: float = 1.0

    # Route path / state
    route_path: str = "none"
    route_state: str = "none"
    route_state_loss_scale: float = 0.1

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
    # Delegate backbone construction to the existing train.py utility.
    # Build a minimal TrainConfig to satisfy _build_backbone_kwargs signature.
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
# Aux-loss scale schedule
# ---------------------------------------------------------------------------

def _aux_scale(cfg: StateRouteConfig, epoch: int) -> float:
    if cfg.aux_loss_schedule == "none":
        return float(cfg.aux_loss_scale)
    start = cfg.aux_loss_decay_start_epoch
    if epoch < start:
        return float(cfg.aux_loss_scale)
    decay = max(cfg.aux_loss_decay_epochs, 1)
    t = min(epoch - start, decay) / float(decay)
    cos_factor = 0.5 * (1.0 + np.cos(np.pi * t))
    return float(
        cfg.aux_loss_min_scale
        + (cfg.aux_loss_scale - cfg.aux_loss_min_scale) * cos_factor
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    cfg: StateRouteConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_fn = nn.MSELoss()
    aux_scale = _aux_scale(cfg, epoch)
    total_losses: list[float] = []
    task_losses: list[float] = []
    aux_losses: list[float] = []
    mu_losses: list[float] = []
    std_losses: list[float] = []
    route_state_losses: list[float] = []

    for batch_x, batch_y, _origin_y, _batch_x_enc, _batch_y_enc in train_loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()

        optimizer.zero_grad()
        pred = model(batch_x)
        task_loss = loss_fn(pred, batch_y)

        aux_loss = torch.tensor(0.0, device=batch_x.device)
        if hasattr(model.nm, "loss"):
            aux_loss = model.nm.loss(batch_y)

        loss = task_loss + aux_scale * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        total_losses.append(float(loss.item()))
        task_losses.append(float(task_loss.item()))
        aux_losses.append(float(aux_loss.item()))
        if hasattr(model.nm, "get_last_aux_stats"):
            aux_stats = model.nm.get_last_aux_stats()
            mu_losses.append(float(aux_stats.get("mu_loss", 0.0)))
            std_losses.append(float(aux_stats.get("std_loss", 0.0)))
            route_state_losses.append(float(aux_stats.get("route_state_loss", 0.0)))

    return {
        "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "task_loss": float(np.mean(task_losses)) if task_losses else 0.0,
        "aux_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
        "aux_total": float(np.mean(aux_losses)) if aux_losses else 0.0,
        "mu_loss": float(np.mean(mu_losses)) if mu_losses else 0.0,
        "std_loss": float(np.mean(std_losses)) if std_losses else 0.0,
        "route_state_loss": float(np.mean(route_state_losses)) if route_state_losses else 0.0,
        "aux_scale": float(aux_scale),
    }


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
    task_losses: list[float] = []
    aux_losses: list[float] = []
    mu_losses: list[float] = []
    std_losses: list[float] = []
    route_state_losses: list[float] = []

    for batch_x, batch_y, _origin_y, _batch_x_enc, _batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()
        pred = model(batch_x)
        task_loss = loss_fn(pred, batch_y)
        task_losses.append(float(task_loss.item()))

        aux_loss = torch.tensor(0.0, device=batch_x.device)
        if hasattr(model.nm, "loss"):
            aux_loss = model.nm.loss(batch_y)
        aux_losses.append(float(aux_loss.item()))
        if hasattr(model.nm, "get_last_aux_stats"):
            aux_stats = model.nm.get_last_aux_stats()
            mu_losses.append(float(aux_stats.get("mu_loss", 0.0)))
            std_losses.append(float(aux_stats.get("std_loss", 0.0)))
            route_state_losses.append(float(aux_stats.get("route_state_loss", 0.0)))

        if cfg.pred_len == 1:
            B = pred.shape[0]
            pred = pred.contiguous().view(B, -1)
            batch_y = batch_y.contiguous().view(B, -1)
        for metric in metrics.values():
            metric.update(pred, batch_y)

    results = {name: float(m.compute()) for name, m in metrics.items()}
    results.update(
        {
            "task_loss": float(np.mean(task_losses)) if task_losses else 0.0,
            "aux_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
            "aux_total": float(np.mean(aux_losses)) if aux_losses else 0.0,
            "mu_loss": float(np.mean(mu_losses)) if mu_losses else 0.0,
            "std_loss": float(np.mean(std_losses)) if std_losses else 0.0,
            "route_state_loss": float(np.mean(route_state_losses)) if route_state_losses else 0.0,
        }
    )
    return results


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def _run_trial(cfg: StateRouteConfig, seed: int, ckpt_path: str) -> dict[str, Any]:
    _set_seed(seed)

    # Build dataloader via existing train.py utility
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    metrics = _build_metrics(torch.device(cfg.device))

    best_val_mse = float("inf")
    best_val_mae = float("inf")
    best_test_mse = float("inf")
    best_test_mae = float("inf")
    best_epoch = 0
    patience_counter = 0

    t0 = time.time()

    for epoch in range(cfg.epochs):
        train_stats = _train_epoch(model, dataloader.train_loader, optimizer, cfg, epoch)

        val_metrics = _eval_loader(model, dataloader.val_loader, cfg, metrics)
        val_mse = val_metrics["mse"]
        val_mae = val_metrics["mae"]
        test_metrics = _eval_loader(model, dataloader.test_loader, cfg, metrics)
        epoch_test_mse = test_metrics["mse"]
        epoch_test_mae = test_metrics["mae"]

        improved = val_mse < best_val_mse - cfg.early_stop_delta
        if improved:
            best_val_mse = val_mse
            best_val_mae = val_mae
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            best_test_mse = test_metrics["mse"]
            best_test_mae = test_metrics["mae"]
        else:
            patience_counter += 1

        print(
            f"[epoch {epoch+1:4d}]"
            f"  train_total={train_stats['total_loss']:.6f}"
            f"  train_task={train_stats['task_loss']:.6f}"
            f"  train_aux={train_stats['aux_loss']:.6f}"
            f"  train_mu={train_stats['mu_loss']:.4e}"
            f"  train_std={train_stats['std_loss']:.4e}"
            f"  train_route_state={train_stats['route_state_loss']:.4e}"
            f"  val_mse={val_mse:.6f}  val_mae={val_mae:.6f}"
            f"  val_task={val_metrics['task_loss']:.6f}"
            f"  val_aux={val_metrics['aux_loss']:.6f}"
            f"  val_mu={val_metrics['mu_loss']:.4e}"
            f"  val_std={val_metrics['std_loss']:.4e}"
            f"  val_route_state={val_metrics['route_state_loss']:.4e}"
            f"  test_mse={epoch_test_mse:.6f}  test_mae={epoch_test_mae:.6f}"
            f"  test_task={test_metrics['task_loss']:.6f}"
            f"  test_aux={test_metrics['aux_loss']:.6f}"
            f"  test_mu={test_metrics['mu_loss']:.4e}"
            f"  test_std={test_metrics['std_loss']:.4e}"
            f"  test_route_state={test_metrics['route_state_loss']:.4e}"
            + ("  *" if improved else "")
        )

        if (
            cfg.early_stop
            and patience_counter >= cfg.early_stop_patience
            and epoch >= cfg.early_stop_min_epochs
        ):
            print(f"  Early stopping at epoch {epoch + 1}.")
            break

    train_time = time.time() - t0
    result = {
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
        "train_time_sec": round(train_time, 1),
    }

    # Explicitly release GPU memory before the next trial
    del model, optimizer, metrics
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
        f"path_{cfg.route_path}",
        f"state_{cfg.route_state}",
    )


def _write_results(cfg: StateRouteConfig, trials: list[dict]) -> None:
    out_dir = _result_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    # trials.jsonl
    trials_path = os.path.join(out_dir, "trials.jsonl")
    with open(trials_path, "w") as f:
        for trial in trials:
            f.write(json.dumps(trial) + "\n")

    # summary.json
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
    summary["n_trials"] = len(trials)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # config.json
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\n[Results] Written to {out_dir}")
    print(
        f"  test_mse = {summary.get('best_test_mse_mean', float('nan')):.6f}"
        f" ± {summary.get('best_test_mse_std', float('nan')):.6f}"
    )
    print(
        f"  test_mae = {summary.get('best_test_mae_mean', float('nan')):.6f}"
        f" ± {summary.get('best_test_mae_std', float('nan')):.6f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> StateRouteConfig:
    parser = argparse.ArgumentParser(
        description="Train SANRouteNorm (single-level SAN base) on a single experiment."
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
            parser.add_argument(arg, type=type(value) if value is not None else str, default=value)
    args = parser.parse_args(argv)
    return StateRouteConfig(**vars(args))


def main(argv=None):
    cfg = _parse_args(argv)
    seeds = cfg.seed_list()
    print(f"[Config] exp_name={cfg.exp_name}  dataset={cfg.dataset}"
          f"  backbone={cfg.backbone}  window={cfg.window}  pred_len={cfg.pred_len}")
    print(f"[Config] seeds={seeds}  san_period_len={cfg.san_period_len}"
          f"  san_stride={cfg.san_stride}  device={cfg.device}")
    print(f"[Config] route_path={cfg.route_path}  route_state={cfg.route_state}"
          f"  route_state_loss_scale={cfg.route_state_loss_scale}")

    trials: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"[Seed {seed}]")
            ckpt_path = os.path.join(tmp_dir, f"best_seed{seed}.pt")
            trial = _run_trial(cfg, seed, ckpt_path)
            trials.append(trial)
            print(
                f"[Seed {seed}] done — test_mse={trial['best_test_mse']:.6f}"
                f"  test_mae={trial['best_test_mae']:.6f}"
                f"  best_epoch={trial['best_epoch']}"
            )

    _write_results(cfg, trials)


if __name__ == "__main__":
    main()
