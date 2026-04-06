"""analyze_oracle_lowpass.py — Oracle low-pass upper-bound evaluation.

Loads a trained SANRouteNorm checkpoint and evaluates three denorm modes:
  1. baseline       — normal SAN denorm (no oracle)
  2. replace_base_mean   — oracle low-pass patch mean replaces predicted base mean; std unchanged
  3. add_lowpass_residual — zero-mean oracle low-pass residual added on top of normal denorm

Results are saved to:
  results/oracle_lowpass/<exp_name>/<dataset>/<backbone>/pred_len_<pred_len>/summary.json

Usage example:
  python -m ttn_norm.experiments.analyze_oracle_lowpass \\
      --ckpt-path ./results/state_routes/.../stage2_s1.pt \\
      --exp-name my_san \\
      --dataset ETTh1 \\
      --backbone DLinear \\
      --window 96 --pred-len 96 \\
      --san-period-len 12 \\
      --split-type popular
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OracleLPConfig:
    # Checkpoint to evaluate
    ckpt_path: str = ""

    # Experiment identity (for result path)
    exp_name: str = "oracle_lp"
    dataset: str = "ETTh1"
    backbone: str = "DLinear"

    # Data
    data_path: str = "./data"
    split_type: str = "popular"
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

    # SANRouteNorm geometry (must match checkpoint)
    san_period_len: int = 12
    san_stride: int = 0
    san_sigma_min: float = 1e-3
    san_w_mu: float = 1.0
    san_w_std: float = 1.0
    route_path: str = "none"
    route_state: str = "none"
    route_state_loss_scale: float = 0.1

    # Eval settings
    device: str = "cuda:0"
    seed: int = 1

    # Results
    result_dir: str = "./results/oracle_lowpass"


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_model(cfg: OracleLPConfig, num_features: int, oracle_mode: str) -> TTNModel:
    """Build a SANRouteNorm model with oracle_lowpass_eval set for the given mode."""
    is_oracle = oracle_mode != "baseline"
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
        oracle_lowpass_eval=is_oracle,
        oracle_lowpass_mode=oracle_mode if is_oracle else "replace_base_mean",
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
# Evaluation
# ---------------------------------------------------------------------------

def _forward_base(
    model: TTNModel,
    batch_x: torch.Tensor,
    batch_x_enc: torch.Tensor,
    batch_y_enc: torch.Tensor,
    cfg: OracleLPConfig,
) -> torch.Tensor:
    """Standard forward (no oracle)."""
    dec_inp, dec_inp_enc = None, None
    if model.is_former:
        label_len = min(cfg.label_len, batch_x.size(1))
        dec_inp, dec_inp_enc = _make_dec_inputs(batch_x, batch_y_enc, batch_x_enc, label_len)
    return model(batch_x, batch_x_enc, dec_inp, dec_inp_enc)


def _forward_oracle(
    model: TTNModel,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_x_enc: torch.Tensor,
    batch_y_enc: torch.Tensor,
    cfg: OracleLPConfig,
) -> torch.Tensor:
    """Forward with oracle ground truth passed to denormalize."""
    dec_inp, dec_inp_enc = None, None
    if model.is_former:
        label_len = min(cfg.label_len, batch_x.size(1))
        dec_inp, dec_inp_enc = _make_dec_inputs(batch_x, batch_y_enc, batch_x_enc, label_len)

    # Run normalize via the normal TTNModel path (sets up all caches)
    batch_x_norm, dec_inp = model.normalize(batch_x, dec_inp=dec_inp)

    if model.is_former:
        pred_norm = model.fm(batch_x_norm, batch_x_enc, dec_inp, dec_inp_enc)
    else:
        pred_norm = model.fm(batch_x_norm)

    # Denormalize with oracle
    pred = model.nm.denormalize(pred_norm, y_true_oracle=batch_y)
    return pred


@torch.no_grad()
def _eval_loader(
    model: TTNModel,
    loader,
    cfg: OracleLPConfig,
    metrics: dict,
    oracle: bool,
) -> dict[str, float]:
    model.eval()
    for m in metrics.values():
        m.reset()

    loss_fn = nn.MSELoss()
    task_list: list[float] = []
    diag_mean_mse_list: list[float] = []
    diag_residual_mse_list: list[float] = []

    for batch_x, batch_y, _origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()

        if oracle:
            pred = _forward_oracle(model, batch_x, batch_y, batch_x_enc, batch_y_enc, cfg)
        else:
            pred = _forward_base(model, batch_x, batch_x_enc, batch_y_enc, cfg)

        task_list.append(float(loss_fn(pred, batch_y).item()))

        if cfg.pred_len == 1:
            B = pred.shape[0]
            pred_m = pred.contiguous().view(B, -1)
            batch_y_m = batch_y.contiguous().view(B, -1)
        else:
            pred_m, batch_y_m = pred, batch_y
        for m in metrics.values():
            m.update(pred_m, batch_y_m)

        if oracle:
            diag = model.nm.get_route_diagnostics()
            diag_mean_mse_list.append(diag.get("oracle_lowpass_mean_mse", 0.0))
            diag_residual_mse_list.append(diag.get("oracle_lowpass_residual_mse", 0.0))

    results = {name: float(m.compute()) for name, m in metrics.items()}
    results["task_loss"] = float(np.mean(task_list)) if task_list else 0.0
    if oracle:
        results["oracle_lowpass_mean_mse"] = (
            float(np.mean(diag_mean_mse_list)) if diag_mean_mse_list else 0.0
        )
        results["oracle_lowpass_residual_mse"] = (
            float(np.mean(diag_residual_mse_list)) if diag_residual_mse_list else 0.0
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _result_dir(cfg: OracleLPConfig) -> str:
    return os.path.join(
        cfg.result_dir,
        cfg.exp_name,
        cfg.dataset,
        cfg.backbone,
        f"pred_len_{cfg.pred_len}",
    )


def run(cfg: OracleLPConfig) -> dict[str, Any]:
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)

    tc = TrainConfig(
        dataset_type=cfg.dataset,
        data_path=cfg.data_path,
        device=cfg.device,
        num_worker=cfg.num_worker,
        seed=cfg.seed,
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
    dataloader, dataset, _scaler, split_info = _build_dataloader(tc)
    num_features = dataset.num_features
    metrics = _build_metrics(device)

    print(f"[Split] {split_info}")
    print(f"[Ckpt]  {cfg.ckpt_path}")

    results: dict[str, Any] = {"config": vars(cfg), "split_info": split_info}

    for split_name, loader in [("val", dataloader.val_loader), ("test", dataloader.test_loader)]:
        print(f"\n--- Evaluating split: {split_name} ---")

        # ---- baseline ----
        model_base = _build_model(cfg, num_features, "baseline")
        model_base = _move_model_to_device(model_base, cfg)
        sd = torch.load(cfg.ckpt_path, map_location=cfg.device, weights_only=True)
        missing, unexpected = model_base.load_state_dict(sd, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        base_m = _eval_loader(model_base, loader, cfg, metrics, oracle=False)
        print(
            f"  baseline   mse={base_m['mse']:.6f}  mae={base_m['mae']:.6f}"
        )
        del model_base

        # ---- replace_base_mean ----
        model_rbm = _build_model(cfg, num_features, "replace_base_mean")
        model_rbm = _move_model_to_device(model_rbm, cfg)
        model_rbm.load_state_dict(sd, strict=False)
        rbm_m = _eval_loader(model_rbm, loader, cfg, metrics, oracle=True)
        print(
            f"  replace_base_mean   mse={rbm_m['mse']:.6f}  mae={rbm_m['mae']:.6f}"
            f"  mean_mse={rbm_m.get('oracle_lowpass_mean_mse', 0):.6f}"
        )
        del model_rbm

        # ---- add_lowpass_residual ----
        model_alr = _build_model(cfg, num_features, "add_lowpass_residual")
        model_alr = _move_model_to_device(model_alr, cfg)
        model_alr.load_state_dict(sd, strict=False)
        alr_m = _eval_loader(model_alr, loader, cfg, metrics, oracle=True)
        print(
            f"  add_lowpass_residual   mse={alr_m['mse']:.6f}  mae={alr_m['mae']:.6f}"
            f"  residual_mse={alr_m.get('oracle_lowpass_residual_mse', 0):.6f}"
        )
        del model_alr

        improve_rbm = (base_m["mse"] - rbm_m["mse"]) / (base_m["mse"] + 1e-12)
        improve_alr = (base_m["mse"] - alr_m["mse"]) / (base_m["mse"] + 1e-12)
        print(
            f"\n  Δ replace_base_mean    = {improve_rbm:+.4f} (relative MSE reduction)"
        )
        print(
            f"  Δ add_lowpass_residual = {improve_alr:+.4f} (relative MSE reduction)"
        )

        results[split_name] = {
            "baseline_mse":                base_m["mse"],
            "baseline_mae":                base_m["mae"],
            "replace_base_mean_mse":       rbm_m["mse"],
            "replace_base_mean_mae":       rbm_m["mae"],
            "add_lowpass_residual_mse":    alr_m["mse"],
            "add_lowpass_residual_mae":    alr_m["mae"],
            "improve_replace_base_mean":   improve_rbm,
            "improve_add_lowpass_residual": improve_alr,
            "oracle_lowpass_mean_mse":      rbm_m.get("oracle_lowpass_mean_mse", 0.0),
            "oracle_lowpass_residual_mse":  alr_m.get("oracle_lowpass_residual_mse", 0.0),
        }

    # Write summary (test split at top level for quick scan)
    test = results.get("test", {})
    summary: dict[str, Any] = {
        "baseline_mse":                test.get("baseline_mse"),
        "baseline_mae":                test.get("baseline_mae"),
        "replace_base_mean_mse":       test.get("replace_base_mean_mse"),
        "replace_base_mean_mae":       test.get("replace_base_mean_mae"),
        "add_lowpass_residual_mse":    test.get("add_lowpass_residual_mse"),
        "add_lowpass_residual_mae":    test.get("add_lowpass_residual_mae"),
        "improve_replace_base_mean":   test.get("improve_replace_base_mean"),
        "improve_add_lowpass_residual": test.get("improve_add_lowpass_residual"),
        "val": results.get("val", {}),
        "test": test,
        "config": vars(cfg),
        "split_info": split_info,
    }

    out_dir = _result_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Results] Written to {out_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> OracleLPConfig:
    parser = argparse.ArgumentParser(
        description="Oracle low-pass upper-bound evaluation for SANRouteNorm."
    )
    cfg = OracleLPConfig()
    # Register all fields as CLI args
    for field, value in vars(cfg).items():
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
    return OracleLPConfig(**vars(args))


def main(argv=None):
    cfg = _parse_args(argv)
    if not cfg.ckpt_path:
        raise ValueError("--ckpt-path is required.")
    print(
        f"[Config] exp={cfg.exp_name}  dataset={cfg.dataset}  backbone={cfg.backbone}"
        f"  window={cfg.window}  pred_len={cfg.pred_len}"
    )
    print(
        f"[Config] san_period_len={cfg.san_period_len}  san_stride={cfg.san_stride}"
        f"  route_path={cfg.route_path}  route_state={cfg.route_state}"
    )
    run(cfg)


if __name__ == "__main__":
    main()
