from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError

# Ensure repo root on path when running via sbatch scripts
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttn_norm.experiments.train import (
    TrainConfig,
    _build_dataloader,
    _build_backbone_kwargs,
    _build_metrics,
    _set_seed,
    build_model,
)
from ttn_norm.models.local_tf_norm.losses import residual_stationarity_loss


def _ensure_fan_on_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fan_root = os.path.join(root, "FAN")
    if fan_root not in os.sys.path:
        os.sys.path.append(fan_root)


def _build_fan_model(
    cfg: TrainConfig, num_features: int, fan_topk: int, fan_rfft: bool = True
):
    _ensure_fan_on_path()
    from torch_timeseries.models.SCINet import SCINet
    from torch_timeseries.normalizations.FAN import FAN
    from torch_timeseries.norm_experiments.Model import Model

    scinet_kwargs = _build_backbone_kwargs(cfg, num_features, cfg.label_len)
    f_model = SCINet(**scinet_kwargs)
    n_model = FAN(
        seq_len=cfg.window,
        pred_len=cfg.pred_len,
        enc_in=num_features,
        freq_topk=fan_topk,
        rfft=fan_rfft,
    )
    return Model("SCINet", f_model, n_model)


def _fan_main_freq_part(x: torch.Tensor, k: int, rfft: bool = True):
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)
    k_values = torch.topk(xf.abs(), k, dim=1)
    indices = k_values.indices
    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask
    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()
    norm_input = x - x_filtered
    return norm_input, x_filtered


def _gate_stats(g_local: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        mean = float(g_local.mean())
        low = float((g_local < 0.1).float().mean())
        high = float((g_local > 0.9).float().mean())
    return {"mean": mean, "pct_lt_0_1": low, "pct_gt_0_9": high}


def _diag_init(device: torch.device):
    return {
        "recon_mae": [],
        "stationarity": [],
        "pred_mse": [],
    }


def _diag_finalize(buf: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.mean(v)) if v else 0.0 for k, v in buf.items()}


@torch.no_grad()
def evaluate_with_diagnostics(
    model,
    loader,
    cfg: TrainConfig,
    scaler,
    mode: str,
    fan_topk: int | None = None,
    fan_rfft: bool = True,
):
    model.eval()
    metrics = _build_metrics(torch.device(cfg.device))
    for metric in metrics.values():
        metric.reset()

    diag = _diag_init(torch.device(cfg.device))
    gate_stats = {"mean": [], "pct_lt_0_1": [], "pct_gt_0_9": []}

    for batch_x, batch_y, origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()
        origin_y = origin_y.to(cfg.device).float()

        if mode == "localtf":
            residual, state = model.nm.normalize(batch_x, return_state=True)
            pred = model.fm(residual)
            pred = model.nm.denormalize(pred, state=state)

            true_n_tf, true_n_time = model.nm._extract_n_time(batch_y)
            if state.pred_n_time is not None and state.pred_n_time.shape == true_n_time.shape:
                pred_mse = torch.mean((state.pred_n_time - true_n_time) ** 2).item()
                diag["pred_mse"].append(pred_mse)

            recon = residual + state.n_time
            recon_mae = torch.mean(torch.abs(batch_x - recon)).item()
            diag["recon_mae"].append(recon_mae)

            stat = residual_stationarity_loss(residual).item()
            diag["stationarity"].append(stat)

            gstats = _gate_stats(state.g_local)
            for k, v in gstats.items():
                gate_stats[k].append(v)

        elif mode == "fan":
            if fan_topk is None:
                raise ValueError("fan_topk is required for FAN diagnostics")
            residual, _ = model.normalize(batch_x)
            pred_residual = model.fm(residual)
            pred = model.denormalize(pred_residual)

            _, true_main = _fan_main_freq_part(batch_y, fan_topk, rfft=fan_rfft)
            pred_main = model.nm.pred_main_freq_signal
            if pred_main.shape == true_main.shape:
                pred_mse = torch.mean((pred_main - true_main) ** 2).item()
                diag["pred_mse"].append(pred_mse)

            _, x_main = _fan_main_freq_part(batch_x, fan_topk, rfft=fan_rfft)
            recon = residual + x_main
            recon_mae = torch.mean(torch.abs(batch_x - recon)).item()
            diag["recon_mae"].append(recon_mae)

            stat = residual_stationarity_loss(residual).item()
            diag["stationarity"].append(stat)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        true = batch_y
        if cfg.invtrans_loss:
            pred = scaler.inverse_transform(pred)
            true = origin_y

        if cfg.pred_len == 1:
            batch_size = pred.shape[0]
            pred = pred.contiguous().view(batch_size, -1)
            true = true.contiguous().view(batch_size, -1)
        else:
            pred = pred.contiguous()
            true = true.contiguous()

        for metric in metrics.values():
            metric.update(pred, true)

    results = {name: float(metric.compute()) for name, metric in metrics.items()}
    diag_out = _diag_finalize(diag)
    gate_out = (
        {k: float(np.mean(v)) if v else 0.0 for k, v in gate_stats.items()}
        if mode == "localtf"
        else {}
    )
    return results, diag_out, gate_out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", required=True)
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--freq", default=None)
    parser.add_argument("--invtrans-loss", action="store_true")

    parser.add_argument("--localtf-ckpt", required=True)
    parser.add_argument("--fan-ckpt", required=True)
    parser.add_argument("--fan-topk", type=int, default=None)
    parser.add_argument("--fan-rfft", action="store_true", default=True)
    parser.add_argument("--result-dir", default="./results/compare")
    return parser.parse_args()


def main():
    args = _parse_args()
    _set_seed(args.seed)

    cfg = TrainConfig(
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        device=args.device,
        seed=args.seed,
        window=args.window,
        pred_len=args.pred_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        label_len=args.label_len,
        freq=args.freq,
        invtrans_loss=args.invtrans_loss,
        norm_type="localtf",
        backbone_type="SCINet",
    )

    dataloader, dataset, scaler = _build_dataloader(cfg)

    # LocalTF model
    localtf_model = build_model(cfg, dataset.num_features).to(cfg.device)
    localtf_state = torch.load(args.localtf_ckpt, map_location=cfg.device)
    localtf_model.load_state_dict(localtf_state)

    # FAN model
    fan_topk_map = {
        "ETTh1": 4,
        "ETTh2": 3,
        "ETTm1": 11,
        "ETTm2": 5,
        "ExchangeRate": 2,
        "Electricity": 3,
        "Traffic": 30,
        "Weather": 2,
    }
    fan_topk = args.fan_topk if args.fan_topk is not None else fan_topk_map.get(args.dataset_type, 4)
    fan_model = _build_fan_model(cfg, dataset.num_features, fan_topk, fan_rfft=args.fan_rfft).to(cfg.device)
    fan_state = torch.load(args.fan_ckpt, map_location=cfg.device)
    fan_model.load_state_dict(fan_state)

    localtf_metrics, localtf_diag, localtf_gate = evaluate_with_diagnostics(
        localtf_model, dataloader.test_loader, cfg, scaler, mode="localtf"
    )
    fan_metrics, fan_diag, _ = evaluate_with_diagnostics(
        fan_model, dataloader.test_loader, cfg, scaler, mode="fan", fan_topk=fan_topk, fan_rfft=args.fan_rfft
    )

    os.makedirs(args.result_dir, exist_ok=True)
    output = {
        "config": asdict(cfg),
        "localtf": {
            "test_metrics": localtf_metrics,
            "diagnostics": localtf_diag,
            "gate_stats": localtf_gate,
            "checkpoint": args.localtf_ckpt,
        },
        "fan": {
            "test_metrics": fan_metrics,
            "diagnostics": fan_diag,
            "checkpoint": args.fan_ckpt,
            "fan_topk": fan_topk,
        },
    }

    out_path = os.path.join(
        args.result_dir, f"compare_{args.dataset_type}_P{args.pred_len}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
