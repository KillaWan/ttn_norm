from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import List


@dataclass
class GridConfig:
    datasets: str = "ETTh1,ETTh2,ETTm1,ETTm2,ExchangeRate,Weather,Electricity,Traffic"
    pred_lens: str = "96,168,336,720"
    backbones: str = "DLinear,Informer,FEDformer,SCINet"
    norm_type: str = "none"
    seeds: str = "1,2,3,4,5"

    data_path: str = "./data"
    device: str = "cuda:0"
    result_dir: str = "./results/grid"
    window: int = 96
    horizon: int = 1
    batch_size: int = 32
    epochs: int = 1000
    gate_weight_decay: float = 2e-3
    predictor_weight_decay: float = 2e-3
    label_len: int = 48
    freq: str | None = None
    early_stop: bool = True
    early_stop_patience: int = 5
    early_stop_min_epochs: int = 1
    early_stop_metric: str = "val_mse"
    val_mse_ema_alpha: float = 1.0
    early_stop_delta: float = 0.0
    aux_loss_schedule: str = "cosine"
    aux_loss_scale: float = 1.0
    aux_loss_min_scale: float = 0.2
    aux_loss_decay_start_epoch: int = 4
    aux_loss_decay_epochs: int = 16
    gate_threshold: float = 0.0
    gate_temperature: float = 1.0
    gate_smooth_weight: float = 0.0
    gate_ratio_weight: float = 0.0
    gate_ratio_target: float = 0.3
    pred_dropout: float = 0.1


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _needs_former_inputs(backbone: str) -> bool:
    return backbone.lower() in {"timesnet", "informer", "autoformer", "fedformer", "koopa"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    defaults = asdict(GridConfig())
    for field, value in defaults.items():
        arg_name = f"--{field.replace('_', '-')}"
        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value)
            parser.add_argument(f"--no-{field.replace('_', '-')}", dest=field, action="store_false")
            continue
        parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args(argv)
    cfg = GridConfig(**vars(args))

    datasets = _parse_csv(cfg.datasets)
    pred_lens = _parse_int_csv(cfg.pred_lens)
    backbones = _parse_csv(cfg.backbones)
    seeds = _parse_int_csv(cfg.seeds)

    os.makedirs(cfg.result_dir, exist_ok=True)
    script_path = os.path.join(os.path.dirname(__file__), "train.py")

    summary = {"runs": {}, "averages": {}}

    for dataset in datasets:
        for pred_len in pred_lens:
            for backbone in backbones:
                metrics_per_seed = []
                for seed in seeds:
                    run_name = f"{dataset}_P{pred_len}_{backbone}_{cfg.norm_type}_seed{seed}"
                    cmd = [
                        "python",
                        script_path,
                        "--dataset-type",
                        dataset,
                        "--data-path",
                        cfg.data_path,
                        "--device",
                        cfg.device,
                        "--result-dir",
                        cfg.result_dir,
                        "--run-name",
                        run_name,
                        "--backbone-type",
                        backbone,
                        "--norm-type",
                        cfg.norm_type,
                        "--baseline-subdir",
                        "",
                        "--window",
                        str(cfg.window),
                        "--pred-len",
                        str(pred_len),
                        "--horizon",
                        str(cfg.horizon),
                        "--batch-size",
                        str(cfg.batch_size),
                        "--epochs",
                        str(cfg.epochs),
                        "--gate-weight-decay",
                        str(cfg.gate_weight_decay),
                        "--predictor-weight-decay",
                        str(cfg.predictor_weight_decay),
                        "--seed",
                        str(seed),
                        "--label-len",
                        str(cfg.label_len),
                    ]
                    if cfg.freq:
                        cmd.extend(["--freq", cfg.freq])
                    cmd.extend(
                        [
                            "--gate-threshold",
                            str(cfg.gate_threshold),
                            "--gate-temperature",
                            str(cfg.gate_temperature),
                            "--gate-smooth-weight",
                            str(cfg.gate_smooth_weight),
                            "--gate-ratio-weight",
                            str(cfg.gate_ratio_weight),
                            "--gate-ratio-target",
                            str(cfg.gate_ratio_target),
                            "--pred-dropout",
                            str(cfg.pred_dropout),
                        ]
                    )
                    if cfg.early_stop:
                        cmd.extend(
                            [
                                "--early-stop",
                                "--early-stop-patience",
                                str(cfg.early_stop_patience),
                                "--early-stop-min-epochs",
                                str(cfg.early_stop_min_epochs),
                                "--early-stop-metric",
                                cfg.early_stop_metric,
                                "--val-mse-ema-alpha",
                                str(cfg.val_mse_ema_alpha),
                                "--early-stop-delta",
                                str(cfg.early_stop_delta),
                                "--aux-loss-schedule",
                                cfg.aux_loss_schedule,
                                "--aux-loss-scale",
                                str(cfg.aux_loss_scale),
                                "--aux-loss-min-scale",
                                str(cfg.aux_loss_min_scale),
                                "--aux-loss-decay-start-epoch",
                                str(cfg.aux_loss_decay_start_epoch),
                                "--aux-loss-decay-epochs",
                                str(cfg.aux_loss_decay_epochs),
                            ]
                        )
                    if _needs_former_inputs(backbone):
                        cmd.append("--force-former")

                    print(f"[Grid] Running: {run_name}")
                    try:
                        subprocess.run(cmd, check=True)
                        result_path = os.path.join(cfg.result_dir, f"{run_name}.json")
                        with open(result_path, "r", encoding="utf-8") as f:
                            result = json.load(f)
                        summary["runs"][run_name] = result
                        metrics_per_seed.append(result.get("test_metrics", {}))
                    except subprocess.CalledProcessError as exc:
                        summary["runs"][run_name] = {
                            "run_name": run_name,
                            "dataset": dataset,
                            "backbone": backbone,
                            "norm_type": cfg.norm_type,
                            "seed": seed,
                            "error": str(exc),
                        }

                if metrics_per_seed:
                    keys = ["mse", "mae", "rmse", "mape"]
                    agg = {}
                    for key in keys:
                        vals = [m.get(key) for m in metrics_per_seed if key in m]
                        if vals:
                            mean = sum(vals) / len(vals)
                            var = sum((v - mean) ** 2 for v in vals) / len(vals)
                            agg[key] = {"mean": mean, "std": var ** 0.5}
                    summary["averages"][f"{dataset}_P{pred_len}_{backbone}_{cfg.norm_type}"] = agg

    summary_path = os.path.join(cfg.result_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
