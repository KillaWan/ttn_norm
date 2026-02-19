from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import List

from ttn_norm.experiments.train import TrainConfig


@dataclass
class BaselineConfig:
    dataset_type: str = "ETTh1"
    data_path: str = "./data"
    device: str = "cuda:0"
    result_dir: str = "./results/baselines"
    norm_type: str = "none"
    backbones: str = "DLinear,TimesNet,Informer,Autoformer,FEDformer"

    window: int = 96
    pred_len: int = 96
    horizon: int = 1
    batch_size: int = 32
    epochs: int = 100
    seeds: str = "1,2,3,4,5"
    freq: str | None = None
    label_len: int = 48


def _parse_backbones(backbones: str) -> List[str]:
    return [name.strip() for name in backbones.split(",") if name.strip()]

def _parse_seeds(seeds: str) -> List[int]:
    return [int(s.strip()) for s in seeds.split(",") if s.strip()]


def _needs_former_inputs(backbone: str) -> bool:
    return backbone.lower() in {"timesnet", "informer", "autoformer", "fedformer", "koopa"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    defaults = asdict(BaselineConfig())
    for field, value in defaults.items():
        arg_name = f"--{field.replace('_', '-')}"
        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value)
            parser.add_argument(f"--no-{field.replace('_', '-')}", dest=field, action="store_false")
            continue
        arg_type = type(value)
        parser.add_argument(arg_name, type=arg_type, default=value)
    args = parser.parse_args(argv)
    cfg = BaselineConfig(**vars(args))

    os.makedirs(cfg.result_dir, exist_ok=True)
    backbones = _parse_backbones(cfg.backbones)
    seeds = _parse_seeds(cfg.seeds)
    script_path = os.path.join(os.path.dirname(__file__), "train.py")

    summary = {"runs": {}, "averages": {}}
    for backbone in backbones:
        metrics_per_seed = []
        for seed in seeds:
            run_name = f"{cfg.dataset_type}_{backbone}_{cfg.norm_type}_seed{seed}"
            cmd = [
                "python",
                script_path,
                "--dataset-type",
                cfg.dataset_type,
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
                "--window",
                str(cfg.window),
                "--pred-len",
                str(cfg.pred_len),
                "--horizon",
                str(cfg.horizon),
                "--batch-size",
                str(cfg.batch_size),
                "--epochs",
                str(cfg.epochs),
                "--seed",
                str(seed),
                "--label-len",
                str(cfg.label_len),
            ]
            if cfg.freq:
                cmd.extend(["--freq", cfg.freq])
            if _needs_former_inputs(backbone):
                cmd.append("--force-former")

            print(f"[Baseline] Running: {run_name}")
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
                    "dataset": cfg.dataset_type,
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
            summary["averages"][f"{cfg.dataset_type}_{backbone}_{cfg.norm_type}"] = agg

    summary_path = os.path.join(cfg.result_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
