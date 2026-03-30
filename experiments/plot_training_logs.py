from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"^Epoch:\s*(?P<epoch>\d+)\s+Training loss:\s*(?P<train_loss>[-+0-9.eE]+)(?:\s*\|\s*task=(?P<task_loss>[-+0-9.eE]+))?"
)
METRIC_LINE_PATTERN = re.compile(r"^(vali_results|test_results):\s*(\{.*\})\s*$")


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float | None = None
    task_loss: float | None = None
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_training_log(log_path: Path) -> list[EpochRecord]:
    records: list[EpochRecord] = []
    current: EpochRecord | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            epoch_match = EPOCH_PATTERN.match(line)
            if epoch_match:
                current = EpochRecord(
                    epoch=int(epoch_match.group("epoch")),
                    train_loss=float(epoch_match.group("train_loss")),
                    task_loss=_safe_float(epoch_match.group("task_loss")),
                )
                records.append(current)
                continue

            metric_match = METRIC_LINE_PATTERN.match(line)
            if metric_match and current is not None:
                split_name = metric_match.group(1)
                metric_dict = ast.literal_eval(metric_match.group(2))
                parsed = {
                    key: float(value)
                    for key, value in metric_dict.items()
                    if isinstance(value, (int, float))
                }
                if split_name == "vali_results":
                    current.val_metrics = parsed
                else:
                    current.test_metrics = parsed

    return records


def _extract_metric(record: EpochRecord, metric_name: str) -> float | None:
    if metric_name == "train_loss":
        return record.train_loss
    if metric_name == "task_loss":
        return record.task_loss
    if metric_name.startswith("val_"):
        return _safe_float(record.val_metrics.get(metric_name[4:]))
    if metric_name.startswith("test_"):
        return _safe_float(record.test_metrics.get(metric_name[5:]))
    return None


def _default_label(log_path: Path) -> str:
    return log_path.stem


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot comparable training curves from multiple TTN training logs."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Paths to training .out logs to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional legend labels aligned with the provided logs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["train_loss", "val_mse", "test_mse", "val_mae", "test_mae"],
        help="Metrics to plot. Supported names include train_loss, task_loss, val_mse, test_mse, val_mae, test_mae, val_rmse, test_rmse, val_mape, test_mape.",
    )
    parser.add_argument(
        "--output",
        default="results/training_log_comparison.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Training Log Comparison",
        help="Figure title.",
    )
    args = parser.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot training logs") from exc

    log_paths = [Path(path) for path in args.logs]
    for log_path in log_paths:
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

    if args.labels is not None and len(args.labels) not in {0, len(log_paths)}:
        raise ValueError("--labels must be omitted or have the same length as logs")

    labels = args.labels if args.labels else [_default_label(path) for path in log_paths]
    parsed_runs: list[tuple[str, list[EpochRecord]]] = []
    for label, log_path in zip(labels, log_paths):
        records = parse_training_log(log_path)
        if not records:
            raise ValueError(f"No epoch records parsed from {log_path}")
        parsed_runs.append((label, records))

    n_metrics = len(args.metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(13, 3.2 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for axis, metric_name in zip(axes, args.metrics):
        plotted_any = False
        for label, records in parsed_runs:
            epochs: list[int] = []
            values: list[float] = []
            for record in records:
                metric_value = _extract_metric(record, metric_name)
                if metric_value is None:
                    continue
                epochs.append(record.epoch)
                values.append(metric_value)
            if not values:
                continue
            axis.plot(epochs, values, linewidth=1.8, label=label)
            plotted_any = True

        axis.set_ylabel(metric_name)
        axis.grid(True, alpha=0.25)
        if plotted_any:
            axis.legend(loc="best", fontsize=9)
        else:
            axis.text(0.5, 0.5, f"No data for {metric_name}", ha="center", va="center")

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    main()