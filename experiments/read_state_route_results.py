"""read_state_route_results.py — Aggregate and export state-route experiment results.

Reads trials.jsonl files written by train_state_routes.py, computes mean/std
aggregations, and exports to aggregate.csv and aggregate.json.

Usage:
    python -m ttn_norm.experiments.read_state_route_results \\
        --result-dir ./results/state_routes \\
        --out-dir ./results/state_routes/aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _find_trials_files(result_dir: str) -> list[Path]:
    """Recursively find all trials.jsonl files under result_dir."""
    return sorted(Path(result_dir).rglob("trials.jsonl"))


def _read_trials(path: Path) -> list[dict]:
    trials = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def _load_config(trials_path: Path) -> dict:
    config_path = trials_path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_SCALAR_KEYS = [
    "best_val_mse",
    "best_val_mae",
    "best_test_mse",
    "best_test_mae",
    "best_epoch",
    "train_time_sec",
]

_ID_KEYS = [
    "exp_name",
    "dataset",
    "backbone",
    "window",
    "pred_len",
]


def _aggregate_trials(trials: list[dict]) -> dict[str, Any]:
    if not trials:
        return {}

    agg: dict[str, Any] = {}
    for key in _ID_KEYS:
        vals = [t[key] for t in trials if key in t]
        if vals:
            agg[key] = vals[0]

    agg["n_trials"] = len(trials)

    for key in _SCALAR_KEYS:
        vals = [float(t[key]) for t in trials if key in t]
        if vals:
            agg[key + "_mean"] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
            agg[key + "_min"] = float(np.min(vals))
            agg[key + "_max"] = float(np.max(vals))

    return agg


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _write_aggregate_json(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[Output] aggregate.json -> {out_path}")


def _write_aggregate_csv(rows: list[dict], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Output] aggregate.csv  -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate state-route experiment results into CSV and JSON."
    )
    parser.add_argument(
        "--result-dir",
        default="./results/state_routes",
        help="Root directory containing trials.jsonl files (searched recursively).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for aggregate files. Defaults to result-dir.",
    )
    parser.add_argument(
        "--filter-exp",
        default=None,
        help="Only include entries whose exp_name matches this string (substring).",
    )
    parser.add_argument(
        "--filter-dataset",
        default=None,
        help="Only include entries whose dataset matches this string (substring).",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir or args.result_dir
    trials_files = _find_trials_files(args.result_dir)

    if not trials_files:
        print(f"[Warning] No trials.jsonl found under {args.result_dir}")
        return

    print(f"[Read] Found {len(trials_files)} trials.jsonl file(s) under {args.result_dir}")

    aggregate_rows: list[dict] = []
    total_trials = 0

    for path in trials_files:
        trials = _read_trials(path)
        if not trials:
            continue

        # Optional filters
        if args.filter_exp is not None:
            exp = trials[0].get("exp_name", "")
            if args.filter_exp not in exp:
                continue
        if args.filter_dataset is not None:
            ds = trials[0].get("dataset", "")
            if args.filter_dataset not in ds:
                continue

        agg = _aggregate_trials(trials)
        if agg:
            # Annotate with relative path for traceability
            rel = str(path.relative_to(args.result_dir))
            agg["source"] = rel
            aggregate_rows.append(agg)
            total_trials += len(trials)

    print(f"[Read] Aggregated {total_trials} trial(s) across {len(aggregate_rows)} experiment(s).")

    if not aggregate_rows:
        print("[Warning] No data to write after filtering.")
        return

    # Sort by dataset, then by test_mse_mean
    aggregate_rows.sort(
        key=lambda r: (
            r.get("dataset", ""),
            r.get("backbone", ""),
            r.get("pred_len", 0),
            r.get("best_test_mse_mean", float("inf")),
        )
    )

    json_path = os.path.join(out_dir, "aggregate.json")
    csv_path = os.path.join(out_dir, "aggregate.csv")
    _write_aggregate_json(aggregate_rows, json_path)
    _write_aggregate_csv(aggregate_rows, csv_path)

    # Print summary table to stdout
    print("\n[Summary]")
    header = f"{'dataset':12s} {'backbone':12s} {'pred_len':8s} {'test_mse':10s} {'test_mae':10s} {'n':4s}"
    print(header)
    print("-" * len(header))
    for row in aggregate_rows:
        print(
            f"{row.get('dataset', '?'):12s}"
            f" {row.get('backbone', '?'):12s}"
            f" {row.get('pred_len', '?'):8}"
            f" {row.get('best_test_mse_mean', float('nan')):10.6f}"
            f" {row.get('best_test_mae_mean', float('nan')):10.6f}"
            f" {row.get('n_trials', 0):4d}"
        )


if __name__ == "__main__":
    main()
