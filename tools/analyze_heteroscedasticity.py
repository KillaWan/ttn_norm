"""analyze_heteroscedasticity.py

Analyse heteroscedasticity strength in raw time-series datasets.

Validations performed
---------------------
A  – Raw variance vs. mean (log-log regression on sliding windows)
B  – Detrended residual variance vs. mean (centered moving-average detrending)
C  – Stability of A/B statistics across window sizes (summary sweep)

Usage
-----
python ttn_norm/tools/analyze_heteroscedasticity.py \\
    --dataset-type ETTh1 \\
    --window-sizes 24 48 96 \\
    --out-dir ./heteroscedasticity_results \\
    --max-scatter-points 20000
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path helpers – reuse the same FAN dataset machinery as the training code
# ---------------------------------------------------------------------------

def _ensure_fan_on_path() -> None:
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    fan_root = os.path.join(root, "FAN")
    if fan_root not in sys.path:
        sys.path.insert(0, fan_root)


# ---------------------------------------------------------------------------
# Dataset / split logic (mirrors ttn_norm/experiments/train.py exactly)
# ---------------------------------------------------------------------------

_ETT_POPULAR_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}

_DATASET_ALIASES = {
    "exchange": "ExchangeRate",
    "exchange_rate": "ExchangeRate",
    "electricity": "Electricity",
    "traffic": "Traffic",
    "weather": "Weather",
}

# Default ratio split parameters (same defaults as training code)
_DEFAULT_TRAIN_RATIO = 0.7
_DEFAULT_VAL_RATIO = 0.2


def _resolve_dataset_name(name: str) -> str:
    return _DATASET_ALIASES.get(name, _DATASET_ALIASES.get(name.lower(), name))


def _load_raw_data(dataset_type: str) -> np.ndarray:
    """Return raw data array (T, C) without any scaling."""
    _ensure_fan_on_path()
    import torch_timeseries.datasets as datasets  # noqa: PLC0415

    resolved = _resolve_dataset_name(dataset_type)
    if not hasattr(datasets, resolved):
        raise ValueError(f"Unknown dataset type: {dataset_type!r}")

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data")
    )
    ds = getattr(datasets, resolved)(root=root)
    data = np.array(ds.data, dtype=np.float64)  # shape (T, C)
    return data


def _get_train_end(dataset_type: str, n_total: int) -> int:
    """Return the exclusive end index of the training segment."""
    resolved = _resolve_dataset_name(dataset_type)
    if resolved in _ETT_POPULAR_DATASETS:
        if resolved.startswith("ETTh"):
            return 12 * 30 * 24          # 8640
        else:                             # ETTm
            return 12 * 30 * 24 * 4      # 34560
    else:
        return int(_DEFAULT_TRAIN_RATIO * n_total)


# ---------------------------------------------------------------------------
# Core statistics – sliding window, log-log fit
# ---------------------------------------------------------------------------

def _sliding_stats_A(
    series: np.ndarray,
    W: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (log_mean, log_var, n_valid) for stride-1 windows using raw var."""
    T = len(series)
    n_windows = T - W + 1
    if n_windows <= 0:
        return np.empty(0), np.empty(0), 0

    # Use cumsum for O(T) computation.
    # Use padded cumsum (prepend 0) so window sum i..i+W-1 = cs[i+W] - cs[i].
    # No views are reused after subtraction, avoiding aliasing.
    cs = np.concatenate(([0.0], np.cumsum(series)))
    cs2 = np.concatenate(([0.0], np.cumsum(series ** 2)))

    win_sum = cs[W:] - cs[:n_windows]      # shape (n_windows,)
    win_sum2 = cs2[W:] - cs2[:n_windows]

    m = win_sum / W
    v = win_sum2 / W - m ** 2  # population variance

    # Keep only positive mean and positive variance
    mask = (m > 0) & (v > 0)
    m_valid = m[mask]
    v_valid = v[mask]

    if len(m_valid) == 0:
        return np.empty(0), np.empty(0), 0

    return np.log(m_valid), np.log(v_valid), int(mask.sum())


def _centered_ma(series: np.ndarray, W: int) -> np.ndarray:
    """Centered moving average with replication padding, same length as input."""
    half = W // 2
    pad_right = W - 1 - half
    padded = np.concatenate([
        np.full(half, series[0]),
        series,
        np.full(pad_right, series[-1]),
    ])
    # padded length = T + W - 1; window i (0-based in original) sums padded[i:i+W]
    cs = np.concatenate(([0.0], np.cumsum(padded)))  # prepend 0 for easy subtraction
    T = len(series)
    return (cs[W : T + W] - cs[:T]) / W  # shape (T,)


def _sliding_stats_B(
    series: np.ndarray,
    W: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (log_mean, log_var_res, n_valid) using detrended residual variance."""
    ma = _centered_ma(series, W)
    residuals = series - ma

    T = len(series)
    n_windows = T - W + 1
    if n_windows <= 0:
        return np.empty(0), np.empty(0), 0

    # Use padded-cumsum for O(T) computation (no aliasing).
    cs = np.concatenate(([0.0], np.cumsum(series)))
    m = (cs[W:] - cs[:n_windows]) / W  # mean of raw series per window

    # Population variance of residuals in each window
    cs_r = np.concatenate(([0.0], np.cumsum(residuals)))
    cs_r2 = np.concatenate(([0.0], np.cumsum(residuals ** 2)))
    m_r = (cs_r[W:] - cs_r[:n_windows]) / W
    e2_r = (cs_r2[W:] - cs_r2[:n_windows]) / W
    v_res = e2_r - m_r ** 2  # population variance of residuals

    mask = (m > 0) & (v_res > 0)
    m_valid = m[mask]
    v_res_valid = v_res[mask]

    if len(m_valid) == 0:
        return np.empty(0), np.empty(0), 0

    return np.log(m_valid), np.log(v_res_valid), int(mask.sum())


def _fit_loglog(log_m: np.ndarray, log_v: np.ndarray) -> tuple[float, float, float]:
    """OLS fit log(v) = a + p * log(m).  Returns (slope, r2, pearson)."""
    if len(log_m) < 2:
        return float("nan"), float("nan"), float("nan")
    # OLS
    n = len(log_m)
    sx = log_m.sum()
    sy = log_v.sum()
    sxx = (log_m ** 2).sum()
    sxy = (log_m * log_v).sum()
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("nan"), float("nan"), float("nan")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_pred = intercept + slope * log_m
    ss_res = ((log_v - y_pred) ** 2).sum()
    ss_tot = ((log_v - log_v.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Pearson
    mx, my = log_m.mean(), log_v.mean()
    num = ((log_m - mx) * (log_v - my)).sum()
    den = np.sqrt(((log_m - mx) ** 2).sum() * ((log_v - my) ** 2).sum())
    pearson = num / den if den > 0 else float("nan")
    return float(slope), float(r2), float(pearson)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _scatter_with_fit(
    log_m: np.ndarray,
    log_v: np.ndarray,
    slope: float,
    intercept: float,
    r2: float,
    max_pts: int,
    title: str,
    out_path: str,
) -> None:
    rng = np.random.default_rng(1)
    if len(log_m) > max_pts:
        idx = rng.choice(len(log_m), max_pts, replace=False)
        lm_plot = log_m[idx]
        lv_plot = log_v[idx]
    else:
        lm_plot = log_m
        lv_plot = log_v

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(lm_plot, lv_plot, s=2, alpha=0.3, color="steelblue", label="data")
    xrange = np.linspace(log_m.min(), log_m.max(), 200)
    # Recompute intercept from full data
    ax.plot(
        xrange,
        intercept + slope * xrange,
        color="tomato",
        linewidth=1.5,
        label=f"fit: slope={slope:.3f}, R²={r2:.3f}",
    )
    ax.set_xlabel("log(mean)")
    ax.set_ylabel("log(var)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _bucket_plot(
    log_m: np.ndarray,
    log_v: np.ndarray,
    title: str,
    out_path: str,
    n_buckets: int = 10,
) -> None:
    df = pd.DataFrame({"lm": log_m, "lv": log_v})
    df["bucket"] = pd.qcut(df["lm"], q=n_buckets, labels=False, duplicates="drop")
    grp = df.groupby("bucket")["lv"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grp["bucket"], grp["lv"], marker="o", linewidth=1.5, color="steelblue")
    ax.set_xlabel("log(mean) bucket (equal-frequency)")
    ax.set_ylabel("mean log(var) in bucket")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-channel, per-window analysis
# ---------------------------------------------------------------------------

def _analyse_channel(
    ch_data: np.ndarray,
    ch_name: str,
    W: int,
    dataset_type: str,
    out_dir: str,
    max_scatter: int,
) -> list[dict]:
    """Run A + B validations for one channel and one window size.

    Returns a list of two row-dicts (one per mode).
    """
    records = []

    # --- A: raw variance ---
    log_m_a, log_v_a, n_a = _sliding_stats_A(ch_data, W)
    if n_a >= 2:
        slope_a, r2_a, pearson_a = _fit_loglog(log_m_a, log_v_a)
        # intercept for plot
        ic_a = (log_v_a.mean() - slope_a * log_m_a.mean()) if not np.isnan(slope_a) else 0.0
    else:
        slope_a = r2_a = pearson_a = float("nan")
        ic_a = 0.0

    records.append({
        "dataset": dataset_type,
        "channel": ch_name,
        "window_size": W,
        "mode": "A_raw",
        "slope": slope_a,
        "r2": r2_a,
        "pearson": pearson_a,
        "num_windows": n_a,
    })

    safe_ch = ch_name.replace("/", "_").replace(" ", "_")
    base_dir_a = os.path.join(out_dir, dataset_type, safe_ch, str(W), "A_raw")

    if n_a >= 2:
        _scatter_with_fit(
            log_m_a, log_v_a, slope_a, ic_a, r2_a,
            max_scatter,
            f"{dataset_type} | {ch_name} | W={W} | A_raw | scatter",
            os.path.join(base_dir_a, "scatter.png"),
        )
        _bucket_plot(
            log_m_a, log_v_a,
            f"{dataset_type} | {ch_name} | W={W} | A_raw | bucket",
            os.path.join(base_dir_a, "bucket.png"),
        )

    # --- B: detrended residual variance ---
    log_m_b, log_v_b, n_b = _sliding_stats_B(ch_data, W)
    if n_b >= 2:
        slope_b, r2_b, pearson_b = _fit_loglog(log_m_b, log_v_b)
        ic_b = (log_v_b.mean() - slope_b * log_m_b.mean()) if not np.isnan(slope_b) else 0.0
    else:
        slope_b = r2_b = pearson_b = float("nan")
        ic_b = 0.0

    records.append({
        "dataset": dataset_type,
        "channel": ch_name,
        "window_size": W,
        "mode": "B_detrended",
        "slope": slope_b,
        "r2": r2_b,
        "pearson": pearson_b,
        "num_windows": n_b,
    })

    base_dir_b = os.path.join(out_dir, dataset_type, safe_ch, str(W), "B_detrended")

    if n_b >= 2:
        _scatter_with_fit(
            log_m_b, log_v_b, slope_b, ic_b, r2_b,
            max_scatter,
            f"{dataset_type} | {ch_name} | W={W} | B_detrended | scatter",
            os.path.join(base_dir_b, "scatter.png"),
        )
        _bucket_plot(
            log_m_b, log_v_b,
            f"{dataset_type} | {ch_name} | W={W} | B_detrended | bucket",
            os.path.join(base_dir_b, "bucket.png"),
        )

    return records


# ---------------------------------------------------------------------------
# C: window-sweep summary
# ---------------------------------------------------------------------------

def _make_summary_plots(
    df_all: pd.DataFrame,
    dataset_type: str,
    window_sizes: list[int],
    out_dir: str,
) -> None:
    """Generate C validation figures and tables."""
    summary_dir = os.path.join(out_dir, dataset_type, "summary", "C_window_sweep")
    os.makedirs(summary_dir, exist_ok=True)

    df_a = df_all[df_all["mode"] == "A_raw"]
    df_b = df_all[df_all["mode"] == "B_detrended"]

    # Per-channel tables
    for ch in df_all["channel"].unique():
        safe_ch = ch.replace("/", "_").replace(" ", "_")
        df_ch_a = df_a[df_a["channel"] == ch][["window_size", "slope", "r2", "pearson"]].set_index("window_size")
        df_ch_b = df_b[df_b["channel"] == ch][["window_size", "slope", "r2", "pearson"]].set_index("window_size")
        ch_dir = os.path.join(summary_dir, "per_channel", safe_ch)
        os.makedirs(ch_dir, exist_ok=True)
        df_ch_a.to_csv(os.path.join(ch_dir, "A_sweep.csv"))
        df_ch_b.to_csv(os.path.join(ch_dir, "B_sweep.csv"))

    # Dataset-level aggregation: mean and median across channels per (window_size, mode)
    agg_rows = []
    for mode, df_m in [("A_raw", df_a), ("B_detrended", df_b)]:
        for W in window_sizes:
            sub = df_m[df_m["window_size"] == W]
            agg_rows.append({
                "dataset": dataset_type,
                "window_size": W,
                "mode": mode,
                "slope_mean": sub["slope"].mean(),
                "slope_median": sub["slope"].median(),
                "r2_mean": sub["r2"].mean(),
                "r2_median": sub["r2"].median(),
                "pearson_mean": sub["pearson"].mean(),
                "pearson_median": sub["pearson"].median(),
            })
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(summary_dir, "dataset_agg.csv"), index=False)

    # Summary plot 1: window size vs A mean slope
    fig, ax = plt.subplots(figsize=(6, 4))
    df_a_agg = df_agg[df_agg["mode"] == "A_raw"].sort_values("window_size")
    ax.plot(df_a_agg["window_size"], df_a_agg["slope_mean"], marker="o", color="steelblue")
    ax.set_xlabel("Window size")
    ax.set_ylabel("Mean slope p (A_raw)")
    ax.set_title(f"{dataset_type} – A: mean slope vs window size")
    fig.tight_layout()
    fig.savefig(os.path.join(summary_dir, "A_mean_slope_vs_window.png"), dpi=100)
    plt.close(fig)

    # Summary plot 2: window size vs B mean slope
    fig, ax = plt.subplots(figsize=(6, 4))
    df_b_agg = df_agg[df_agg["mode"] == "B_detrended"].sort_values("window_size")
    ax.plot(df_b_agg["window_size"], df_b_agg["slope_mean"], marker="o", color="tomato")
    ax.set_xlabel("Window size")
    ax.set_ylabel("Mean slope p_res (B_detrended)")
    ax.set_title(f"{dataset_type} – B: mean slope vs window size")
    fig.tight_layout()
    fig.savefig(os.path.join(summary_dir, "B_mean_slope_vs_window.png"), dpi=100)
    plt.close(fig)

    return df_agg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse heteroscedasticity in a raw time-series dataset."
    )
    parser.add_argument(
        "--dataset-type",
        required=True,
        help="Dataset name, e.g. ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather, Traffic, ExchangeRate",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=[24, 48, 96],
        metavar="W",
        help="One or more sliding-window lengths (default: 24 48 96)",
    )
    parser.add_argument(
        "--out-dir",
        default="./heteroscedasticity_results",
        help="Root output directory (default: ./heteroscedasticity_results)",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=20000,
        metavar="N",
        help="Max scatter-plot points before random downsampling (default: 20000)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_type = args.dataset_type
    window_sizes = sorted(set(args.window_sizes))
    out_dir = args.out_dir
    max_scatter = args.max_scatter_points

    print(f"Loading dataset: {dataset_type}")
    _ensure_fan_on_path()
    import torch_timeseries.datasets as datasets  # noqa: PLC0415

    _resolved = _resolve_dataset_name(dataset_type)
    _root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data")
    )
    if not hasattr(datasets, _resolved):
        raise ValueError(f"Unknown dataset type: {dataset_type!r}")
    _ds_obj = getattr(datasets, _resolved)(root=_root)
    data = np.array(_ds_obj.data, dtype=np.float64)   # (T, C), raw unscaled
    T, C = data.shape
    print(f"  shape: {T} timesteps × {C} channels")

    train_end = _get_train_end(dataset_type, T)
    print(f"  training segment: [0, {train_end}) ({train_end} rows)")

    train_data = data[:train_end]  # (train_end, C)

    # Use actual column names from the dataset CSV (skip the date column)
    _raw_cols = list(_ds_obj.df.columns[1:])  # skip date
    ch_names = [str(c) for c in _raw_cols[:C]]

    all_records: list[dict] = []

    for ch_idx, ch_name in enumerate(ch_names):
        ch_data = train_data[:, ch_idx].astype(np.float64)
        for W in window_sizes:
            rows = _analyse_channel(
                ch_data, ch_name, W, dataset_type, out_dir, max_scatter
            )
            all_records.extend(rows)

        if (ch_idx + 1) % max(1, C // 10) == 0 or ch_idx == C - 1:
            print(f"  processed channel {ch_idx + 1}/{C}")

    df_all = pd.DataFrame(all_records)

    # Save per-channel metrics CSV
    per_ch_path = os.path.join(out_dir, dataset_type, "per_channel_metrics.csv")
    os.makedirs(os.path.dirname(per_ch_path), exist_ok=True)
    df_all.to_csv(per_ch_path, index=False)
    print(f"\nPer-channel metrics saved to: {per_ch_path}")

    # C validation + dataset summary
    df_agg = _make_summary_plots(df_all, dataset_type, window_sizes, out_dir)

    # Dataset-level summary CSV (across channels: mean and median per window+mode)
    ds_summary_path = os.path.join(out_dir, dataset_type, "dataset_summary_metrics.csv")

    # Build dataset_summary_metrics.csv with required fields
    summary_rows = []
    for mode in ("A_raw", "B_detrended"):
        for W in window_sizes:
            sub = df_all[(df_all["mode"] == mode) & (df_all["window_size"] == W)]
            summary_rows.append({
                "dataset": dataset_type,
                "channel": "ALL",
                "window_size": W,
                "mode": mode,
                "slope": sub["slope"].mean(),
                "r2": sub["r2"].mean(),
                "pearson": sub["pearson"].mean(),
                "num_windows": sub["num_windows"].sum(),
            })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(ds_summary_path, index=False)
    print(f"Dataset summary metrics saved to: {ds_summary_path}")

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"SUMMARY  dataset={dataset_type}")
    print("=" * 60)
    print(f"{'W':>6}  {'A slope':>10}  {'B slope':>10}  {'B/A':>8}")
    print("-" * 40)
    for W in window_sizes:
        a_slope = df_summary.loc[
            (df_summary["mode"] == "A_raw") & (df_summary["window_size"] == W), "slope"
        ].values[0]
        b_slope = df_summary.loc[
            (df_summary["mode"] == "B_detrended") & (df_summary["window_size"] == W), "slope"
        ].values[0]
        ratio = b_slope / a_slope if (not np.isnan(a_slope) and a_slope != 0) else float("nan")
        print(f"{W:>6}  {a_slope:>10.4f}  {b_slope:>10.4f}  {ratio:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
