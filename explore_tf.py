"""TF-domain exploratory analysis for time-series datasets.

Usage examples:
    # ETTh1, random window, default STFT params
    python ttn_norm/explore_tf.py

    # Weather dataset, fixed seed, longer window
    python ttn_norm/explore_tf.py --dataset weather --seed 7 --window 192

    # Show oracle score overlay too
    python ttn_norm/explore_tf.py --dataset ETTh1 --oracle

    # Custom STFT params
    python ttn_norm/explore_tf.py --n-fft 64 --hop 16

    # Save figure without displaying
    python ttn_norm/explore_tf.py --save tf_heatmap.png --no-show
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import stft as scipy_stft

# ---------------------------------------------------------------------------
# Matplotlib backend selection: use non-interactive if DISPLAY is not set
# ---------------------------------------------------------------------------
import matplotlib
if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
_DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "data"
)

DATASETS = {
    "ETTh1":        os.path.join(_DATA_ROOT, "ETTh1",    "ETTh1.csv"),
    "ETTh2":        os.path.join(_DATA_ROOT, "ETTh2",    "ETTh2.csv"),
    "ETTm1":        os.path.join(_DATA_ROOT, "ETTm1",    "ETTm1.csv"),
    "ETTm2":        os.path.join(_DATA_ROOT, "ETTm2",    "ETTm2.csv"),
    "weather":      os.path.join(_DATA_ROOT, "weather",  "weather", "weather.csv"),
    "electricity":  os.path.join(_DATA_ROOT, "electricity", "electricity.csv"),
}


def load_dataset(name: str) -> tuple[np.ndarray, list[str]]:
    """Load a dataset by name.  Returns (data: float32 ndarray (T, C), col_names)."""
    path = DATASETS.get(name)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {path}.\n"
            f"Available: {list(DATASETS.keys())}"
        )
    df = pd.read_csv(path)
    # Drop non-numeric columns (date etc.)
    numeric = df.select_dtypes(include=[np.number])
    return numeric.values.astype(np.float32), list(numeric.columns)


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def describe_all() -> None:
    """Print a summary table for all available datasets."""
    print("\n" + "=" * 72)
    print(f"{'Dataset':<14} {'T':>7} {'C':>5} {'NaN':>6} {'mean':>10} {'std':>10}")
    print("-" * 72)
    for name in DATASETS:
        try:
            arr, cols = load_dataset(name)
            T, C = arr.shape
            nans = int(np.isnan(arr).sum())
            print(
                f"{name:<14} {T:>7} {C:>5} {nans:>6}"
                f" {np.nanmean(arr):>10.3f} {np.nanstd(arr):>10.3f}"
            )
        except FileNotFoundError as e:
            print(f"{name:<14}  (not found: {e})")
    print("=" * 72 + "\n")


def describe_dataset(arr: np.ndarray, col_names: list[str]) -> None:
    """Print per-channel stats for one dataset."""
    T, C = arr.shape
    print(f"\n  Shape : {T} time-steps × {C} channels")
    print(f"  {'Channel':<30} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'NaN':>6}")
    print("  " + "-" * 72)
    for i, cn in enumerate(col_names):
        col = arr[:, i]
        print(
            f"  {cn[:30]:<30} {np.nanmean(col):>10.3f} {np.nanstd(col):>10.3f}"
            f" {np.nanmin(col):>10.3f} {np.nanmax(col):>10.3f} {int(np.isnan(col).sum()):>6}"
        )


# ---------------------------------------------------------------------------
# STFT helpers
# ---------------------------------------------------------------------------

def make_stft_params(n_fft: int, hop: int | None, win: int | None):
    hop = hop if hop is not None else n_fft // 4
    win = win if win is not None else n_fft
    hann_win = np.hanning(win).astype(np.float32)
    return n_fft, hop, win, hann_win


def _scipy_stft(sig: np.ndarray, n_fft: int, hop: int, win_arr: np.ndarray):
    """Wrap scipy.signal.stft to match torch.stft(center=True) convention.

    Returns complex array (F, TT).
    """
    _, _, Zxx = scipy_stft(
        sig,
        nperseg=len(win_arr),
        noverlap=len(win_arr) - hop,
        nfft=n_fft,
        window=win_arr,
        boundary="zeros",   # mirrors torch's center=True zero-padding
        padded=True,
        return_onesided=True,
    )
    return Zxx.astype(np.complex64)   # (F, TT)


def stft_window(
    x: np.ndarray,          # (T, C) float32
    n_fft: int,
    hop: int,
    win: int,
    hann_win: np.ndarray,
) -> np.ndarray:
    """Compute STFT magnitude (dB) for each channel.

    Returns:
        spec: (C, F, TT) float32  log-magnitude spectrogram in dB
    """
    T, C = x.shape
    specs = []
    for c in range(C):
        Xc = _scipy_stft(x[:, c], n_fft, hop, hann_win)   # (F, TT)
        mag = np.abs(Xc) + 1e-8
        spec_db = 20.0 * np.log10(mag)                     # dB
        specs.append(spec_db.astype(np.float32))
    return np.stack(specs, axis=0)                          # (C, F, TT)


def oracle_score(
    x: np.ndarray,          # (T, C) float32
    n_fft: int,
    hop: int,
    win: int,
    hann_win: np.ndarray,
    lambda_p: float = 0.25,
) -> np.ndarray:
    """Compute oracle proxy score per (C, F, TT-1): |ΔlogE| + λ_p |wrap(Δφ)|."""
    T, C = x.shape
    scores = []
    for c in range(C):
        Xc   = _scipy_stft(x[:, c], n_fft, hop, hann_win)  # (F, TT)
        mag  = np.abs(Xc).astype(np.float32)
        logE = np.log(mag * mag + 1e-8)
        phi  = np.angle(Xc).astype(np.float32)

        dlogE  = np.abs(np.diff(logE, axis=-1))             # (F, TT-1)
        dP_raw = np.diff(phi, axis=-1)                      # (F, TT-1)
        dP     = np.abs((dP_raw + math.pi) % (2 * math.pi) - math.pi)

        sc = dlogE + lambda_p * dP
        scores.append(sc.astype(np.float32))
    return np.stack(scores, axis=0)                         # (C, F, TT-1)


def dlogE_window(
    x: np.ndarray,          # (T, C) float32
    n_fft: int,
    hop: int,
    win: int,
    hann_win: np.ndarray,
) -> np.ndarray:
    """Compute per-cell |Δlog(|X|²)| (log-energy temporal difference).

    This is the energy component of the oracle proxy score, free of phase.

    Returns:
        dlogE: (C, F, TT-1) float32
    """
    T, C = x.shape
    out = []
    for c in range(C):
        Xc  = _scipy_stft(x[:, c], n_fft, hop, hann_win)
        mag = np.abs(Xc).astype(np.float32)
        logE = np.log(mag * mag + 1e-8)
        dl   = np.abs(np.diff(logE, axis=-1))              # (F, TT-1)
        out.append(dl)
    return np.stack(out, axis=0)                            # (C, F, TT-1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tf_heatmaps(
    arr_win: np.ndarray,           # (window, C) selected window
    col_names: list[str],
    n_fft: int,
    hop: int,
    win_len: int,
    hann_win: np.ndarray,
    dataset: str,
    start_idx: int,
    show_oracle: bool = False,
    oracle_lambda_p: float = 0.25,
    show_dlogE: bool = False,
    unified_cbar: bool = True,
    save_path: str | None = None,
    no_show: bool = False,
    max_channels: int = 21,
) -> None:
    T, C = arr_win.shape
    C_plot = min(C, max_channels)
    col_names = col_names[:C_plot]
    arr_win = arr_win[:, :C_plot]

    # ---- Compute arrays ----
    specs = stft_window(arr_win, n_fft, hop, win_len, hann_win)  # (C, F, TT)
    n_freq, n_tt = specs.shape[1], specs.shape[2]

    dl_all: np.ndarray | None = None
    if show_dlogE:
        dl_all = dlogE_window(arr_win, n_fft, hop, win_len, hann_win)  # (C, F, TT-1)

    osc: np.ndarray | None = None
    if show_oracle:
        osc = oracle_score(arr_win, n_fft, hop, win_len, hann_win, oracle_lambda_p)

    # ---- Axes ----
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0)[:n_freq]   # normalised [0, 0.5]
    t_axis    = np.arange(n_tt) * hop                      # sample index
    t_diff    = t_axis[:-1] if n_tt > 1 else t_axis        # TT-1 frames

    # ---- Global color limits (across ALL channels) ----
    if unified_cbar:
        spec_vmin = float(np.percentile(specs, 2))
        spec_vmax = float(np.percentile(specs, 98))
        dl_vmin   = 0.0
        dl_vmax   = float(np.percentile(dl_all, 98)) if dl_all is not None else 1.0
    else:
        spec_vmin = spec_vmax = None
        dl_vmin   = dl_vmax   = None

    # ---- Layout ----
    rows_per_ch = 1 + (1 if show_dlogE else 0) + (1 if show_oracle else 0)
    n_sub = 1 + C_plot * rows_per_ch
    hr_ch = 1.2
    height_ratios = [0.8] + [hr_ch] * (n_sub - 1)
    fig = plt.figure(figsize=(max(14, n_tt // 2), max(6, hr_ch * C_plot * rows_per_ch + 2)))
    gs  = gridspec.GridSpec(
        n_sub, 1, figure=fig,
        hspace=0.06,
        height_ratios=height_ratios,
    )
    fig.suptitle(
        f"Dataset: {dataset}  |  window [{start_idx} : {start_idx+T}]  |  "
        f"n_fft={n_fft}, hop={hop}, T={T}, C_shown={C_plot}",
        fontsize=10, y=1.002,
    )

    # ---- Row 0: raw time-series ----
    ax0 = fig.add_subplot(gs[0])
    for ci in range(C_plot):
        col = arr_win[:, ci].copy()
        col_range = col.max() - col.min()
        col_norm  = (col - col.min()) / (col_range + 1e-8)
        ax0.plot(np.arange(T), col_norm + ci, lw=0.7)
    ax0.set_xlim(0, T - 1)
    ax0.set_yticks([])
    ax0.set_ylabel("channels\n(stacked)", fontsize=7)
    ax0.tick_params(labelbottom=False)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # ---- Per-channel TF rows ----
    spec_axes: list = []
    spec_ims:  list = []
    dl_axes:   list = []
    dl_ims:    list = []

    row = 1
    for ci in range(C_plot):
        # --- log-magnitude spectrogram ---
        spec = specs[ci]
        vmin = spec_vmin if unified_cbar else float(np.percentile(spec, 5))
        vmax = spec_vmax if unified_cbar else float(np.percentile(spec, 99))

        ax = fig.add_subplot(gs[row])
        im = ax.imshow(
            spec, origin="lower", aspect="auto",
            extent=[t_axis[0], t_axis[-1], freq_axis[0], freq_axis[-1]],
            cmap="magma", vmin=vmin, vmax=vmax,
        )
        ax.set_ylabel(_short_name(col_names[ci], 16),
                      fontsize=7, rotation=0, labelpad=55, va="center")
        ax.tick_params(labelsize=6, labelbottom=False)
        if not unified_cbar:
            cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.015)
            cb.ax.tick_params(labelsize=5)
            cb.set_label("dB", fontsize=5)
        spec_axes.append(ax)
        spec_ims.append(im)
        row += 1

        # --- |ΔlogE| heatmap ---
        if show_dlogE and dl_all is not None:
            dl = dl_all[ci]
            dv_min = dl_vmin if unified_cbar else 0.0
            dv_max = dl_vmax if unified_cbar else float(np.percentile(dl, 98))

            ax2 = fig.add_subplot(gs[row])
            im2 = ax2.imshow(
                dl, origin="lower", aspect="auto",
                extent=[t_diff[0], t_diff[-1], freq_axis[0], freq_axis[-1]],
                cmap="hot", vmin=dv_min, vmax=dv_max,
            )
            ax2.set_ylabel(f"{_short_name(col_names[ci], 12)}\nΔlogE",
                           fontsize=6, rotation=0, labelpad=55, va="center")
            ax2.tick_params(labelsize=6, labelbottom=False)
            if not unified_cbar:
                cb2 = fig.colorbar(im2, ax=ax2, pad=0.01, fraction=0.015)
                cb2.ax.tick_params(labelsize=5)
                cb2.set_label("|ΔlogE|", fontsize=5)
            dl_axes.append(ax2)
            dl_ims.append(im2)
            row += 1

        # --- oracle proxy score ---
        if show_oracle and osc is not None:
            sc   = osc[ci]
            sc_n = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)

            ax3 = fig.add_subplot(gs[row])
            im3 = ax3.imshow(
                sc_n, origin="lower", aspect="auto",
                extent=[t_diff[0], t_diff[-1], freq_axis[0], freq_axis[-1]],
                cmap="YlOrRd", vmin=0.0, vmax=1.0,
            )
            ax3.set_ylabel(f"{_short_name(col_names[ci], 12)}\noracle",
                           fontsize=6, rotation=0, labelpad=55, va="center")
            ax3.tick_params(labelsize=6, labelbottom=False)
            cb3 = fig.colorbar(im3, ax=ax3, pad=0.01, fraction=0.015)
            cb3.ax.tick_params(labelsize=5)
            cb3.set_label("score\n(norm)", fontsize=5)
            row += 1

    # ---- Shared colorbars (unified mode) ----
    if unified_cbar and spec_axes:
        cb_s = fig.colorbar(spec_ims[-1], ax=spec_axes, pad=0.02, fraction=0.012, shrink=0.9)
        cb_s.ax.tick_params(labelsize=5)
        cb_s.set_label("dB", fontsize=6)

    if unified_cbar and dl_axes:
        cb_d = fig.colorbar(dl_ims[-1], ax=dl_axes, pad=0.02, fraction=0.012, shrink=0.9)
        cb_d.ax.tick_params(labelsize=5)
        cb_d.set_label("|ΔlogE|", fontsize=6)

    # x-label on last subplot
    last_ax = fig.axes[-1]
    last_ax.set_xlabel("sample index (within window)", fontsize=8)
    last_ax.tick_params(labelbottom=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[explore_tf] Figure saved to: {save_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


def _short_name(name: str, maxlen: int) -> str:
    return name if len(name) <= maxlen else name[:maxlen - 1] + "…"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse time-series datasets and visualise STFT heatmaps."
    )
    p.add_argument("--dataset",  default="ETTh1",
                   choices=list(DATASETS.keys()),
                   help="Dataset to visualise (default: ETTh1)")
    p.add_argument("--window",   type=int, default=96,
                   help="Look-back window length in time steps (default: 96)")
    p.add_argument("--seed",     type=int, default=None,
                   help="Random seed for window selection (default: random)")
    p.add_argument("--start",    type=int, default=None,
                   help="Fixed start index (overrides --seed)")
    p.add_argument("--n-fft",    type=int, default=32,
                   help="STFT FFT size (default: 32)")
    p.add_argument("--hop",      type=int, default=None,
                   help="STFT hop length (default: n_fft // 4)")
    p.add_argument("--win-len",  type=int, default=None,
                   help="STFT window length (default: n_fft)")
    p.add_argument("--max-ch",   type=int, default=8,
                   help="Max channels to plot (default: 8)")
    p.add_argument("--oracle",   action="store_true",
                   help="Show oracle proxy-score overlay below each spectrogram")
    p.add_argument("--oracle-lambda-p", type=float, default=0.25,
                   help="Phase weight for oracle score (default: 0.25)")
    p.add_argument("--dlogE",    action="store_true",
                   help="Add |ΔlogE| heatmap rows showing where oracle/trigger fires")
    p.add_argument("--per-ch-cbar", action="store_true",
                   help="Use per-channel colorbar instead of a single unified colorbar")
    p.add_argument("--all-datasets", action="store_true",
                   help="Print summary table for all datasets, then exit")
    p.add_argument("--save",     default=None,
                   help="Path to save the figure (e.g. heatmap.png)")
    p.add_argument("--no-show",  action="store_true",
                   help="Do not call plt.show() (useful in headless mode)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.all_datasets:
        describe_all()
        return

    # ---- Load ----
    print(f"\n[explore_tf] Loading dataset: {args.dataset}")
    arr, col_names = load_dataset(args.dataset)
    T_total, C = arr.shape
    describe_dataset(arr, col_names)

    # ---- Replace NaN with column median ----
    n_nan = int(np.isnan(arr).sum())
    if n_nan > 0:
        print(f"\n  [!] {n_nan} NaN values found — replacing with per-column median.")
        for ci in range(C):
            col = arr[:, ci]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                med = np.nanmedian(col)
                arr[nan_mask, ci] = med

    # ---- Select window ----
    window = args.window
    if window >= T_total:
        window = T_total
        start = 0
    elif args.start is not None:
        start = max(0, min(args.start, T_total - window))
    else:
        rng = np.random.default_rng(args.seed)
        start = int(rng.integers(0, T_total - window))

    arr_win = arr[start : start + window]   # (window, C)

    n_fft   = args.n_fft
    hop     = args.hop if args.hop is not None else n_fft // 4
    win_len = args.win_len if args.win_len is not None else n_fft
    _, _, _, hann_win = make_stft_params(n_fft, hop, win_len)

    n_freq_bins = n_fft // 2 + 1
    n_time_bins = math.ceil(window / hop) + 1  # approx (center=True)

    print(f"\n[explore_tf] Window : [{start} : {start + window}]  ({window} steps)")
    print(f"[explore_tf] STFT   : n_fft={n_fft}, hop={hop}, win={win_len}")
    print(f"[explore_tf] Output : ~{n_freq_bins} freq-bins × ~{n_time_bins} time-frames")
    print(f"[explore_tf] Channels plotted: {min(C, args.max_ch)} / {C}")

    # ---- Detect stationarity cue (log) ----
    # Quick scalar: std of local variance across window (high = non-stationary)
    block = max(window // 8, 4)
    blocks = [arr_win[i : i + block] for i in range(0, window - block + 1, block)]
    block_stds = np.array([b.std(axis=0) for b in blocks])   # (n_blocks, C)
    cv_of_std = (block_stds.std(axis=0) / (block_stds.mean(axis=0) + 1e-8)).mean()
    print(f"[explore_tf] Stationarity proxy (CV of block-std, lower=more stationary): {cv_of_std:.4f}")

    # ---- Save path default ----
    save_path = args.save
    if save_path is None and args.no_show:
        save_path = f"tf_heatmap_{args.dataset}_start{start}.png"
        print(f"[explore_tf] --no-show set without --save; saving to: {save_path}")

    # ---- Plot ----
    plot_tf_heatmaps(
        arr_win=arr_win,
        col_names=col_names,
        n_fft=n_fft,
        hop=hop,
        win_len=win_len,
        hann_win=hann_win,
        dataset=args.dataset,
        start_idx=start,
        show_oracle=args.oracle,
        oracle_lambda_p=args.oracle_lambda_p,
        show_dlogE=args.dlogE,
        unified_cbar=not args.per_ch_cbar,
        save_path=save_path,
        no_show=args.no_show,
        max_channels=args.max_ch,
    )


if __name__ == "__main__":
    main()
