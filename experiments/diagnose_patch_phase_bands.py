"""diagnose_patch_phase_bands.py — Offline diagnostic for patch-wise phase band coherence.

Reads data only. Does NOT train, modify checkpoints, or alter any model weights.
Does NOT affect the default patch-wise norm training behaviour.

Usage example
-------------
python -m ttn_norm.experiments.diagnose_patch_phase_bands \
    --dataset-type ETTh1 \
    --window 96 \
    --pred-len 96 \
    --san-period-len 12 \
    --split test \
    --output phase_band_report.json

For popular (ETT) split:
    --split-type popular

Frequency band definitions (default):
    low  = bins [1 .. low_k]
    mid  = bins [low_k+1 .. mid_end]
    high = bins [mid_end+1 .. W//2]
Any band that ends up with fewer than 2 bins is reported as NaN / invalid.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# Ensure repo root is importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ttn_norm.experiments.train import _build_dataloader
from ttn_norm.analysis.phase_band_coherence import (
    patch_mean_center,
    compute_phase_rotation,
    band_coherence_score,
    phase_only_oracle_gain,
    interpret_band_result,
)


# ---------------------------------------------------------------------------
# Minimal config object compatible with _build_dataloader
# ---------------------------------------------------------------------------

@dataclass
class _DiagCfg:
    """Minimal config that satisfies all fields read by _build_dataloader."""
    dataset_type: str = "ETTh1"
    data_path: str = "./data"
    device: str = "cpu"
    num_worker: int = 0
    window: int = 96
    pred_len: int = 96
    horizon: int = 1
    batch_size: int = 32
    freq: str = "h"
    scaler_type: str = "StandarScaler"
    scale_in_train: bool = False
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    split_type: str = "ratio"
    # Not wavband → no extra context patches
    norm_type: str = "san_route"
    wav_ctx_patches: int = 0
    wav_patch_len: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> Optional[float]:
    """Convert tensor scalar or Python float to JSON-safe float (None for NaN)."""
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) or math.isinf(f) else f


def _nanquantile(t: torch.Tensor, q: float) -> Optional[float]:
    """Compute quantile of a 1-D tensor, ignoring NaN values."""
    valid = t[~torch.isnan(t)]
    if valid.numel() == 0:
        return None
    return float(torch.quantile(valid.float(), q).item())


def _nanmedian(t: torch.Tensor) -> Optional[float]:
    return _nanquantile(t, 0.5)


def _nanmean(t: torch.Tensor) -> Optional[float]:
    valid = t[~torch.isnan(t)]
    if valid.numel() == 0:
        return None
    return float(valid.float().mean().item())


def _make_band_bins(low_k: int, mid_end: int, max_bin: int) -> dict[str, list[int]]:
    """Build low / mid / high bin lists, clamped to [1, max_bin]."""
    low_k   = min(low_k,  max_bin)
    mid_end = min(mid_end, max_bin)

    low_bins  = list(range(1,         low_k + 1))
    mid_bins  = list(range(low_k + 1, mid_end + 1))
    high_bins = list(range(mid_end + 1, max_bin + 1))

    return {"low": low_bins, "mid": mid_bins, "high": high_bins}


def _compute_band_stats(
    energy_ratios:  torch.Tensor,  # (N,) all samples
    coherences:     torch.Tensor,  # (N,) NaN for invalid
    tau_bests:      torch.Tensor,  # (N,) NaN for invalid
    oracle_gains:   torch.Tensor,  # (N,) NaN for invalid
    mse_bases:      torch.Tensor,  # (N,) NaN for invalid
    mse_taus:       torch.Tensor,  # (N,) NaN for invalid
    n_total:        int,
) -> dict:
    """Aggregate per-sample tensors into band statistics."""
    n_valid = int((~torch.isnan(coherences)).sum().item())
    valid_ratio = n_valid / max(n_total, 1)

    stats = {
        "energy_ratio_mean":   _safe_float(_nanmean(energy_ratios)),
        "energy_ratio_median": _safe_float(_nanmedian(energy_ratios)),
        "valid_ratio":         valid_ratio,
        "coherence_median":    _safe_float(_nanmedian(coherences)),
        "coherence_q25":       _safe_float(_nanquantile(coherences, 0.25)),
        "coherence_q75":       _safe_float(_nanquantile(coherences, 0.75)),
        "tau_best_median":     _safe_float(_nanmedian(tau_bests)),
        "tau_best_q25":        _safe_float(_nanquantile(tau_bests, 0.25)),
        "tau_best_q75":        _safe_float(_nanquantile(tau_bests, 0.75)),
        "oracle_gain_median":  _safe_float(_nanmedian(oracle_gains)),
        "oracle_gain_q25":     _safe_float(_nanquantile(oracle_gains, 0.25)),
        "oracle_gain_q75":     _safe_float(_nanquantile(oracle_gains, 0.75)),
        "oracle_mse_base_mean": _safe_float(_nanmean(mse_bases)),
        "oracle_mse_tau_mean":  _safe_float(_nanmean(mse_taus)),
    }
    stats["label"] = interpret_band_result(stats)
    return stats


# ---------------------------------------------------------------------------
# Core diagnostic loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_diagnostic(
    loader,
    W: int,
    pred_len: int,
    band_bins: dict[str, list[int]],
    tau_grid: torch.Tensor,
    max_batches: Optional[int],
    device: str,
    amp_eps: float = 1e-6,
    eps: float = 1e-8,
) -> dict[str, dict]:
    """Run full batch-level diagnostic loop.

    Args:
        loader:      DataLoader yielding (batch_x, batch_y, origin_y, enc_x, enc_y)
        W:           Patch length (san_period_len)
        pred_len:    Prediction horizon in timesteps
        band_bins:   Dict mapping "low"/"mid"/"high" to list of freq bin indices
        tau_grid:    (T,) tensor of candidate shifts
        max_batches: Optional limit on number of batches to process
        device:      Torch device string
        amp_eps:     Minimum band amplitude energy for valid_mask
        eps:         Numerical stability constant

    Returns:
        Dict mapping "low"/"mid"/"high" to their aggregate statistics dicts.
    """
    num_fut = pred_len // W
    if num_fut == 0:
        raise ValueError(
            f"pred_len={pred_len} < san_period_len={W}: no complete future patch."
        )

    max_bin = W // 2

    # Accumulators per band: lists of 1-D tensors (one entry per processed sample)
    accum: dict[str, dict[str, list[torch.Tensor]]] = {}
    for band_name in band_bins:
        accum[band_name] = {
            "energy_ratios": [],
            "coherences":    [],
            "tau_bests":     [],
            "oracle_gains":  [],
            "mse_bases":     [],
            "mse_taus":      [],
        }

    n_batches = 0
    tau_grid = tau_grid.to(device)

    for batch in loader:
        batch_x, batch_y = batch[0], batch[1]
        batch_x = batch_x.to(device).float()   # (B, L, C)
        batch_y = batch_y.to(device).float()   # (B, H, C)

        B, L, C = batch_x.shape
        _, H, _ = batch_y.shape

        # Reference patch: last W steps of the history window
        x_ref = batch_x[:, -W:, :]             # (B, W, C)

        # rFFT of future patches for energy ratio denominator (computed below per patch)
        # precompute full future rfft bins 1..max_bin for E_total denominator
        # We'll compute per-patch

        for j in range(num_fut):
            y_patch = batch_y[:, j * W:(j + 1) * W, :]   # (B, W, C)
            if y_patch.shape[1] < W:
                continue  # incomplete patch (shouldn't happen but be safe)

            # Reshape to (N, W) where N = B * C (channel-independent modelling)
            # x_ref: (B, W, C) → (B, C, W) → (B*C, W)
            z_ref  = x_ref.permute(0, 2, 1).reshape(B * C, W)
            z_true = y_patch.permute(0, 2, 1).reshape(B * C, W)

            # Patch mean centering
            z_ref  = patch_mean_center(z_ref)
            z_true = patch_mean_center(z_true)

            # Phase rotation
            R, w_mat, X_ref, X_true = compute_phase_rotation(z_ref, z_true, eps=eps)

            # Energy of z_true across bins 1..max_bin (excludes DC)
            X_true_abs_sq = X_true.abs().pow(2)   # (N, F)
            # E_total: sum over bins 1..max_bin
            if max_bin >= 1:
                E_total = X_true_abs_sq[:, 1:max_bin + 1].sum(dim=-1) + eps   # (N,)
            else:
                E_total = torch.full((B * C,), eps, device=device)

            for band_name, bins in band_bins.items():
                if len(bins) == 0:
                    # Empty band → fill with NaN
                    n = B * C
                    nan_t = torch.full((n,), float("nan"), device=device)
                    accum[band_name]["energy_ratios"].append(torch.zeros(n, device=device))
                    accum[band_name]["coherences"].append(nan_t.clone())
                    accum[band_name]["tau_bests"].append(nan_t.clone())
                    accum[band_name]["oracle_gains"].append(nan_t.clone())
                    accum[band_name]["mse_bases"].append(nan_t.clone())
                    accum[band_name]["mse_taus"].append(nan_t.clone())
                    continue

                # Energy ratio for this band
                bins_t = torch.tensor(bins, dtype=torch.long, device=device)
                E_band = X_true_abs_sq[:, bins_t].sum(dim=-1)   # (N,)
                energy_ratio = E_band / E_total                  # (N,)

                # Coherence score
                C_band, tau_best, valid_mask = band_coherence_score(
                    R, w_mat, bins, W, tau_grid, eps=eps, amp_eps=amp_eps
                )

                # Oracle gain (only on valid samples to avoid NaN tau issues)
                n = B * C
                oracle_gains_n = torch.full((n,), float("nan"), device=device)
                mse_bases_n    = torch.full((n,), float("nan"), device=device)
                mse_taus_n     = torch.full((n,), float("nan"), device=device)

                if valid_mask.any() and len(bins) >= 2:
                    idx_valid = valid_mask.nonzero(as_tuple=True)[0]
                    z_ref_v   = z_ref[idx_valid]
                    z_true_v  = z_true[idx_valid]
                    tau_v     = tau_best[idx_valid]

                    mb, mt, gn = phase_only_oracle_gain(
                        z_ref_v, z_true_v, bins, W, tau_v, eps=eps
                    )
                    oracle_gains_n[idx_valid] = gn
                    mse_bases_n[idx_valid]    = mb
                    mse_taus_n[idx_valid]     = mt

                accum[band_name]["energy_ratios"].append(energy_ratio.cpu())
                accum[band_name]["coherences"].append(C_band.cpu())
                accum[band_name]["tau_bests"].append(tau_best.cpu())
                accum[band_name]["oracle_gains"].append(oracle_gains_n.cpu())
                accum[band_name]["mse_bases"].append(mse_bases_n.cpu())
                accum[band_name]["mse_taus"].append(mse_taus_n.cpu())

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    # Concatenate and compute statistics
    result: dict[str, dict] = {}
    for band_name in band_bins:
        a = accum[band_name]
        if not a["energy_ratios"]:
            result[band_name] = {}
            continue

        energy_ratios = torch.cat(a["energy_ratios"])
        coherences    = torch.cat(a["coherences"])
        tau_bests     = torch.cat(a["tau_bests"])
        oracle_gains  = torch.cat(a["oracle_gains"])
        mse_bases     = torch.cat(a["mse_bases"])
        mse_taus      = torch.cat(a["mse_taus"])

        n_total = energy_ratios.shape[0]
        result[band_name] = _compute_band_stats(
            energy_ratios, coherences, tau_bests,
            oracle_gains, mse_bases, mse_taus,
            n_total,
        )
        result[band_name]["n_total"] = n_total
        result[band_name]["n_valid"] = int((~torch.isnan(coherences)).sum().item())
        result[band_name]["bins"]    = band_bins[band_name]

    return result


# ---------------------------------------------------------------------------
# Stdout table
# ---------------------------------------------------------------------------

def _print_table(band_stats: dict[str, dict]) -> None:
    header = f"{'band':<6} | {'energy_med':>10} | {'valid':>6} | {'C_med':>6} | {'tau_med':>7} | {'oracle_gain_med':>15}"
    print()
    print(header)
    print("-" * len(header))
    for band_name in ("low", "mid", "high"):
        s = band_stats.get(band_name, {})
        if not s:
            print(f"{band_name:<6} | {'N/A':>10} | {'N/A':>6} | {'N/A':>6} | {'N/A':>7} | {'N/A':>15}")
            continue
        em  = s.get("energy_ratio_median")
        vr  = s.get("valid_ratio")
        cm  = s.get("coherence_median")
        tm  = s.get("tau_best_median")
        gm  = s.get("oracle_gain_median")

        def _fmt(v, fmt=".4f"):
            return format(v, fmt) if v is not None else "NaN"

        print(
            f"{band_name:<6} | {_fmt(em, '.4f'):>10} | {_fmt(vr, '.3f'):>6}"
            f" | {_fmt(cm, '.4f'):>6} | {_fmt(tm, '+.3f'):>7} | {_fmt(gm, '+.4f'):>15}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline phase-band coherence diagnostic for patch-wise normalisation."
    )
    p.add_argument("--dataset-type",  default="ETTh1",    help="Dataset name (e.g. ETTh1, Weather)")
    p.add_argument("--data-path",     default="./data",   help="Root data directory")
    p.add_argument("--window",        type=int, default=96,  help="Input lookback window")
    p.add_argument("--pred-len",      type=int, default=96,  help="Forecast horizon")
    p.add_argument("--split-type",    default="ratio",    choices=["ratio", "popular"],
                   help="Data split strategy")
    p.add_argument("--train-ratio",   type=float, default=0.7)
    p.add_argument("--val-ratio",     type=float, default=0.1)
    p.add_argument("--freq",          default="h",
                   help="Time-series frequency token (h, t, s, m, a, w, d, b)")
    p.add_argument("--san-period-len", type=int, default=12,
                   help="Patch length W (san_period_len)")
    p.add_argument("--batch-size",    type=int, default=64)
    p.add_argument("--num-worker",    type=int, default=0)
    p.add_argument("--max-batches",   type=int, default=None,
                   help="Limit number of batches (None = all)")
    p.add_argument("--low-k",         type=int, default=3,
                   help="Highest bin in low band (inclusive)")
    p.add_argument("--mid-end",       type=int, default=6,
                   help="Highest bin in mid band (inclusive)")
    p.add_argument("--tau-min",       type=float, default=-4.0,
                   help="Minimum shift to evaluate")
    p.add_argument("--tau-max",       type=float, default=4.0,
                   help="Maximum shift to evaluate")
    p.add_argument("--tau-steps",     type=int, default=161,
                   help="Number of tau candidates (linearly spaced)")
    p.add_argument("--split",         default="test", choices=["train", "val", "test"],
                   help="Which data split to analyse")
    p.add_argument("--output",        default="phase_band_report.json",
                   help="Output JSON report path")
    p.add_argument("--device",        default="cpu",
                   help="Torch device (cpu / cuda:0 / ...)")
    p.add_argument("--amp-eps",       type=float, default=1e-6,
                   help="Min band amplitude energy for a sample to be valid")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    W       = args.san_period_len
    max_bin = W // 2

    # Build band bin lists
    band_bins = _make_band_bins(args.low_k, args.mid_end, max_bin)
    print(f"[Config] W={W}  max_bin={max_bin}")
    for bn, bl in band_bins.items():
        print(f"         {bn:>4} bins: {bl}  (len={len(bl)})")

    # Build tau grid
    tau_grid = torch.linspace(args.tau_min, args.tau_max, args.tau_steps)

    # Build dataloader
    cfg = _DiagCfg(
        dataset_type = args.dataset_type,
        data_path    = args.data_path,
        device       = args.device,
        num_worker   = args.num_worker,
        window       = args.window,
        pred_len     = args.pred_len,
        horizon      = 1,
        batch_size   = args.batch_size,
        freq         = args.freq,
        train_ratio  = args.train_ratio,
        val_ratio    = args.val_ratio,
        split_type   = args.split_type,
    )

    print(f"[Data] Building loader for {args.dataset_type} split={args.split} ...")
    dataloader, dataset, scaler, split_info = _build_dataloader(cfg)

    split_map = {
        "train": dataloader.train_loader,
        "val":   dataloader.val_loader,
        "test":  dataloader.test_loader,
    }
    loader = split_map[args.split]
    print(f"[Data] split_info: {split_info}")

    num_fut = args.pred_len // W
    if num_fut == 0:
        raise ValueError(
            f"pred_len={args.pred_len} < san_period_len={W}: "
            "no complete future patch. Increase pred_len or reduce san_period_len."
        )
    print(f"[Diag] num_future_patches={num_fut}  tau_steps={args.tau_steps}")

    # Run diagnostic
    print("[Diag] Running ...")
    band_stats = run_diagnostic(
        loader      = loader,
        W           = W,
        pred_len    = args.pred_len,
        band_bins   = band_bins,
        tau_grid    = tau_grid,
        max_batches = args.max_batches,
        device      = args.device,
        amp_eps     = args.amp_eps,
    )

    # Print summary table
    _print_table(band_stats)

    # Build output JSON
    config_dict = {
        "dataset_type":  args.dataset_type,
        "data_path":     args.data_path,
        "window":        args.window,
        "pred_len":      args.pred_len,
        "split_type":    args.split_type,
        "san_period_len": W,
        "low_k":         args.low_k,
        "mid_end":       args.mid_end,
        "tau_min":       args.tau_min,
        "tau_max":       args.tau_max,
        "tau_steps":     args.tau_steps,
        "split":         args.split,
        "max_batches":   args.max_batches,
        "amp_eps":       args.amp_eps,
    }
    output = {
        "config": config_dict,
        "bands":  band_stats,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[Done] Report saved to: {args.output}")


if __name__ == "__main__":
    main()
