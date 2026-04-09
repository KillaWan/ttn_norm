"""Training entry-point for MissCorrRevIN (missing-induced state correction).

This script is intentionally self-contained for the missing-norm workflow and
does NOT modify train.py.  It reuses the helper functions
    _set_seed, _build_backbone_kwargs, _build_metrics,
    _make_dec_inputs, _move_model_to_device, TrainConfig
from experiments/train.py wherever possible.

Data loading uses a custom MissingWindowDataset that returns a stable
sample_uid per window (= absolute start index in the full dataset), enabling
fully deterministic contiguous-block missing masks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reuse helpers from train.py
# ---------------------------------------------------------------------------
from ttn_norm.experiments.train import (
    TrainConfig,
    _build_backbone_kwargs,
    _build_metrics,
    _make_dec_inputs,
    _set_seed,
)
from ttn_norm.models import TTNModel
from ttn_norm.normalizations.misscorr_revin import MissCorrRevIN
from ttn_norm.normalizations.revon import RevON
from ttn_norm.normalizations.sas_norm import SASNorm
from ttn_norm.normalizations.flow_norm import FlowNorm
from ttn_norm.normalizations.ot_norm import OTNorm
from ttn_norm.normalizations.regime_norm import RegimeNorm


# ---------------------------------------------------------------------------
# Ensure FAN is on sys.path (for torch_timeseries datasets)
# ---------------------------------------------------------------------------
def _ensure_fan_on_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fan_root = os.path.join(root, "FAN")
    if fan_root not in sys.path:
        sys.path.append(fan_root)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MissingNormConfig:
    # ---- basic training ----
    dataset: str = "ETTh1"
    data_path: str = "./data"
    backbone: str = "DLinear"
    window: int = 96
    pred_len: int = 96
    horizon: int = 1
    label_len: int = 48
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1.5e-4
    weight_decay: float = 5e-4
    max_grad_norm: float = 5.0
    early_stop: bool = True
    early_stop_patience: int = 10
    early_stop_min_epochs: int = 5
    # ---- split ----
    split_type: str = "ratio"    # "ratio" | "popular"
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    freq: str = "h"
    seed: int = 1
    num_worker: int = 4
    # ---- device / paths ----
    device: str = "cuda:0"
    result_dir: str = "./results/missing_norm"
    run_name: str = ""
    # ---- missing-specific ----
    norm_type: str = "misscorr_revin"  # "misscorr_revin" | "revon"
    missing_ratio: float = 0.2
    sigma_min: float = 1e-4
    delta_mu_scale: float = 1.0
    delta_log_sigma_scale: float = 1.0
    state_loss_weight: float = 0.1
    delta_reg_weight: float = 1e-2
    corr_hidden_dim: int = 0   # 0 = auto (min(128, 2*C))
    anchor_period: int = 24
    init_missing_token: float = 0.0
    # ---- FlowNorm ----
    flow_num_knots: int = 8
    flow_hidden_dim: int = 32
    flow_tail_bound: float = 5.0
    # ---- OTNorm ----
    ot_num_quantiles: int = 64
    # ---- RegimeNorm ----
    regime_num_prototypes: int = 8
    regime_hidden_dim: int = 32
    regime_temperature: float = 1.0
    regime_diversity_weight: float = 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MissingWindowDataset(torch.utils.data.Dataset):
    """Fixed-window dataset that exposes stable sample_uid per window.

    sample_uid is the absolute start index of the lookback window in the
    full (pre-split) dataset, ensuring globally unique and reproducible IDs.

    Returns:
        x_full  (window, C)    scaled full lookback
        y_full  (pred_len, C)  scaled target
        x_enc   (window, d)    time feature encoding for lookback
        y_enc   (pred_len, d)  time feature encoding for target
        sample_uid  int        absolute start index in full dataset
    """

    def __init__(
        self,
        scaled_data: np.ndarray,    # (T_subset, C)
        date_enc: np.ndarray,       # (T_subset, d_feat)
        abs_offset: int,            # global index of scaled_data[0]
        window: int,
        pred_len: int,
        horizon: int,
    ) -> None:
        self.scaled = torch.from_numpy(scaled_data.astype(np.float32))
        self.enc    = torch.from_numpy(date_enc.astype(np.float32))
        self.abs_offset = abs_offset
        self.window   = window
        self.pred_len = pred_len
        self.horizon  = horizon
        T = len(scaled_data)
        # same count formula as MultiStepTimeFeatureSet.__len__
        self.n_samples = T - window - horizon + 1 - pred_len + 1

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, local_idx: int):
        w, h, p = self.window, self.horizon, self.pred_len
        x_s = local_idx
        y_s = local_idx + w + h - 1
        x_full = self.scaled[x_s : x_s + w]     # (w, C)
        y_full = self.scaled[y_s : y_s + p]     # (p, C)
        x_enc  = self.enc[x_s : x_s + w]        # (w, d)
        y_enc  = self.enc[y_s : y_s + p]        # (p, d)
        sample_uid = self.abs_offset + local_idx
        return x_full, y_full, x_enc, y_enc, sample_uid


# ---------------------------------------------------------------------------
# Split-aware data building
# ---------------------------------------------------------------------------
_ETT_POPULAR = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}


def _build_split_data(cfg: MissingNormConfig):
    """Build train/val/test MissingWindowDatasets using the same split
    boundaries as experiments/train.py.

    Returns (train_ds, val_ds, test_ds, split_info, scaler).
    """
    _ensure_fan_on_path()
    import torch_timeseries.datasets as datasets
    from torch_timeseries.data.scaler import StandarScaler
    from torch_timeseries.utils.timefeatures import time_features

    aliases = {
        "exchange": "ExchangeRate",
        "exchange_rate": "ExchangeRate",
        "electricity": "Electricity",
        "traffic": "Traffic",
        "weather": "Weather",
    }
    ds_name = aliases.get(cfg.dataset, aliases.get(cfg.dataset.lower(), cfg.dataset))
    if not hasattr(datasets, ds_name):
        raise ValueError(f"Unknown dataset: {cfg.dataset!r}")
    dataset = getattr(datasets, ds_name)(root=cfg.data_path)

    scaler = StandarScaler(device=cfg.device)
    n_total = len(dataset.data)
    w, h, p = cfg.window, cfg.horizon, cfg.pred_len
    freq = cfg.freq.lower() if cfg.freq else getattr(dataset, "freq", "h")

    if cfg.split_type == "popular":
        if cfg.dataset not in _ETT_POPULAR:
            raise ValueError(
                f"split_type='popular' only supports ETT datasets; got {cfg.dataset!r}. "
                "Use --split-type ratio instead."
            )
        if cfg.dataset.startswith("ETTh"):
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        else:  # ETTm
            border2s = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]
        val_start  = border2s[0] - w - h + 1
        test_start = border2s[1] - w - h + 1
        test_end   = min(border2s[2], n_total)

        # scaler fitted on training rows (same as ETTHLoader / ETTMLoader)
        scaler.fit(dataset.data[0 : border2s[0]])

        split_info = {
            "split_type": "popular",
            "dataset_len": n_total,
            "train_idx": (0, border2s[0]),
            "val_idx":   (val_start, border2s[1]),
            "test_idx":  (test_start, test_end),
        }

    else:  # ratio
        train_size = int(cfg.train_ratio * n_total)
        val_size   = int(cfg.val_ratio   * n_total)
        test_size  = n_total - train_size - val_size

        # Boundaries matching ChunkSequenceTimefeatureDataLoader (uniform_eval=True)
        val_start  = train_size - w - h + 1
        test_start = n_total - test_size - w - h + 1
        test_end   = n_total

        scaler.fit(dataset.data[0:train_size])

        split_info = {
            "split_type": "ratio",
            "dataset_len": n_total,
            "train_idx": (0, train_size),
            "val_idx":   (val_start, train_size + val_size),
            "test_idx":  (test_start, test_end),
        }

    # --- scale full dataset once ---
    full_scaled = scaler.transform(dataset.data)   # (T, C) numpy float64

    # --- date encodings for full dataset (timeenc=0) ---
    full_date_enc = time_features(dataset.dates.copy(), timeenc=0, freq=freq)  # (T, d)

    def _make_ds(t0: int, t1: int) -> MissingWindowDataset:
        return MissingWindowDataset(
            scaled_data=full_scaled[t0:t1],
            date_enc=full_date_enc[t0:t1],
            abs_offset=t0,
            window=w,
            pred_len=p,
            horizon=h,
        )

    ti = split_info["train_idx"]
    vi = split_info["val_idx"]
    ei = split_info["test_idx"]

    return (
        _make_ds(ti[0], ti[1]),
        _make_ds(vi[0], vi[1]),
        _make_ds(ei[0], ei[1]),
        split_info,
        scaler,
        dataset.num_features,
    )


# ---------------------------------------------------------------------------
# Deterministic contiguous-block missing mask
# ---------------------------------------------------------------------------
def _make_contiguous_block_mask(
    sample_uid: int,
    window: int,
    missing_span: int,
    num_features: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a (window, num_features) mask with a deterministic missing block.

    The block start is derived from sample_uid using a fixed hash so that the
    same uid always produces the same mask regardless of split or epoch.
    """
    n_positions = window - missing_span + 1
    start = (sample_uid * 1315423911) % n_positions
    mask = torch.ones(window, num_features, device=device)
    mask[start : start + missing_span, :] = 0.0
    return mask


def _batch_masks(
    sample_uids: torch.Tensor,
    window: int,
    missing_span: int,
    num_features: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a (B, T, C) mask tensor for a batch of sample_uids."""
    masks = torch.stack(
        [
            _make_contiguous_block_mask(int(uid), window, missing_span, num_features, device)
            for uid in sample_uids
        ],
        dim=0,
    )
    return masks   # (B, T, C)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
_SUPPORTED_NORM_TYPES = {
    "misscorr_revin", "revon", "sas_norm",
    "flow_norm", "ot_norm", "regime_norm",
}


def build_phase_anchors(
    train_ds: "MissingWindowDataset",
    anchor_period: int,
    window: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute per-phase anchor statistics from the training split only.

    Phase is determined by the lookback window's end-point:
        phase_id = (sample_uid + window - 1) % anchor_period

    Returns:
        (anchor_mu, anchor_log_sigma): each shape (anchor_period, C).
        Empty phase buckets are filled with global training-set statistics.
    """
    from torch.utils.data import DataLoader as _DL

    C = train_ds.scaled.shape[1]
    P = anchor_period

    phase_mu_sum = torch.zeros(P, C)
    phase_ls_sum = torch.zeros(P, C)
    phase_count  = torch.zeros(P)

    loader = _DL(train_ds, batch_size=512, shuffle=False, num_workers=0)
    for x_full, _, _, _, sample_uids in loader:
        # x_full: (B, T, C),  sample_uids: (B,)
        phases   = (sample_uids.long() + window - 1) % P  # (B,)
        mu_batch = x_full.float().mean(dim=1)              # (B, C)
        ls_batch = (
            x_full.float().std(dim=1, unbiased=False) + eps
        ).log()                                            # (B, C)

        idx_exp = phases.unsqueeze(1).expand_as(mu_batch)  # (B, C)
        phase_mu_sum.scatter_add_(0, idx_exp, mu_batch)
        phase_ls_sum.scatter_add_(0, idx_exp, ls_batch)
        phase_count.scatter_add_(0, phases, torch.ones(len(phases)))

    total_count  = phase_count.sum().clamp(min=1)
    global_mu    = phase_mu_sum.sum(dim=0) / total_count
    global_ls    = phase_ls_sum.sum(dim=0) / total_count

    count_safe       = phase_count.clamp(min=1).unsqueeze(1)  # (P, 1)
    anchor_mu        = phase_mu_sum / count_safe
    anchor_log_sigma = phase_ls_sum / count_safe

    empty = (phase_count == 0)
    if empty.any():
        anchor_mu[empty]        = global_mu
        anchor_log_sigma[empty] = global_ls

    return anchor_mu, anchor_log_sigma


def build_missing_model(
    cfg: MissingNormConfig,
    num_features: int,
    anchor_mu: Optional[torch.Tensor] = None,
    anchor_log_sigma: Optional[torch.Tensor] = None,
) -> TTNModel:
    """Build TTNModel wrapping a missing-aware norm.

    The norm uses the mask internally; the backbone receives num_features
    input channels (C) carrying the masked-and-normalized signal.
    """
    nt = cfg.norm_type.lower()
    if nt not in _SUPPORTED_NORM_TYPES:
        raise ValueError(
            f"norm_type must be one of {sorted(_SUPPORTED_NORM_TYPES)}, got {cfg.norm_type!r}"
        )

    if nt == "revon":
        norm_model = RevON(
            num_features=num_features,
            sigma_min=cfg.sigma_min,
        )
    elif nt == "flow_norm":
        norm_model = FlowNorm(
            num_features=num_features,
            num_knots=cfg.flow_num_knots,
            hidden_dim=cfg.flow_hidden_dim,
            tail_bound=cfg.flow_tail_bound,
            sigma_min=cfg.sigma_min,
        )
    elif nt == "ot_norm":
        norm_model = OTNorm(
            num_features=num_features,
            num_quantiles=cfg.ot_num_quantiles,
            sigma_min=cfg.sigma_min,
        )
    elif nt == "regime_norm":
        norm_model = RegimeNorm(
            num_features=num_features,
            num_prototypes=cfg.regime_num_prototypes,
            hidden_dim=cfg.regime_hidden_dim,
            sigma_min=cfg.sigma_min,
            temperature=cfg.regime_temperature,
            diversity_weight=cfg.regime_diversity_weight,
        )
    elif nt == "sas_norm":
        if anchor_mu is None or anchor_log_sigma is None:
            raise ValueError(
                "build_missing_model() requires anchor_mu and anchor_log_sigma "
                "for norm_type='sas_norm'.  Call build_phase_anchors() first."
            )
        norm_model = SASNorm(
            num_features=num_features,
            anchor_period=cfg.anchor_period,
            anchor_mu=anchor_mu,
            anchor_log_sigma=anchor_log_sigma,
            sigma_min=cfg.sigma_min,
            init_missing_token=cfg.init_missing_token,
        )
    else:  # misscorr_revin
        hidden = cfg.corr_hidden_dim if cfg.corr_hidden_dim > 0 else None
        norm_model = MissCorrRevIN(
            num_features=num_features,
            corr_hidden_dim=hidden,
            sigma_min=cfg.sigma_min,
            delta_mu_scale=cfg.delta_mu_scale,
            delta_log_sigma_scale=cfg.delta_log_sigma_scale,
            delta_reg_weight=cfg.delta_reg_weight,
        )

    # Backbone receives C channels: the masked-and-normalized signal mask*z
    backbone_input_features = num_features
    label_len = min(cfg.label_len, cfg.window)

    # Build a minimal TrainConfig stub so we can reuse _build_backbone_kwargs
    tc = TrainConfig(
        backbone_type=cfg.backbone,
        window=cfg.window,
        pred_len=cfg.pred_len,
        horizon=cfg.horizon,
        freq=cfg.freq,
        label_len=cfg.label_len,
    )
    backbone_kwargs = _build_backbone_kwargs(tc, backbone_input_features, label_len)

    return TTNModel(
        backbone_type=cfg.backbone,
        backbone_kwargs=backbone_kwargs,
        norm_model=norm_model,
        is_former=None,   # auto-detect from backbone name
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _prep_former_inputs(
    x_obs: torch.Tensor,
    y_enc: torch.Tensor,
    x_enc: torch.Tensor,
    label_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dec_inp and dec_inp_enc for former-style backbones.

    Mirrors the logic in experiments/train.py::_make_dec_inputs but takes
    x_obs (already normalized / observed-only) rather than raw batch_x.
    """
    return _make_dec_inputs(x_obs, y_enc, x_enc, label_len)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _train_one_epoch(
    model: TTNModel,
    loader: DataLoader,
    optimizer: Adam,
    cfg: MissingNormConfig,
    missing_span: int,
    num_features: int,
) -> dict[str, float]:
    model.train()
    loss_fn = nn.MSELoss()
    device = torch.device(cfg.device)

    forecast_losses: list[float] = []
    state_losses:    list[float] = []
    mu_losses:       list[float] = []
    sigma_losses:    list[float] = []
    dm_l2s:          list[float] = []
    dls_l2s:         list[float] = []
    is_sas = cfg.norm_type.lower() == "sas_norm"

    with tqdm(total=len(loader.dataset), leave=False) as pbar:
        for x_full, y_full, x_enc, y_enc, sample_uids in loader:
            x_full = x_full.to(device)        # (B, T, C)
            y_full = y_full.to(device)        # (B, H, C)
            x_enc  = x_enc.to(device)
            y_enc  = y_enc.to(device)

            # --- deterministic mask ---
            mask = _batch_masks(
                sample_uids, cfg.window, missing_span, num_features, device
            )  # (B, T, C)
            x_obs = mask * x_full             # (B, T, C)

            # --- set context before forward ---
            if is_sas:
                phase_id = (
                    (sample_uids.long() + cfg.window - 1) % cfg.anchor_period
                ).to(device)
                model.nm.set_missing_context(mask, phase_id, x_full=x_full)
            else:
                model.nm.set_missing_context(mask, x_full)

            optimizer.zero_grad()

            # --- build decoder inputs for former backbones ---
            label_len = min(cfg.label_len, cfg.window)
            if model.is_former:
                dec_inp, dec_inp_enc = _prep_former_inputs(
                    x_obs, y_enc, x_enc, label_len
                )
            else:
                dec_inp, dec_inp_enc = None, None

            pred = model(x_obs, x_enc, dec_inp, dec_inp_enc)  # (B, H, C)

            # --- losses ---
            forecast_loss = loss_fn(pred, y_full)
            if is_sas:
                # sas_norm: total loss = forecast only (no auxiliary term)
                state_loss = forecast_loss.new_zeros(())
                total_loss = forecast_loss
            else:
                corr_reg   = model.nm.loss(x_full)  # correction regularization penalty
                state_loss = corr_reg
                total_loss = forecast_loss + cfg.state_loss_weight * corr_reg

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # --- logging ---
            aux = model.nm.get_last_aux_stats()
            forecast_losses.append(forecast_loss.detach().item())
            state_losses.append(state_loss.detach().item())
            mu_losses.append(aux.get("mu_loss", float("nan")))
            sigma_losses.append(aux.get("sigma_loss", float("nan")))
            dm_l2s.append(aux.get("delta_mu_l2", float("nan")))
            dls_l2s.append(aux.get("delta_log_sigma_l2", float("nan")))

            pbar.update(x_full.size(0))
            pbar.set_postfix(
                fc=f"{forecast_loss.item():.4f}",
                st=f"{state_loss.item():.4f}",
            )

    def _mean(lst: list[float]) -> float:
        vals = [v for v in lst if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "forecast_loss":    _mean(forecast_losses),
        "state_loss":       _mean(state_losses),
        "mu_loss":          _mean(mu_losses),
        "sigma_loss":       _mean(sigma_losses),
        "delta_mu_l2":      _mean(dm_l2s),
        "delta_log_sigma_l2": _mean(dls_l2s),
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate(
    model: TTNModel,
    loader: DataLoader,
    cfg: MissingNormConfig,
    missing_span: int,
    num_features: int,
) -> dict[str, float]:
    model.eval()
    device = torch.device(cfg.device)
    metrics = _build_metrics(device)
    for m in metrics.values():
        m.reset()

    is_sas = cfg.norm_type.lower() == "sas_norm"
    for x_full, y_full, x_enc, y_enc, sample_uids in loader:
        x_full = x_full.to(device)
        y_full = y_full.to(device)
        x_enc  = x_enc.to(device)
        y_enc  = y_enc.to(device)

        mask = _batch_masks(
            sample_uids, cfg.window, missing_span, num_features, device
        )
        x_obs = mask * x_full

        if is_sas:
            phase_id = (
                (sample_uids.long() + cfg.window - 1) % cfg.anchor_period
            ).to(device)
            model.nm.set_missing_context(mask, phase_id, x_full=x_full)
        else:
            model.nm.set_missing_context(mask, x_full)

        # --- build decoder inputs for former backbones ---
        label_len = min(cfg.label_len, cfg.window)
        if model.is_former:
            dec_inp, dec_inp_enc = _prep_former_inputs(
                x_obs, y_enc, x_enc, label_len
            )
        else:
            dec_inp, dec_inp_enc = None, None

        pred = model(x_obs, x_enc, dec_inp, dec_inp_enc)  # (B, H, C)

        pred   = pred.contiguous()
        y_full = y_full.contiguous()
        if cfg.pred_len == 1:
            B = pred.shape[0]
            pred   = pred.view(B, -1)
            y_full = y_full.view(B, -1)
        for m in metrics.values():
            m.update(pred, y_full)

    return {name: float(m.compute()) for name, m in metrics.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train MissCorrRevIN (missing-induced RevIN state correction)"
    )
    defaults = asdict(MissingNormConfig())
    for key, val in defaults.items():
        arg = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            parser.add_argument(arg, action="store_true", default=val)
            parser.add_argument(
                f"--no-{key.replace('_', '-')}",
                dest=key,
                action="store_false",
            )
        else:
            parser.add_argument(arg, type=type(val), default=val)
    args = parser.parse_args(argv)
    cfg  = MissingNormConfig(**vars(args))

    # Validate missing_ratio
    if not (0.0 < cfg.missing_ratio < 1.0):
        raise ValueError(f"missing_ratio must be in (0, 1), got {cfg.missing_ratio}")

    _set_seed(cfg.seed)

    # -- build data ---------------------------------------------------------
    train_ds, val_ds, test_ds, split_info, scaler, num_features = _build_split_data(cfg)

    missing_span = max(1, round(cfg.missing_ratio * cfg.window))
    print(
        f"[{cfg.norm_type}] dataset={cfg.dataset}  backbone={cfg.backbone}"
        f"  window={cfg.window}  pred_len={cfg.pred_len}"
        f"\n  missing_ratio={cfg.missing_ratio}  missing_span={missing_span}"
        f"  num_features={num_features}"
        f"\n  split={cfg.split_type}  train={split_info['train_idx']}"
        f"  val={split_info['val_idx']}  test={split_info['test_idx']}"
        f"\n  train_samples={len(train_ds)}  val_samples={len(val_ds)}"
        f"  test_samples={len(test_ds)}"
    )

    _train_gen = torch.Generator()
    _train_gen.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_worker, generator=_train_gen,
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_worker
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_worker
    )

    # -- build anchor tables for sas_norm -----------------------------------
    anchor_mu_tbl = anchor_ls_tbl = None
    if cfg.norm_type.lower() == "sas_norm":
        print(
            f"[sas_norm] pre-computing phase anchors "
            f"(period={cfg.anchor_period}, train_samples={len(train_ds)})..."
        )
        anchor_mu_tbl, anchor_ls_tbl = build_phase_anchors(
            train_ds, cfg.anchor_period, cfg.window
        )
        print(f"  anchor_mu={tuple(anchor_mu_tbl.shape)}  done.")

    # -- build model --------------------------------------------------------
    model = build_missing_model(cfg, num_features, anchor_mu_tbl, anchor_ls_tbl)
    model = model.to(cfg.device)
    print(
        f"[Model] backbone={cfg.backbone}  backbone_input_channels={num_features}"
        f"  norm={cfg.norm_type}(C={num_features})"
    )

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    run_name = cfg.run_name or (
        f"{cfg.dataset}_{cfg.backbone}_{cfg.norm_type}"
        f"_mr{cfg.missing_ratio}_w{cfg.window}_p{cfg.pred_len}"
    )
    os.makedirs(cfg.result_dir, exist_ok=True)
    result_path  = os.path.join(cfg.result_dir, f"{run_name}.json")
    ckpt_path    = os.path.join(cfg.result_dir, f"{run_name}_best.pth")

    # -- training -----------------------------------------------------------
    best_val_mse    = float("inf")
    best_val_metrics: dict[str, float] = {}
    best_test_metrics: dict[str, float] = {}
    no_improve      = 0
    train_losses: list[float] = []

    for epoch in range(cfg.epochs):
        train_stat = _train_one_epoch(
            model, train_loader, optimizer, cfg, missing_span, num_features
        )
        scheduler.step()

        val_metrics  = _evaluate(
            model, val_loader, cfg, missing_span, num_features
        )
        test_metrics = _evaluate(
            model, test_loader, cfg, missing_span, num_features
        )
        val_mse  = val_metrics["mse"]
        test_mse = test_metrics["mse"]

        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs}"
            f"  forecast_loss={train_stat['forecast_loss']:.6f}"
            f"  state_loss={train_stat['state_loss']:.6f}"
            f"  mu_loss={train_stat['mu_loss']:.6f}"
            f"  sigma_loss={train_stat['sigma_loss']:.6f}"
            f"  delta_mu_l2={train_stat['delta_mu_l2']:.6f}"
            f"  delta_log_sigma_l2={train_stat['delta_log_sigma_l2']:.6f}"
            f"  val_mse={val_mse:.6f}"
            f"  test_mse={test_mse:.6f}"
        )
        train_losses.append(train_stat["forecast_loss"])

        # -- early stopping -------------------------------------------------
        improved = val_mse < best_val_mse
        if improved:
            best_val_mse      = val_mse
            best_val_metrics  = dict(val_metrics)
            best_test_metrics = dict(test_metrics)
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(
                f"  [BEST] val_mse={best_val_mse:.6f}"
                f"  test_mse={best_test_metrics['mse']:.6f}"
                f"  → saved {ckpt_path}"
            )
        else:
            no_improve += 1

        if (
            cfg.early_stop
            and epoch + 1 >= cfg.early_stop_min_epochs
            and no_improve >= cfg.early_stop_patience
        ):
            print(f"  [EarlyStop] no improvement for {no_improve} epochs, stopping.")
            break

    # -- save results -------------------------------------------------------
    result = {
        "run_name":      run_name,
        "dataset":       cfg.dataset,
        "backbone":      cfg.backbone,
        "norm_type":     cfg.norm_type,
        "missing_ratio": cfg.missing_ratio,
        "missing_span":  missing_span,
        "window":        cfg.window,
        "pred_len":      cfg.pred_len,
        "train_loss":    float(np.mean(train_losses)) if train_losses else float("nan"),
        "val_metrics":   best_val_metrics,
        "test_metrics":  best_test_metrics,
        "config":        asdict(cfg),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Done] Results saved → {result_path}")
    print(f"       Best checkpoint   → {ckpt_path}")
    print(
        f"       val_mse={best_val_metrics.get('mse', float('nan')):.6f}"
        f"  val_mae={best_val_metrics.get('mae', float('nan')):.6f}"
        f"  test_mse={best_test_metrics.get('mse', float('nan')):.6f}"
        f"  test_mae={best_test_metrics.get('mae', float('nan')):.6f}"
    )


if __name__ == "__main__":
    main()
