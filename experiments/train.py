from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError

from ttn_norm.models import LocalTFNorm, TTNModel
from ttn_norm.utils.metrics import RMSE


def _ensure_fan_on_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fan_root = os.path.join(root, "FAN")
    if fan_root not in sys.path:
        sys.path.append(fan_root)


def _parse_type(name: str):
    _ensure_fan_on_path()
    import torch_timeseries.datasets as datasets

    if not hasattr(datasets, name):
        raise ValueError(f"Unknown dataset type: {name}")
    return getattr(datasets, name)


def _parse_scaler(name: str):
    _ensure_fan_on_path()
    from torch_timeseries.data.scaler import MaxAbsScaler, NoScaler, StandarScaler

    mapping = {
        "StandarScaler": StandarScaler,
        "MaxAbsScaler": MaxAbsScaler,
        "NoScaler": NoScaler,
    }
    if name not in mapping:
        raise ValueError(f"Unknown scaler type: {name}")
    return mapping[name]


# ETT datasets that support the "popular" fixed calendar split
_ETT_POPULAR_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}


def _build_dataloader(cfg):
    """Build train/val/test dataloaders according to cfg.split_type.

    Returns:
        (dataloader, dataset, scaler, split_info)
        where split_info is a dict describing the exact row boundaries used.
    """
    _ensure_fan_on_path()
    from torch_timeseries.datasets.dataloader import (
        ChunkSequenceTimefeatureDataLoader,
        ETTHLoader,
        ETTMLoader,
    )

    if cfg.split_type not in {"ratio", "popular"}:
        raise ValueError(
            f"split_type must be 'ratio' or 'popular', got: {cfg.split_type!r}"
        )

    dataset = _parse_type(cfg.dataset_type)(root=cfg.data_path)
    scaler = _parse_scaler(cfg.scaler_type)(device=cfg.device)
    n_total = len(dataset)

    if cfg.split_type == "popular":
        if cfg.dataset_type not in _ETT_POPULAR_DATASETS:
            raise ValueError(
                f"split_type='popular' is only supported for ETT datasets "
                f"({sorted(_ETT_POPULAR_DATASETS)}). "
                f"Got dataset_type={cfg.dataset_type!r}. "
                f"Use --split-type ratio instead."
            )
        # Use FAN's fixed calendar-based borders (the standard ETT "popular" split):
        #   ETTh: 12 months train / 4 months val / 4 months test (hourly, 30-day month)
        #   ETTm: same × 4 (15-minute resolution)
        w, h = cfg.window, cfg.horizon
        if cfg.dataset_type.startswith("ETTh"):
            border2s = [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ]
            LoaderCls = ETTHLoader
        else:  # ETTm1 / ETTm2
            border2s = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]
            LoaderCls = ETTMLoader

        # Val/test raw data windows start slightly before the split boundary so
        # the first evaluation sample has a full input context (same as FAN).
        val_start  = border2s[0] - w - h + 1
        test_start = border2s[1] - w - h + 1
        train_end, val_end, test_end = border2s
        test_end_capped = min(test_end, n_total)

        split_info = {
            "split_type": "popular",
            "dataset_len": n_total,
            "train_rows": train_end,
            "val_rows":   val_end - train_end,
            "test_rows":  test_end_capped - val_end,
            "train_idx":  (0, train_end),
            "val_idx":    (val_start, val_end),
            "test_idx":   (test_start, test_end_capped),
        }

        dataloader = LoaderCls(
            dataset,
            scaler,
            window=cfg.window,
            horizon=cfg.horizon,
            steps=cfg.pred_len,
            shuffle_train=True,
            freq=cfg.freq,
            batch_size=cfg.batch_size,
            num_worker=cfg.num_worker,
        )

    else:  # split_type == "ratio"
        train_size = int(cfg.train_ratio * n_total)
        val_size   = int(cfg.val_ratio   * n_total)
        test_size  = n_total - train_size - val_size

        split_info = {
            "split_type": "ratio",
            "dataset_len": n_total,
            "train_rows": train_size,
            "val_rows":   val_size,
            "test_rows":  test_size,
            "train_ratio": cfg.train_ratio,
            "val_ratio":   cfg.val_ratio,
            "train_idx":  (0, train_size),
            "val_idx":    (train_size, train_size + val_size),
            "test_idx":   (train_size + val_size, n_total),
        }

        dataloader = ChunkSequenceTimefeatureDataLoader(
            dataset,
            scaler,
            window=cfg.window,
            horizon=cfg.horizon,
            steps=cfg.pred_len,
            scale_in_train=cfg.scale_in_train,
            shuffle_train=True,
            freq=cfg.freq,
            batch_size=cfg.batch_size,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            num_worker=cfg.num_worker,
        )

    return dataloader, dataset, scaler, split_info


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_dec_inputs(
    batch_x: torch.Tensor,
    batch_y_date_enc: torch.Tensor,
    batch_x_date_enc: torch.Tensor,
    label_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_len = batch_y_date_enc.shape[1]
    dec_inp_pred = torch.zeros(
        [batch_x.size(0), pred_len, batch_x.size(-1)], device=batch_x.device
    )
    dec_inp_label = batch_x[:, -label_len:, :]
    dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
    dec_inp_date_enc = torch.cat(
        [batch_x_date_enc[:, -label_len:, :], batch_y_date_enc], dim=1
    )
    return dec_inp, dec_inp_date_enc


def _build_metrics(device: torch.device):
    metrics = {
        "mse": MeanSquaredError().to(device),
        "mae": MeanAbsoluteError().to(device),
        "mape": MeanAbsolutePercentageError().to(device),
        "rmse": RMSE().to(device),
    }
    return metrics


def _build_backbone_kwargs(cfg: TrainConfig, num_features: int, label_len: int) -> dict[str, Any]:
    name = cfg.backbone_type
    if name == "DLinear":
        return dict(seq_len=cfg.window, pred_len=cfg.pred_len, enc_in=num_features, individual=False)
    if name == "TSMixer":
        return dict(L=cfg.window, C=num_features, T=cfg.pred_len)
    if name == "TimesNet":
        return dict(
            seq_len=cfg.window,
            label_len=label_len,
            pred_len=cfg.pred_len,
            e_layers=2,
            d_ff=128,
            num_kernels=3,
            top_k=5,
            d_model=128,
            embed="timeF",
            enc_in=num_features,
            freq=cfg.freq or "h",
            dropout=0.0,
            c_out=num_features,
            task_name="long_term_forecast",
        )
    if name == "Informer":
        return dict(
            enc_in=num_features,
            dec_in=num_features,
            c_out=num_features,
            out_len=cfg.pred_len,
            factor=5,
            d_model=512,
            n_heads=8,
            e_layers=2,
            d_layers=2,
            d_ff=512,
            dropout=0.0,
            attn="prob",
            embed="fixed",
            freq=cfg.freq or "h",
            activation="gelu",
            distil=True,
            mix=True,
        )
    if name == "Autoformer":
        return dict(
            enc_in=num_features,
            dec_in=num_features,
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            label_len=label_len,
            c_out=num_features,
            factor=1,
            d_ff=2048,
            activation="gelu",
            e_layers=2,
            d_layers=1,
            output_attention=False,
            moving_avg=[24],
            n_heads=8,
            d_model=512,
            embed="timeF",
            freq=cfg.freq or "h",
            dropout=0.0,
        )
    if name == "FEDformer":
        return dict(
            enc_in=num_features,
            dec_in=num_features,
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            label_len=label_len,
            c_out=num_features,
            version="Fourier",
            L=3,
            d_ff=2048,
            activation="gelu",
            e_layers=2,
            d_layers=1,
            mode_select="random",
            modes=64,
            output_attention=False,
            moving_avg=[24],
            n_heads=8,
            cross_activation="tanh",
            d_model=512,
            embed="timeF",
            freq=cfg.freq or "h",
            dropout=0.0,
            base="legendre",
        )
    if name == "SCINet":
        return dict(
            output_len=cfg.pred_len,
            input_len=cfg.window,
            input_dim=num_features,
            hid_size=1,
            num_stacks=1,
            num_levels=3,
            num_decoder_layer=1,
            concat_len=0,
            groups=1,
            kernel=5,
            dropout=0.5,
            single_step_output_One=0,
            input_len_seg=0,
            positionalE=False,
            modified=True,
            RIN=False,
        )
    raise ValueError(
        f"Backbone {name} requires --backbone-kwargs JSON for initialization."
    )


@dataclass
class TrainConfig:
    dataset_type: str = "ETTh1"
    data_path: str = "./data"
    device: str = "cuda:0"
    num_worker: int = 4
    seed: int = 1

    backbone_type: str = "DLinear"
    window: int = 96
    pred_len: int = 96
    horizon: int = 1
    batch_size: int = 32
    epochs: int = 1000
    lr: float = 1.5e-4
    max_grad_norm: float = 5.0
    weight_decay: float = 5e-4
    gate_weight_decay: float = 2e-3
    predictor_weight_decay: float = 2e-3
    early_stop: bool = True
    early_stop_patience: int = 5
    early_stop_min_epochs: int = 1
    early_stop_metric: str = "val_mse"
    val_mse_ema_alpha: float = 1.0
    early_stop_delta: float = 0.0
    aux_loss_schedule: str = "cosine"
    aux_loss_scale: float = 0.2
    aux_loss_min_scale: float = 0.05
    aux_loss_decay_start_epoch: int = 4
    aux_loss_decay_epochs: int = 16

    scaler_type: str = "StandarScaler"
    scale_in_train: bool = False
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    split_type: str = "ratio"   # "ratio" | "popular"
    freq: str | None = None

    label_len: int = 48
    invtrans_loss: bool = False
    force_former: bool = False

    norm_type: str = "localtf"

    backbone_kwargs: str = ""
    result_dir: str = "./results"
    baseline_subdir: str = "baselines"
    run_name: str = ""

    # Local TF norm config
    n_fft: int | None = None
    hop_length: int | None = None
    win_length: int | None = None
    gate_type: str = "depthwise"
    gate_log_mag: bool = True
    gate_arch: str = "pointwise"
    gate_threshold_mode: str = "shift"
    gate_entropy_weight: float = 0.0
    gate_use_log_mag: bool = True
    stationarity_loss_weight: float = 0.0
    stationarity_chunks: int = 4
    future_mode: str = "repeat_last"
    predict_n_time: bool = False
    pred_hidden_dim: int = 64
    pred_dropout: float = 0.1
    pred_loss_weight: float = 0.0
    gate_threshold: float = 0.0
    gate_temperature: float = 1.0
    gate_smooth_weight: float = 0.0
    gate_temporal_smooth_weight: float = 0.0
    gate_ratio_weight: float = 0.0
    gate_ratio_target: float = 0.3
    gate_mode: str = "sigmoid"
    gate_budget_dim: str = "freq"
    pred_input: str = "n_tf"
    gate_lr: float = 0.0
    predictor_lr: float = 0.0
    use_instance_norm: bool = True
    lambda_E: float = 1.0
    lambda_P: float = 1.0
    eps_E: float = 1e-6
    delta_E: float = 0.0
    delta_P: float = 0.0
    auto_thresholds: bool = True
    trigger_q: float = 0.99
    target_frames: int = 24
    auto_stft: bool = True
    trigger_mask: bool = True
    delta_E_mask: float = 0.0
    delta_P_mask: float = 0.0


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def build_model(cfg: TrainConfig, num_features: int) -> TTNModel:
    if cfg.norm_type.lower() in {"none", "baseline", "no"}:
        norm_model: nn.Module | None = nn.Identity()
    else:
        # Auto-compute STFT params from target_frames when requested
        if cfg.auto_stft and cfg.hop_length is None and cfg.win_length is None and cfg.n_fft is None:
            hop = max(1, cfg.window // cfg.target_frames)
            win = 4 * hop
            nfft = _next_power_of_two(win)
            cfg.hop_length = hop
            cfg.win_length = win
            cfg.n_fft = nfft

        norm_model = LocalTFNorm(
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            enc_in=num_features,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            gate_type=cfg.gate_type,
            gate_log_mag=cfg.gate_log_mag,
            gate_arch=cfg.gate_arch,
            gate_threshold_mode=cfg.gate_threshold_mode,
            gate_entropy_weight=cfg.gate_entropy_weight,
            gate_use_log_mag=cfg.gate_use_log_mag,
            stationarity_loss_weight=cfg.stationarity_loss_weight,
            stationarity_chunks=cfg.stationarity_chunks,
            future_mode=cfg.future_mode,
            predict_n_time=cfg.predict_n_time,
            pred_hidden_dim=cfg.pred_hidden_dim,
            pred_dropout=cfg.pred_dropout,
            pred_loss_weight=cfg.pred_loss_weight,
            gate_threshold=cfg.gate_threshold,
            gate_temperature=cfg.gate_temperature,
            gate_smooth_weight=cfg.gate_smooth_weight,
            gate_temporal_smooth_weight=cfg.gate_temporal_smooth_weight,
            gate_ratio_weight=cfg.gate_ratio_weight,
            gate_ratio_target=cfg.gate_ratio_target,
            gate_mode=cfg.gate_mode,
            gate_budget_dim=cfg.gate_budget_dim,
            pred_input=cfg.pred_input,
            use_instance_norm=cfg.use_instance_norm,
            lambda_E=cfg.lambda_E,
            lambda_P=cfg.lambda_P,
            eps_E=cfg.eps_E,
            delta_E=cfg.delta_E,
            delta_P=cfg.delta_P,
            trigger_mask=cfg.trigger_mask,
            delta_E_mask=cfg.delta_E_mask,
            delta_P_mask=cfg.delta_P_mask,
        )
    label_len = cfg.label_len or (cfg.window // 2)
    label_len = min(label_len, cfg.window)
    backbone_kwargs = _build_backbone_kwargs(cfg, num_features, label_len)
    if cfg.backbone_kwargs:
        extra = json.loads(cfg.backbone_kwargs)
        backbone_kwargs.update(extra)
    ttn = TTNModel(
        backbone_type=cfg.backbone_type,
        backbone_kwargs=backbone_kwargs,
        norm_model=norm_model,
        is_former=True if cfg.force_former else None,
    )
    # Print STFT configuration for reproducibility
    _nm = getattr(ttn, "nm", None)
    if _nm is not None and hasattr(_nm, "stft"):
        _stft = _nm.stft
        _freq_bins = _stft.n_fft // 2 + 1
        _in_frames = _stft.time_bins(cfg.window)
        _out_frames = _stft.time_bins(cfg.pred_len)
        print(
            f"[STFT_CFG] n_fft={_stft.n_fft} hop_length={_stft.hop_length}"
            f" win_length={_stft.win_length} freq_bins={_freq_bins}"
            f" in_frames={_in_frames} out_frames={_out_frames}"
            f" target_frames={cfg.target_frames} auto_stft={cfg.auto_stft}"
        )
    return ttn


@torch.no_grad()
def calibrate_thresholds(
    model: nn.Module,
    train_loader,
    cfg: TrainConfig,
    max_batches: int = 200,
) -> tuple[float, float, float, float]:
    """Estimate delta_E, delta_P, delta_E_mask, and delta_P_mask from training data.

    Runs up to *max_batches* forward passes, collects per-transition log-energy
    differences (dE) and per-transition mean spectral-shape differences (dP)
    from both the residual r_tf (for stationarity loss thresholds) and the raw
    input x_tf (for trigger mask thresholds), then returns the trigger_q quantile
    of each as the margin thresholds.
    """
    nm = getattr(model, "nm", None)
    if nm is None or not hasattr(nm, "_last_r_tf"):
        return 0.0, 0.0, 0.0, 0.0

    model.eval()
    eps = 1e-8
    eps_E = float(getattr(nm, "eps_E", 1e-6))

    # Temporarily disable trigger_mask so forward passes succeed before thresholds are known
    _orig_trigger_mask = getattr(nm, "trigger_mask", False)
    nm.trigger_mask = False

    all_dE: list[torch.Tensor] = []
    all_dP: list[torch.Tensor] = []
    all_dE_mask: list[torch.Tensor] = []
    all_dP_mask: list[torch.Tensor] = []

    for i, (batch_x, batch_y, origin_y, batch_x_enc, batch_y_enc) in enumerate(train_loader):
        if i >= max_batches:
            break
        batch_x = batch_x.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()

        dec_inp, dec_inp_enc = None, None
        if model.is_former:
            batch_y = batch_y.to(cfg.device).float()
            batch_y_enc = batch_y_enc.to(cfg.device).float()
            label_len = min(cfg.label_len or (cfg.window // 2), batch_x.size(1))
            dec_inp, dec_inp_enc = _make_dec_inputs(
                batch_x, batch_y_enc, batch_x_enc, label_len
            )

        model(batch_x, batch_x_enc, dec_inp, dec_inp_enc)

        # --- dE/dP from residual r_tf (for stationarity loss thresholds) ---
        r_tf = nm._last_r_tf  # (B, C, F, T) complex
        if r_tf is None:
            continue

        P = r_tf.abs() ** 2                              # (B, C, F, T)
        E = P.mean(dim=2)                                # (B, C, T)
        p = P / (P.sum(dim=2, keepdim=True) + eps)      # (B, C, F, T)

        logE = torch.log(E + eps_E)
        dE = torch.abs(logE[..., 1:] - logE[..., :-1])  # (B, C, T-1)
        dP = torch.abs(p[..., :, 1:] - p[..., :, :-1]).mean(dim=2)  # (B, C, T-1)

        all_dE.append(dE.cpu().flatten())
        all_dP.append(dP.cpu().flatten())

        # --- dE/dP from raw input x_tf (for trigger mask thresholds) ---
        x_tf = nm._last_x_tf  # (B, C, F, T) complex
        if x_tf is not None:
            P_x = x_tf.abs() ** 2                               # (B, C, F, T)
            E_x = P_x.mean(dim=2)                               # (B, C, T)
            p_x = P_x / (P_x.sum(dim=2, keepdim=True) + eps)   # (B, C, F, T)
            logE_x = torch.log(E_x + eps_E)
            dE_x = torch.abs(logE_x[..., 1:] - logE_x[..., :-1])          # (B, C, T-1)
            dP_x = torch.abs(p_x[..., :, 1:] - p_x[..., :, :-1]).mean(dim=2)  # (B, C, T-1)
            all_dE_mask.append(dE_x.cpu().flatten())
            all_dP_mask.append(dP_x.cpu().flatten())

    # Restore original trigger_mask setting
    nm.trigger_mask = _orig_trigger_mask

    if not all_dE:
        return 0.0, 0.0, 0.0, 0.0

    cat_dE = torch.cat(all_dE)
    cat_dP = torch.cat(all_dP)
    q = float(cfg.trigger_q)
    delta_E = float(torch.quantile(cat_dE, q).item())
    delta_P = float(torch.quantile(cat_dP, q).item())

    def _qs(t: torch.Tensor, qs: list[float]) -> list[float]:
        return [float(torch.quantile(t, qi).item()) for qi in qs]

    qs = [0.50, 0.90, 0.95, 0.99, 0.999]
    eqs = _qs(cat_dE, qs)
    pqs = _qs(cat_dP, qs)
    e99, e95 = eqs[3], eqs[2]
    p99, p95 = pqs[3], pqs[2]
    heavy_E = e99 / e95 if e95 > 1e-15 else float("nan")
    heavy_P = p99 / p95 if p95 > 1e-15 else float("nan")
    print(
        f"[Calibration][dE_dist]"
        f" n={cat_dE.numel()}"
        f" mean={cat_dE.mean():.6f} std={cat_dE.std():.6f}"
        f" p50={eqs[0]:.6f} p90={eqs[1]:.6f} p95={eqs[2]:.6f}"
        f" p99={eqs[3]:.6f} p999={eqs[4]:.6f}"
        f" heavy_tail_ratio={heavy_E:.4f}"
    )
    print(
        f"[Calibration][dP_dist]"
        f" n={cat_dP.numel()}"
        f" mean={cat_dP.mean():.6f} std={cat_dP.std():.6f}"
        f" p50={pqs[0]:.6f} p90={pqs[1]:.6f} p95={pqs[2]:.6f}"
        f" p99={pqs[3]:.6f} p999={pqs[4]:.6f}"
        f" heavy_tail_ratio={heavy_P:.4f}"
    )

    # --- mask thresholds from x_tf distribution ---
    if all_dE_mask:
        cat_dE_mask = torch.cat(all_dE_mask)
        cat_dP_mask = torch.cat(all_dP_mask)
        delta_E_mask = float(torch.quantile(cat_dE_mask, q).item())
        delta_P_mask = float(torch.quantile(cat_dP_mask, q).item())
        meqs = _qs(cat_dE_mask, qs)
        mpqs = _qs(cat_dP_mask, qs)
        me99, me95 = meqs[3], meqs[2]
        mp99, mp95 = mpqs[3], mpqs[2]
        heavy_mE = me99 / me95 if me95 > 1e-15 else float("nan")
        heavy_mP = mp99 / mp95 if mp95 > 1e-15 else float("nan")
        print(
            f"[Calibration][dE_mask_dist]"
            f" n={cat_dE_mask.numel()}"
            f" mean={cat_dE_mask.mean():.6f} std={cat_dE_mask.std():.6f}"
            f" p50={meqs[0]:.6f} p90={meqs[1]:.6f} p95={meqs[2]:.6f}"
            f" p99={meqs[3]:.6f} p999={meqs[4]:.6f}"
            f" heavy_tail_ratio={heavy_mE:.4f}"
        )
        print(
            f"[Calibration][dP_mask_dist]"
            f" n={cat_dP_mask.numel()}"
            f" mean={cat_dP_mask.mean():.6f} std={cat_dP_mask.std():.6f}"
            f" p50={mpqs[0]:.6f} p90={mpqs[1]:.6f} p95={mpqs[2]:.6f}"
            f" p99={mpqs[3]:.6f} p999={mpqs[4]:.6f}"
            f" heavy_tail_ratio={heavy_mP:.4f}"
        )
    else:
        delta_E_mask = 0.0
        delta_P_mask = 0.0

    return delta_E, delta_P, delta_E_mask, delta_P_mask


def _sq(t: torch.Tensor, q: float) -> float:
    """Safe quantile on a flattened tensor; returns nan if empty."""
    t = t.float().flatten()
    return float(torch.quantile(t, q).item()) if t.numel() > 0 else float("nan")


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two tensors (flattened)."""
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a * b).sum() / denom)


def _norm_or_nan(params) -> float:
    total = sum(p.data.norm() ** 2 for p in params if p is not None)
    return float(total ** 0.5) if total > 0 else float("nan")


def _grad_norm_or_nan(params) -> float:
    grads = [p.grad for p in params if p is not None and p.grad is not None]
    if not grads:
        return float("nan")
    total = sum(g.norm() ** 2 for g in grads)
    return float(total ** 0.5)


@torch.no_grad()
def collect_and_print_debug(
    prefix: str,
    model: nn.Module,
    cfg: TrainConfig,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
    y_pred: torch.Tensor | None = None,
    y_true: torch.Tensor | None = None,
    step_info: dict | None = None,
    grad_info: dict | None = None,
) -> None:
    """Print a fixed set of diagnostic fields for one (prefix, batch) pair.

    Each logical group is printed on its own line, prefixed with
    [{prefix}][CATEGORY], fields separated by spaces.  Missing values → nan.
    """
    nm = getattr(model, "nm", None)
    gate = getattr(nm, "gate", None) if nm is not None else None
    nan = float("nan")
    eps = 1e-12

    # ------------------------------------------------------------------ RECON
    x = getattr(nm, "_last_x_time", None)
    r = getattr(nm, "_last_r_time", None)
    n = getattr(nm, "_last_n_time", None)
    if x is not None and r is not None and n is not None:
        recon = x - (r + n)
        x_norm = x.norm().clamp(min=eps)
        recon_rel = float(recon.norm() / x_norm)
        recon_max_abs = float(recon.abs().max())
        ratio_n_time = float(n.pow(2).mean() / (x.pow(2).mean() + eps))
        ratio_r_time = float(r.pow(2).mean() / (x.pow(2).mean() + eps))
        corr_x_r = _pearson(x, r)
        corr_x_n = _pearson(x, n)
    else:
        recon_rel = recon_max_abs = ratio_n_time = ratio_r_time = nan
        corr_x_r = corr_x_n = nan
    print(
        f"[{prefix}][RECON]"
        f" recon_rel={recon_rel:.6f} recon_max_abs={recon_max_abs:.6e}"
        f" ratio_n_time={ratio_n_time:.6f} ratio_r_time={ratio_r_time:.6f}"
        f" corr_x_r={corr_x_r:.6f} corr_x_n={corr_x_n:.6f}"
    )

    # ------------------------------------------------------------------ STAT
    dE = getattr(nm, "_last_dE", None)
    dP = getattr(nm, "_last_dP", None)
    delta_E = float(cfg.delta_E) if nm is None else float(getattr(nm, "delta_E", cfg.delta_E))
    delta_P = float(cfg.delta_P) if nm is None else float(getattr(nm, "delta_P", cfg.delta_P))
    if dE is not None:
        dE_f = dE.float().flatten()
        mask_E = dE_f > delta_E
        trigger_E = float(mask_E.float().mean())
        excess_E = float((dE_f[mask_E] - delta_E).mean()) if mask_E.any() else nan
        mean_dE = float(dE_f.mean())
        p95_dE, p99_dE = _sq(dE_f, 0.95), _sq(dE_f, 0.99)
    else:
        trigger_E = excess_E = mean_dE = p95_dE = p99_dE = nan
    if dP is not None:
        dP_f = dP.float().flatten()
        mask_P = dP_f > delta_P
        trigger_P = float(mask_P.float().mean())
        excess_P = float((dP_f[mask_P] - delta_P).mean()) if mask_P.any() else nan
        mean_dP = float(dP_f.mean())
        p95_dP, p99_dP = _sq(dP_f, 0.95), _sq(dP_f, 0.99)
    else:
        trigger_P = excess_P = mean_dP = p95_dP = p99_dP = nan
    print(
        f"[{prefix}][STAT]"
        f" delta_E={delta_E:.6f} delta_P={delta_P:.6f}"
        f" trigger_E={trigger_E:.6f} trigger_P={trigger_P:.6f}"
        f" excess_E={excess_E:.6e} excess_P={excess_P:.6e}"
        f" mean_dE={mean_dE:.6f} mean_dP={mean_dP:.6f}"
        f" p95_dE={p95_dE:.6f} p99_dE={p99_dE:.6f}"
        f" p95_dP={p95_dP:.6f} p99_dP={p99_dP:.6f}"
    )

    # ------------------------------------------------------------------ REVIN
    inst_std = getattr(nm, "_last_inst_std", None)
    if inst_std is not None:
        s = inst_std.float().flatten()
        std_min = float(s.min())
        std_p1 = _sq(s, 0.01)
        std_p50 = _sq(s, 0.50)
        std_p99 = _sq(s, 0.99)
        small_frac = float((s < 1e-3).float().mean())
    else:
        std_min = std_p1 = std_p50 = std_p99 = small_frac = nan
    print(
        f"[{prefix}][REVIN]"
        f" inst_std_min={std_min:.6e} inst_std_p1={std_p1:.6e}"
        f" inst_std_p50={std_p50:.6f} inst_std_p99={std_p99:.6f}"
        f" inst_std_small_frac={small_frac:.6f}"
    )

    # ------------------------------------------------------------------ GATE
    g = getattr(nm, "_last_gate", None)
    gate_stats = nm.get_last_gate_stats() if nm is not None and hasattr(nm, "get_last_gate_stats") else {}
    gate_mean = gate_stats.get("gate_mean", nan)
    gate_maxF = gate_stats.get("gate_max_f", nan)
    gate_sumF = gate_stats.get("gate_sum_f", nan)
    gate_entF = gate_stats.get("gate_ent_f", nan)
    if g is not None:
        g_f = g.float().flatten()
        gate_sat0 = float((g_f < 0.01).float().mean())
        gate_sat1 = float((g_f > 0.99).float().mean())
    else:
        gate_sat0 = gate_sat1 = nan
    if gate is not None and hasattr(gate, "threshold") and gate.threshold is not None:
        thr = gate.threshold.float().flatten()
        thr_mean = float(thr.mean())
        thr_std = float(thr.std()) if thr.numel() > 1 else 0.0
        thr_min = float(thr.min())
        thr_max = float(thr.max())
    else:
        thr_mean = thr_std = thr_min = thr_max = nan
    logits = getattr(gate, "_last_logits", None) if gate is not None else None
    if logits is not None:
        lf = logits.float().flatten()
        lgts_mean = float(lf.mean())
        lgts_std = float(lf.std())
        lgts_p1 = _sq(lf, 0.01)
        lgts_p99 = _sq(lf, 0.99)
    else:
        lgts_mean = lgts_std = lgts_p1 = lgts_p99 = nan
    print(
        f"[{prefix}][GATE]"
        f" gate_mean={gate_mean:.6f} gate_maxF={gate_maxF:.6f}"
        f" gate_sumF={gate_sumF:.6f} gate_entF={gate_entF:.6f}"
        f" gate_sat0={gate_sat0:.6f} gate_sat1={gate_sat1:.6f}"
        f" thr_mean={thr_mean:.6f} thr_std={thr_std:.6f}"
        f" thr_min={thr_min:.6f} thr_max={thr_max:.6f}"
        f" lgts_mean={lgts_mean:.6f} lgts_std={lgts_std:.6f}"
        f" lgts_p1={lgts_p1:.6f} lgts_p99={lgts_p99:.6f}"
    )

    # ------------------------------------------------------------------ MASK (trigger mask)
    trigger_mask_on = bool(getattr(cfg, "trigger_mask", False))
    if nm is not None and trigger_mask_on:
        m_rate = float(getattr(nm, "_last_mask_rate", nan))
        m_trig_rate = float(getattr(nm, "_last_mask_trig_rate", nan))
        m_delta_E = float(getattr(nm, "_last_mask_delta_E", nan))
        m_delta_P = float(getattr(nm, "_last_mask_delta_P", nan))
    else:
        m_rate = 1.0
        m_trig_rate = 0.0
        m_delta_E = nan
        m_delta_P = nan
    print(
        f"[{prefix}][MASK]"
        f" mask_rate={m_rate:.6f}"
        f" mask_trig_rate={m_trig_rate:.6f}"
        f" delta_E_mask={m_delta_E:.6f}"
        f" delta_P_mask={m_delta_P:.6f}"
    )

    # ------------------------------------------------------------------ GFLK (gate flicker)
    if g is not None:
        flicker_t = float(g[..., 1:].sub(g[..., :-1]).abs().mean()) if g.shape[-1] > 1 else nan
        rough_f = float(g[:, :, 1:, :].sub(g[:, :, :-1, :]).abs().mean()) if g.shape[2] > 1 else nan
    else:
        flicker_t = rough_f = nan
    print(
        f"[{prefix}][GFLK]"
        f" flicker_t={flicker_t:.6f} rough_f={rough_f:.6f}"
    )

    # ------------------------------------------------------------------ PRED (branch usage)
    pnt_enabled = bool(getattr(cfg, "predict_n_time", False))
    pred_n_time_cache = getattr(nm, "_last_pred_n_time", None)
    pred_n_time_exists = pred_n_time_cache is not None
    denorm_pred = int(getattr(nm, "_denorm_used_pred", 0))
    denorm_input = int(getattr(nm, "_denorm_used_input", 0))
    denorm_extrap = int(getattr(nm, "_denorm_used_extrap", 0))
    print(
        f"[{prefix}][PRED]"
        f" predict_n_time={int(pnt_enabled)}"
        f" pred_n_time_exists={int(pred_n_time_exists)}"
        f" denorm_pred={denorm_pred}"
        f" denorm_input={denorm_input}"
        f" denorm_extrap={denorm_extrap}"
    )

    # ------------------------------------------------------------------ PACC (predictor accuracy)
    if pnt_enabled and pred_n_time_exists and batch_x is not None and batch_y is not None and nm is not None:
        try:
            x_full = torch.cat([batch_x.float(), batch_y.float()], dim=1)
            oracle_n_full = nm.extract_n_time_only(x_full.to(next(nm.parameters()).device))
            pred_len = cfg.pred_len
            oracle_n_future = oracle_n_full[:, -pred_len:, :]
            p_pred = pred_n_time_cache.float()
            o_gt = oracle_n_future.float()
            if p_pred.shape == o_gt.shape:
                mse_n = float(((p_pred - o_gt) ** 2).mean())
                rel_err_n = float((p_pred - o_gt).norm() / (o_gt.norm().clamp(min=eps)))
                corr_n = _pearson(p_pred, o_gt)
            else:
                mse_n = rel_err_n = corr_n = nan
        except Exception:
            mse_n = rel_err_n = corr_n = nan
        print(
            f"[{prefix}][PACC]"
            f" mse_n={mse_n:.6e} rel_err_n={rel_err_n:.6f} corr_n={corr_n:.6f}"
        )

    # ------------------------------------------------------------------ GRAD
    gi = grad_info or {}
    print(
        f"[{prefix}][GRAD]"
        f" gate_grad_norm={gi.get('gate_grad_norm', nan):.6e}"
        f" gate_param_norm={gi.get('gate_param_norm', nan):.6f}"
        f" pred_grad_norm={gi.get('pred_grad_norm', nan):.6e}"
        f" pred_param_norm={gi.get('pred_param_norm', nan):.6f}"
        f" update_ratio_predictor={gi.get('update_ratio_predictor', nan):.6e}"
    )

    # ------------------------------------------------------------------ MASKCOV
    if nm is not None and hasattr(nm, "get_last_mask_coverage_stats"):
        cov_stats = nm.get_last_mask_coverage_stats()
        _cov = cov_stats.get("cov", nan)
        _mr95 = cov_stats.get("min_rate_cov95", nan)
        _mr99 = cov_stats.get("min_rate_cov99", nan)
        _mr995 = cov_stats.get("min_rate_cov995", nan)
    else:
        _cov = _mr95 = _mr99 = _mr995 = nan
    print(
        f"[{prefix}][MASKCOV]"
        f" cov={_cov:.6f}"
        f" min_rate@95={_mr95:.6f}"
        f" min_rate@99={_mr99:.6f}"
        f" min_rate@995={_mr995:.6f}"
    )

    # ------------------------------------------------------------------ MASKP
    if nm is not None and hasattr(nm, "get_last_mask_coverage_stats"):
        _p_table = nm.get_last_mask_coverage_stats().get("p_table", {})
        _p950 = _p_table.get(0.95, (nan, nan))
        _p990 = _p_table.get(0.99, (nan, nan))
        _p995 = _p_table.get(0.995, (nan, nan))
        print(
            f"[{prefix}][MASKP]"
            f" p=0.950:mr={_p950[0]:.6f},cov={_p950[1]:.6f}"
            f" | p=0.990:mr={_p990[0]:.6f},cov={_p990[1]:.6f}"
            f" | p=0.995:mr={_p995[0]:.6f},cov={_p995[1]:.6f}"
        )
    else:
        print(f"[{prefix}][MASKP] p=0.950:mr={nan:.6f},cov={nan:.6f} | p=0.990:mr={nan:.6f},cov={nan:.6f} | p=0.995:mr={nan:.6f},cov={nan:.6f}")

    # ------------------------------------------------------------------ PSUP
    _pred_sup_mse = float(getattr(nm, "_last_pred_sup_loss", nan)) if nm is not None else nan
    print(f"[{prefix}][PSUP] pred_sup_mse={_pred_sup_mse:.6e}")

    # ------------------------------------------------------------------ GQ (gate quality diagnostics)
    if nm is not None and hasattr(nm, "get_last_gq_stats"):
        _gq = nm.get_last_gq_stats()
        _gq_entF = _gq.get("entF_norm", nan)
        _gq_topk = _gq.get("topk_mass", nan)
        _gq_ratio = _gq.get("maxF_meanF", nan)
        _gq_corr = _gq.get("corr_mag", nan)
    else:
        _gq_entF = _gq_topk = _gq_ratio = _gq_corr = nan
    print(
        f"[{prefix}][GQ]"
        f" entF_norm={_gq_entF:.4f}"
        f" topk_mass={_gq_topk:.4f}"
        f" maxF_meanF={_gq_ratio:.4f}"
        f" corr_mag={_gq_corr:.4f}"
    )


def _aux_scale(cfg: TrainConfig, epoch_idx: int) -> float:
    if cfg.aux_loss_schedule == "none":
        return float(cfg.aux_loss_scale)
    if cfg.aux_loss_schedule != "cosine":
        raise ValueError(
            f"Unsupported aux_loss_schedule: {cfg.aux_loss_schedule}. "
            "Use 'none' or 'cosine'."
        )
    start = int(cfg.aux_loss_decay_start_epoch)
    if epoch_idx < start:
        return float(cfg.aux_loss_scale)
    decay_epochs = max(int(cfg.aux_loss_decay_epochs), 1)
    t = min(epoch_idx - start, decay_epochs) / float(decay_epochs)
    # Cosine anneal auxiliary loss weight from scale -> min_scale.
    cos_factor = 0.5 * (1.0 + np.cos(np.pi * t))
    return float(cfg.aux_loss_min_scale + (cfg.aux_loss_scale - cfg.aux_loss_min_scale) * cos_factor)


def _build_optimizer(model: nn.Module, cfg: TrainConfig) -> Adam:
    base_lr = float(cfg.lr)
    gate_wd = float(cfg.gate_weight_decay)
    pred_wd = float(cfg.predictor_weight_decay)
    
    # Separate learning rates (0 means use base_lr)
    gate_lr = float(cfg.gate_lr) if cfg.gate_lr > 0 else base_lr
    predictor_lr = float(cfg.predictor_lr) if cfg.predictor_lr > 0 else base_lr

    gate_params: list[torch.nn.Parameter] = []
    pred_params: list[torch.nn.Parameter] = []
    nm = getattr(model, "nm", None)
    if nm is not None:
        gate_mod = getattr(nm, "gate", None)
        if gate_mod is not None:
            gate_params = [p for p in gate_mod.parameters() if p.requires_grad]
        pred_mod = getattr(nm, "n_tf_predictor", None)
        if pred_mod is not None:
            pred_params = [p for p in pred_mod.parameters() if p.requires_grad]

    gate_ids = {id(p) for p in gate_params}
    pred_ids = {id(p) for p in pred_params}
    remaining = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in gate_ids and id(p) not in pred_ids
    ]

    groups: list[dict[str, object]] = []
    if remaining:
        groups.append({"params": remaining, "weight_decay": cfg.weight_decay, "lr": base_lr})
    if gate_params:
        groups.append({"params": gate_params, "weight_decay": gate_wd, "lr": gate_lr})
    if pred_params:
        groups.append({"params": pred_params, "weight_decay": pred_wd, "lr": predictor_lr})

    return Adam(groups, lr=base_lr)


def _get_nm_params(model: nn.Module) -> tuple[list, list]:
    """Return (gate_params, pred_params) from model.nm."""
    nm = getattr(model, "nm", None)
    gate_params: list = []
    pred_params: list = []
    if nm is not None:
        gate_mod = getattr(nm, "gate", None)
        if gate_mod is not None:
            gate_params = [p for p in gate_mod.parameters() if p.requires_grad]
        pred_mod = getattr(nm, "n_tf_predictor", None)
        if pred_mod is not None:
            pred_params = [p for p in pred_mod.parameters() if p.requires_grad]
    return gate_params, pred_params


def train_one_epoch(model, loader, optimizer, cfg, scaler, epoch_idx: int):
    model.train()
    loss_fn = nn.MSELoss()
    losses = []
    task_losses = []
    aux_losses = []
    stat_losses: list[float] = []
    l_e_vals: list[float] = []
    l_p_vals: list[float] = []
    pred_sup_mse_vals: list[float] = []
    has_nm_stat = (
        hasattr(model, "nm")
        and model.nm is not None
        and hasattr(model.nm, "get_last_stationarity_stats")
    )
    # Determine if pred supervision loss should be computed each batch
    _nm_for_pred = getattr(model, "nm", None)
    _use_pred_sup = (
        bool(getattr(cfg, "predict_n_time", False))
        and float(getattr(cfg, "pred_loss_weight", 0.0)) > 0.0
        and _nm_for_pred is not None
        and hasattr(_nm_for_pred, "pred_supervision_loss")
    )
    aux_scale = _aux_scale(cfg, epoch_idx)
    print(f"aux_loss_scale: {aux_scale:.6f}")

    # Reset denorm branch counters at epoch start
    nm = getattr(model, "nm", None)
    if nm is not None:
        nm._denorm_used_pred = 0
        nm._denorm_used_input = 0
        nm._denorm_used_extrap = 0

    gate_params, pred_params = _get_nm_params(model)
    first_batch_done = False

    with tqdm(total=len(loader.dataset), leave=True) as pbar:
        for batch_x, batch_y, origin_y, batch_x_enc, batch_y_enc in loader:
            batch_x = batch_x.to(cfg.device).float()
            batch_y = batch_y.to(cfg.device).float()
            batch_x_enc = batch_x_enc.to(cfg.device).float()
            batch_y_enc = batch_y_enc.to(cfg.device).float()
            origin_y = origin_y.to(cfg.device).float()

            dec_inp, dec_inp_enc = None, None
            if model.is_former:
                label_len = min(cfg.label_len or (cfg.window // 2), batch_x.size(1))
                dec_inp, dec_inp_enc = _make_dec_inputs(
                    batch_x, batch_y_enc, batch_x_enc, label_len
                )

            optimizer.zero_grad()
            pred = model(batch_x, batch_x_enc, dec_inp, dec_inp_enc)
            true = batch_y
            if cfg.invtrans_loss:
                pred = scaler.inverse_transform(pred)
                true = origin_y

            task_loss = loss_fn(pred, true)
            aux_loss = torch.tensor(0.0, device=task_loss.device)
            if hasattr(model.nm, "loss_with_target"):
                aux_loss = model.nm.loss_with_target(true)
            elif hasattr(model.nm, "loss"):
                aux_loss = model.nm.loss(true)

            # Prediction supervision loss (teacher extraction oracle)
            pred_sup_loss = torch.tensor(0.0, device=task_loss.device)
            if _use_pred_sup:
                pred_sup_loss = _nm_for_pred.pred_supervision_loss(batch_y)
                pred_sup_mse_vals.append(float(_nm_for_pred._last_pred_sup_loss))

            loss = task_loss + aux_loss * aux_scale + cfg.pred_loss_weight * pred_sup_loss

            # --- First-batch gradient / update-ratio collection ---
            if not first_batch_done:
                # Snapshot param values before optimizer.step (for update ratio)
                pred_snap = [p.data.clone() for p in pred_params]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            if not first_batch_done:
                gate_grad_norm = _grad_norm_or_nan(gate_params)
                gate_param_norm = _norm_or_nan(gate_params)
                pred_grad_norm = _grad_norm_or_nan(pred_params)
                pred_param_norm = _norm_or_nan(pred_params)

            optimizer.step()

            if not first_batch_done:
                if pred_params and pred_snap:
                    num = float(sum((p.data - s).norm() ** 2 for p, s in zip(pred_params, pred_snap)) ** 0.5)
                    den = float(sum(s.norm() ** 2 for s in pred_snap) ** 0.5) + 1e-12
                    update_ratio_pred = num / den
                else:
                    update_ratio_pred = float("nan")
                grad_info = {
                    "gate_grad_norm": gate_grad_norm,
                    "gate_param_norm": gate_param_norm,
                    "pred_grad_norm": pred_grad_norm,
                    "pred_param_norm": pred_param_norm,
                    "update_ratio_predictor": update_ratio_pred,
                }
                collect_and_print_debug(
                    "TRAIN_DBG", model, cfg,
                    batch_x=batch_x.detach().cpu(),
                    batch_y=batch_y.detach().cpu(),
                    y_pred=pred.detach().cpu(),
                    y_true=true.detach().cpu(),
                    grad_info=grad_info,
                )
                first_batch_done = True

            losses.append(loss.item())
            task_losses.append(task_loss.item())
            aux_losses.append(aux_loss.item())
            if has_nm_stat:
                s = model.nm.get_last_stationarity_stats()
                stat_losses.append(s["stationarity_loss"])
                l_e_vals.append(s["L_E"])
                l_p_vals.append(s["L_P"])
            pbar.update(batch_x.size(0))
            pbar.set_postfix(
                task=f"{task_loss.item():.4f}",
                aux=f"{aux_loss.item():.4f}",
                total=f"{loss.item():.4f}",
            )
    
    # Get gate statistics from model
    gate_stats_str = ""
    if hasattr(model, "nm") and model.nm is not None and hasattr(model.nm, "get_last_gate_stats"):
        stats = model.nm.get_last_gate_stats()
        gate_stats_str = (
            f" | gate: mean={stats['gate_mean']:.4f}, "
            f"sumF={stats['gate_sum_f']:.4f}, maxF={stats['gate_max_f']:.4f}, "
            f"entF={stats['gate_ent_f']:.4f}"
        )

    train_stat: dict[str, float] = {}
    if stat_losses:
        train_stat = {
            "stationarity_loss": float(np.mean(stat_losses)),
            "L_E": float(np.mean(l_e_vals)),
            "L_P": float(np.mean(l_p_vals)),
            "task_loss": float(np.mean(task_losses)),
            "aux_loss": float(np.mean(aux_losses)),
        }
    if pred_sup_mse_vals:
        train_stat["pred_sup_mse"] = float(np.mean(pred_sup_mse_vals))

    return float(np.mean(losses)) if losses else 0.0, gate_stats_str, train_stat


@torch.no_grad()
def evaluate(model, loader, cfg, scaler, debug_prefix: str | None = None):
    model.eval()
    metrics = _build_metrics(torch.device(cfg.device))
    for metric in metrics.values():
        metric.reset()
    has_nm_stat = (
        hasattr(model, "nm")
        and model.nm is not None
        and hasattr(model.nm, "get_last_stationarity_stats")
        and hasattr(model.nm, "loss")
    )
    stat_losses: list[float] = []
    l_e_vals: list[float] = []
    l_p_vals: list[float] = []
    pred_sup_mse_vals: list[float] = []
    _nm_eval = getattr(model, "nm", None)
    _use_pred_sup_eval = (
        bool(getattr(cfg, "predict_n_time", False))
        and float(getattr(cfg, "pred_loss_weight", 0.0)) > 0.0
        and _nm_eval is not None
        and hasattr(_nm_eval, "pred_supervision_loss")
    )
    first_batch_done = False
    for batch_x, batch_y, origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()
        origin_y = origin_y.to(cfg.device).float()

        dec_inp, dec_inp_enc = None, None
        if model.is_former:
            label_len = min(cfg.label_len or (cfg.window // 2), batch_x.size(1))
            dec_inp, dec_inp_enc = _make_dec_inputs(
                batch_x, batch_y_enc, batch_x_enc, label_len
            )
        pred = model(batch_x, batch_x_enc, dec_inp, dec_inp_enc)
        if has_nm_stat:
            # Compute stationarity stats from the forward pass (_last_r_tf is set)
            model.nm.loss()
            s = model.nm.get_last_stationarity_stats()
            stat_losses.append(s["stationarity_loss"])
            l_e_vals.append(s["L_E"])
            l_p_vals.append(s["L_P"])

        # Prediction supervision tracking (no grad; updates _last_pred_sup_loss cache)
        if _use_pred_sup_eval:
            _nm_eval.pred_supervision_loss(batch_y)
            pred_sup_mse_vals.append(float(_nm_eval._last_pred_sup_loss))

        # First-batch debug print for val/test
        if debug_prefix is not None and not first_batch_done:
            collect_and_print_debug(
                debug_prefix, model, cfg,
                batch_x=batch_x.detach().cpu(),
                batch_y=batch_y.detach().cpu(),
                y_pred=pred.detach().cpu(),
                y_true=batch_y.detach().cpu(),
            )
            first_batch_done = True

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
    if stat_losses:
        results["stationarity_loss"] = float(np.mean(stat_losses))
        results["L_E"] = float(np.mean(l_e_vals))
        results["L_P"] = float(np.mean(l_p_vals))
    if pred_sup_mse_vals:
        results["pred_sup_mse"] = float(np.mean(pred_sup_mse_vals))
    return results


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    defaults = asdict(TrainConfig())
    for field, value in defaults.items():
        arg_name = f"--{field.replace('_', '-')}"
        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value)
            parser.add_argument(f"--no-{field.replace('_', '-')}", dest=field, action="store_false")
            continue
        arg_type = type(value)
        if value is None and field in {"n_fft", "hop_length", "win_length"}:
            arg_type = int
        parser.add_argument(arg_name, type=arg_type, default=value)
    args = parser.parse_args(argv)
    cfg = TrainConfig(**vars(args))

    _set_seed(cfg.seed)
    dataloader, dataset, scaler, split_info = _build_dataloader(cfg)

    # Log the exact row boundaries used for this run (for reproducibility)
    si = split_info
    print(
        f"[Split] type={si['split_type']}  dataset_len={si['dataset_len']}"
        f"  train={si['train_rows']} rows {si['train_idx']}"
        f"  val={si['val_rows']} rows {si['val_idx']}"
        f"  test={si['test_rows']} rows {si['test_idx']}"
    )
    print(
        f"[Split] dataloader sizes: "
        f"train={dataloader.train_size}, val={dataloader.val_size}, test={dataloader.test_size}"
    )

    model = build_model(cfg, dataset.num_features).to(cfg.device)

    # Auto-calibrate stationarity margin thresholds from training data
    # Skipped when user explicitly supplied non-zero delta_E or delta_P
    if cfg.auto_thresholds and cfg.delta_E == 0.0 and cfg.delta_P == 0.0:
        nm = getattr(model, "nm", None)
        if nm is not None and hasattr(nm, "delta_E"):
            delta_E, delta_P, delta_E_mask, delta_P_mask = calibrate_thresholds(
                model, dataloader.train_loader, cfg, max_batches=200
            )
            cfg.delta_E = delta_E
            cfg.delta_P = delta_P
            cfg.delta_E_mask = delta_E_mask
            cfg.delta_P_mask = delta_P_mask
            nm.delta_E = delta_E
            nm.delta_P = delta_P
            nm.delta_E_mask = delta_E_mask
            nm.delta_P_mask = delta_P_mask
            print(
                f"[Calibration] trigger_q={cfg.trigger_q}"
                f"  delta_E={delta_E:.6f}  delta_P={delta_P:.6f}"
                f"  delta_E_mask={delta_E_mask:.6f}  delta_P_mask={delta_P_mask:.6f}"
                f"  trigger_mask={cfg.trigger_mask}"
            )

    optimizer = _build_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    run_name = cfg.run_name or f"{cfg.dataset_type}_{cfg.backbone_type}_{cfg.norm_type}"
    result_root = cfg.result_dir
    if cfg.norm_type.lower() in {"none", "baseline", "no"}:
        result_root = os.path.join(cfg.result_dir, cfg.baseline_subdir)
    os.makedirs(result_root, exist_ok=True)
    result_path = os.path.join(result_root, f"{run_name}.json")
    best_checkpoint_path = os.path.join(result_root, f"{run_name}_best.pth")

    print(f"Train config: {cfg}")
    if cfg.early_stop_metric not in {"val_mse", "ema_val_mse"}:
        raise ValueError(
            f"Unsupported early_stop_metric: {cfg.early_stop_metric}. "
            "Use 'val_mse' or 'ema_val_mse'."
        )
    if not (0.0 < cfg.val_mse_ema_alpha <= 1.0):
        raise ValueError("val_mse_ema_alpha must be in (0, 1].")
    if cfg.aux_loss_schedule not in {"none", "cosine"}:
        raise ValueError("aux_loss_schedule must be 'none' or 'cosine'.")

    best_val = float("inf")
    best_monitor = float("inf")
    best_epoch = -1
    best_val_at_best = float("inf")
    best_test_at_best = float("inf")
    best_test_seen = float("inf")
    best_test_seen_epoch = -1
    ema_val_mse = None
    no_improve_count = 0

    for epoch in range(cfg.epochs):
        result = train_one_epoch(
            model, dataloader.train_loader, optimizer, cfg, scaler, epoch_idx=epoch
        )
        if isinstance(result, tuple) and len(result) == 3:
            train_loss, gate_info, train_stat = result
        elif isinstance(result, tuple):
            train_loss, gate_info = result
            train_stat = {}
        else:
            train_loss = result
            gate_info = ""
            train_stat = {}
        val_metrics = evaluate(model, dataloader.val_loader, cfg, scaler, debug_prefix="VAL_DBG")
        test_metrics = evaluate(model, dataloader.test_loader, cfg, scaler)

        # Build stationarity info strings
        train_stat_str = ""
        if train_stat:
            train_stat_str = (
                f" | task_loss={train_stat['task_loss']:.6f}"
                f", stat_loss={train_stat['stationarity_loss']:.6f}"
                f" (L_E={train_stat['L_E']:.6f}, L_P={train_stat['L_P']:.6f})"
            )
            if "pred_sup_mse" in train_stat:
                train_stat_str += f", pred_sup_mse={train_stat['pred_sup_mse']:.6e}"
        val_stat_str = ""
        if "stationarity_loss" in val_metrics:
            val_stat_str = (
                f" | stat_loss={val_metrics['stationarity_loss']:.6f}"
                f" (L_E={val_metrics['L_E']:.6f}, L_P={val_metrics['L_P']:.6f})"
            )
        if "pred_sup_mse" in val_metrics:
            val_stat_str += f", pred_sup_mse={val_metrics['pred_sup_mse']:.6e}"

        print(f"Epoch: {epoch + 1} Training loss: {train_loss:.6f}{gate_info}{train_stat_str}")
        print(
            "vali_results: "
            f"{{'mae': {val_metrics['mae']:.6f}, "
            f"'mape': {val_metrics['mape']:.6f}, "
            f"'mse': {val_metrics['mse']:.6f}, "
            f"'rmse': {val_metrics['rmse']:.6f}}}"
            f"{val_stat_str}"
        )
        print(
            "test_results: "
            f"{{'mae': {test_metrics['mae']:.6f}, "
            f"'mape': {test_metrics['mape']:.6f}, "
            f"'mse': {test_metrics['mse']:.6f}, "
            f"'rmse': {test_metrics['rmse']:.6f}}}"
        )

        best_val = min(best_val, val_metrics["mse"])
        if test_metrics["mse"] < best_test_seen:
            best_test_seen = test_metrics["mse"]
            best_test_seen_epoch = epoch + 1

        if ema_val_mse is None:
            ema_val_mse = val_metrics["mse"]
        else:
            alpha = cfg.val_mse_ema_alpha
            ema_val_mse = alpha * val_metrics["mse"] + (1.0 - alpha) * ema_val_mse

        monitor_value = (
            ema_val_mse if cfg.early_stop_metric == "ema_val_mse" else val_metrics["mse"]
        )
        improved = monitor_value < (best_monitor - cfg.early_stop_delta)
        if improved:
            print(
                f"Monitor improved ({best_monitor:.6f} --> {monitor_value:.6f}), "
                "saving model ..."
            )
            best_monitor = monitor_value
            best_epoch = epoch + 1
            best_val_at_best = val_metrics["mse"]
            best_test_at_best = test_metrics["mse"]
            torch.save(model.state_dict(), best_checkpoint_path)
            no_improve_count = 0
        elif cfg.early_stop and (epoch + 1) >= cfg.early_stop_min_epochs:
            no_improve_count += 1
            print(
                f"EarlyStopping counter: {no_improve_count} out of {cfg.early_stop_patience}"
            )
            if no_improve_count >= cfg.early_stop_patience:
                print(
                    f"loss no decreased for {cfg.early_stop_patience} epochs,  early stopping ...."
                )
                break

        scheduler.step()

    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=cfg.device))

    test_metrics = evaluate(model, dataloader.test_loader, cfg, scaler)
    print(
        "test_results: "
        f"{{'mae': {test_metrics['mae']:.6f}, "
        f"'mape': {test_metrics['mape']:.6f}, "
        f"'mse': {test_metrics['mse']:.6f}, "
        f"'rmse': {test_metrics['rmse']:.6f}}}"
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "dataset": cfg.dataset_type,
                "backbone": cfg.backbone_type,
                "norm_type": cfg.norm_type,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "window": cfg.window,
                "pred_len": cfg.pred_len,
                "horizon": cfg.horizon,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "vali_results": val_metrics,
                "test_results": test_metrics,
                "early_stop": cfg.early_stop,
                "early_stop_patience": cfg.early_stop_patience,
                "early_stop_min_epochs": cfg.early_stop_min_epochs,
                "early_stop_metric": cfg.early_stop_metric,
                "val_mse_ema_alpha": cfg.val_mse_ema_alpha,
                "early_stop_delta": cfg.early_stop_delta,
                "best_val_mse": best_val,
                "best_monitor": best_monitor,
                "best_epoch": best_epoch,
                "best_val_mse_at_best_epoch": best_val_at_best,
                "best_test_mse_at_best_epoch": best_test_at_best,
                "best_test_mse_seen_any_epoch": best_test_seen,
                "best_test_mse_seen_any_epoch_epoch": best_test_seen_epoch,
                "config": asdict(cfg),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
