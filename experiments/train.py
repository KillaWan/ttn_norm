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
from ttn_norm.normalizations import DishTS, FAN, No, RevIN, SAN, TFBackgroundNorm
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
    gate_use_log_mag: bool = True
    future_mode: str = "repeat_last"
    gate_threshold: float = 0.0
    gate_temperature: float = 1.0
    gate_lr: float = 0.0
    use_instance_norm: bool = True
    eps_E: float = 1e-6
    auto_thresholds: bool = True
    trigger_q: float = 0.99
    target_frames: int = 24
    auto_stft: bool = True
    trigger_mask: bool = True
    delta_E_mask: float = 0.0
    delta_P_mask: float = 0.0
    trigger_mask_mode: str = "time"
    trigger_soft: bool = True
    trigger_soft_tau: float = 0.25
    ftsep5_feat_mode: str = "mdm"

    # New aux loss hyperparams
    easy_ar_weight: float = 0.0
    easy_ar_k: int = 8
    easy_ar_ridge: float = 1e-3
    white_acf_weight: float = 0.0
    white_acf_lags: int = 8
    shape_js_weight: float = 0.0
    shape_w1_weight: float = 0.0
    shape_weighting: str = "trigger"
    min_remove_weight: float = 0.0
    min_remove_mode: str = "ntf_l2"
    energy_tv_weight: float = 0.0
    n_ratio_min: float = 0.0
    n_ratio_max: float = 0.0
    n_ratio_weight: float = 0.0
    n_ratio_power: int = 2
    # N-predictor (teacher mask + add-back)
    teacher_mask_only: bool = False
    n_pred_weight: float = 0.0
    n_pred_arch: str = "mlp"
    # Low-rank gate hyperparams
    gate_lowrank_rank: int = 4
    gate_lowrank_time_ks: int = 5
    gate_lowrank_freq_ks: int = 3
    gate_lowrank_use_bias: bool = True
    gate_sparse_l1_weight: float = 0.0
    gate_lowrank_u_tv_weight: float = 0.0

    # ------------------------------------------------------------------ baseline norm configs
    # RevIN
    revin_affine: bool = True
    # FAN (frequency adaptive normalization)
    fan_freq_topk: int = 20
    fan_rfft: bool = True
    # DishTS
    dish_init: str = "uniform"   # "standard" | "avg" | "uniform"
    # SAN (seasonal adaptive normalization)
    san_period_len: int = 12
    # SAN spike-robust stat estimation (used by norm_type="san_spike")
    spike_q: float = 0.99
    spike_dilate: int = 1
    spike_mode: str = "mad"
    # TFBackgroundNorm (norm_type="tf_bg")
    tfbg_n_fft: int = 0        # 0 = auto (next power of 2 above T//4)
    tfbg_hop: int = 0          # 0 = auto (n_fft // 4)
    tfbg_time_kernel: int = 9
    tfbg_freq_kernel: int = 5
    tfbg_bmax: float = 2.0
    # Oracle gate ablation (LocalTFNorm only)
    gate_mode: str = "learned"       # "learned" | "oracle_train" | "oracle_eval"
    oracle_q: float = 0.99           # quantile for oracle trigger
    oracle_lambda_p: float = 0.25    # phase-change weight in proxy score
    oracle_dilate: int = 1           # temporal dilation passes (1 = base mask only)


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def build_model(cfg: TrainConfig, num_features: int) -> TTNModel:
    _nt = cfg.norm_type.lower()

    if _nt in {"none", "baseline", "no"}:
        norm_model: nn.Module | None = nn.Identity()

    elif _nt == "revin":
        norm_model = RevIN(n_series=num_features, affine=cfg.revin_affine)

    elif _nt == "fan":
        norm_model = FAN(
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            enc_in=num_features,
            freq_topk=cfg.fan_freq_topk,
            rfft=cfg.fan_rfft,
        )

    elif _nt == "dishts":
        norm_model = DishTS(
            n_series=num_features,
            seq_len=cfg.window,
            dish_init=cfg.dish_init,
        )

    elif _nt == "san":
        norm_model = SAN(
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            period_len=cfg.san_period_len,
            enc_in=num_features,
        )

    elif _nt == "san_spike":
        norm_model = SAN(
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            period_len=cfg.san_period_len,
            enc_in=num_features,
            spike_stats=True,
            spike_q=cfg.spike_q,
            spike_dilate=cfg.spike_dilate,
            spike_mode=cfg.spike_mode,
        )

    elif _nt == "tf_bg":
        norm_model = TFBackgroundNorm(
            n_fft=cfg.tfbg_n_fft,
            hop_length=cfg.tfbg_hop,
            time_kernel=cfg.tfbg_time_kernel,
            freq_kernel=cfg.tfbg_freq_kernel,
            bmax=cfg.tfbg_bmax,
        )

    else:
        # Default: LocalTF normalization
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
            gate_use_log_mag=cfg.gate_use_log_mag,
            future_mode=cfg.future_mode,
            gate_threshold=cfg.gate_threshold,
            gate_temperature=cfg.gate_temperature,
            use_instance_norm=cfg.use_instance_norm,
            eps_E=cfg.eps_E,
            trigger_mask=cfg.trigger_mask,
            delta_E_mask=cfg.delta_E_mask,
            delta_P_mask=cfg.delta_P_mask,
            trigger_mask_mode=cfg.trigger_mask_mode,
            trigger_soft=cfg.trigger_soft,
            trigger_soft_tau=cfg.trigger_soft_tau,
            ftsep5_feat_mode=cfg.ftsep5_feat_mode,
            easy_ar_weight=cfg.easy_ar_weight,
            easy_ar_k=cfg.easy_ar_k,
            easy_ar_ridge=cfg.easy_ar_ridge,
            white_acf_weight=cfg.white_acf_weight,
            white_acf_lags=cfg.white_acf_lags,
            shape_js_weight=cfg.shape_js_weight,
            shape_w1_weight=cfg.shape_w1_weight,
            shape_weighting=cfg.shape_weighting,
            min_remove_weight=cfg.min_remove_weight,
            min_remove_mode=cfg.min_remove_mode,
            energy_tv_weight=cfg.energy_tv_weight,
            n_ratio_min=cfg.n_ratio_min,
            n_ratio_max=cfg.n_ratio_max,
            n_ratio_weight=cfg.n_ratio_weight,
            n_ratio_power=cfg.n_ratio_power,
            teacher_mask_only=cfg.teacher_mask_only,
            n_pred_weight=cfg.n_pred_weight,
            n_pred_arch=cfg.n_pred_arch,
            gate_lowrank_rank=cfg.gate_lowrank_rank,
            gate_lowrank_time_ks=cfg.gate_lowrank_time_ks,
            gate_lowrank_freq_ks=cfg.gate_lowrank_freq_ks,
            gate_lowrank_use_bias=cfg.gate_lowrank_use_bias,
            gate_sparse_l1_weight=cfg.gate_sparse_l1_weight,
            gate_lowrank_u_tv_weight=cfg.gate_lowrank_u_tv_weight,
            gate_mode=cfg.gate_mode,
            oracle_q=cfg.oracle_q,
            oracle_lambda_p=cfg.oracle_lambda_p,
            oracle_dilate=cfg.oracle_dilate,
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
) -> tuple[float, float]:
    """Estimate delta_E_mask and delta_P_mask from training data.

    Runs up to *max_batches* forward passes, collects per-transition
    log-energy differences (dE) and spectral-shape differences (dP)
    from the raw input x_tf, then returns the trigger_q quantile of
    each as the trigger mask thresholds.

    Returns:
        (delta_E_mask, delta_P_mask)
    """
    nm = getattr(model, "nm", None)
    if nm is None or not hasattr(nm, "_last_x_tf"):
        return 0.0, 0.0

    model.eval()
    eps = 1e-8
    eps_E = float(getattr(nm, "eps_E", 1e-6))

    # Temporarily disable trigger_mask so forward passes succeed before thresholds are known
    _orig_trigger_mask = getattr(nm, "trigger_mask", False)
    nm.trigger_mask = False

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

        # dE/dP from raw input x_tf (for trigger mask thresholds)
        x_tf = nm._last_x_tf  # (B, C, F, T) complex, detached
        if x_tf is None:
            continue

        P_x = x_tf.abs() ** 2                                      # (B, C, F, T)
        E_x = P_x.mean(dim=2)                                      # (B, C, T)
        p_x = P_x / (P_x.sum(dim=2, keepdim=True) + eps)           # (B, C, F, T)
        logE_x = torch.log(E_x + eps_E)
        dE_x = torch.abs(logE_x[..., 1:] - logE_x[..., :-1])      # (B, C, T-1)
        dP_f_x = torch.abs(p_x[..., :, 1:] - p_x[..., :, :-1])    # (B, C, F, T-1)
        dP_mean_x = dP_f_x.mean(dim=2)                             # (B, C, T-1)

        all_dE_mask.append(dE_x.cpu().flatten())
        # For delta_P_mask: use per-F elements if trigger_mask_mode=="tf"
        if cfg.trigger_mask_mode == "tf":
            all_dP_mask.append(dP_f_x.cpu().flatten())
        else:
            all_dP_mask.append(dP_mean_x.cpu().flatten())

    # Restore original trigger_mask setting
    nm.trigger_mask = _orig_trigger_mask

    if not all_dE_mask:
        return 0.0, 0.0

    cat_dE = torch.cat(all_dE_mask)
    cat_dP = torch.cat(all_dP_mask)
    q = float(cfg.trigger_q)
    delta_E_mask = float(torch.quantile(cat_dE, q).item())
    delta_P_mask = float(torch.quantile(cat_dP, q).item())

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
        f"[Calibration][dE_mask_dist]"
        f" n={cat_dE.numel()}"
        f" mean={cat_dE.mean():.6f} std={cat_dE.std():.6f}"
        f" p50={eqs[0]:.6f} p90={eqs[1]:.6f} p95={eqs[2]:.6f}"
        f" p99={eqs[3]:.6f} p999={eqs[4]:.6f}"
        f" heavy_tail_ratio={heavy_E:.4f}"
    )
    print(
        f"[Calibration][dP_mask_dist]"
        f" n={cat_dP.numel()}"
        f" mean={cat_dP.mean():.6f} std={cat_dP.std():.6f}"
        f" p50={pqs[0]:.6f} p90={pqs[1]:.6f} p95={pqs[2]:.6f}"
        f" p99={pqs[3]:.6f} p999={pqs[4]:.6f}"
        f" heavy_tail_ratio={heavy_P:.4f}"
    )

    return delta_E_mask, delta_P_mask


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

    # Fetch summary dicts once
    gate_stats = nm.get_last_gate_stats() if nm is not None and hasattr(nm, "get_last_gate_stats") else {}
    decomp_stats = nm.get_last_decomp_stats() if nm is not None and hasattr(nm, "get_last_decomp_stats") else {}
    aux_stats = nm.get_last_aux_stats() if nm is not None and hasattr(nm, "get_last_aux_stats") else {}

    # ------------------------------------------------------------------ RECON
    x = getattr(nm, "_last_x_time", None)
    r = getattr(nm, "_last_r_time", None)
    n = getattr(nm, "_last_n_time", None)
    if x is not None and r is not None and n is not None:
        recon = x - (r + n)
        x_norm = x.norm().clamp(min=eps)
        recon_rel = float(recon.norm() / x_norm)
        recon_max_abs = float(recon.abs().max())
        ratio_r_time = float(r.pow(2).mean() / (x.pow(2).mean() + eps))
        corr_x_n = _pearson(x, n)
    else:
        recon_rel = recon_max_abs = ratio_r_time = nan
        corr_x_n = nan
    ratio_n_time = decomp_stats.get("ratio_n_time", nan)
    corr_x_r = decomp_stats.get("corr_x_r", nan)
    print(
        f"[{prefix}][RECON]"
        f" recon_rel={recon_rel:.6f} recon_max_abs={recon_max_abs:.6e}"
        f" ratio_n_time={ratio_n_time:.6f} ratio_r_time={ratio_r_time:.6f}"
        f" corr_x_r={corr_x_r:.6f} corr_x_n={corr_x_n:.6f}"
    )

    # ------------------------------------------------------------------ REVIN
    inst_std = getattr(nm, "_last_inst_std", None) if nm is not None else None
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
    gate_mean = gate_stats.get("gate_mean", nan)
    gate_maxF = gate_stats.get("gate_max_f", nan)
    gate_sumF = gate_stats.get("gate_sum_f", nan)
    gate_entF = gate_stats.get("gate_ent_f", nan)
    g_eff = getattr(nm, "_last_gate_eff", None) if nm is not None else None
    if g_eff is not None:
        g_f = g_eff.detach().float().flatten()
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
    w_mean = decomp_stats.get("w_mean", nan)
    w_max = decomp_stats.get("w_max", nan)
    delta_E_mask = float(getattr(nm, "delta_E_mask", nan)) if nm is not None else nan
    delta_P_mask = float(getattr(nm, "delta_P_mask", nan)) if nm is not None else nan
    print(
        f"[{prefix}][MASK]"
        f" w_mean={w_mean:.6f} w_max={w_max:.6f}"
        f" delta_E_mask={delta_E_mask:.6f} delta_P_mask={delta_P_mask:.6f}"
    )

    # ------------------------------------------------------------------ GFLK (gate flicker)
    g_raw = getattr(nm, "_last_g_raw", None) if nm is not None else None
    if g_raw is not None:
        flicker_t = float(g_raw[..., 1:].sub(g_raw[..., :-1]).abs().mean()) if g_raw.shape[-1] > 1 else nan
        rough_f = float(g_raw[:, :, 1:, :].sub(g_raw[:, :, :-1, :]).abs().mean()) if g_raw.shape[2] > 1 else nan
    else:
        flicker_t = rough_f = nan
    print(
        f"[{prefix}][GFLK]"
        f" flicker_t={flicker_t:.6f} rough_f={rough_f:.6f}"
    )

    # ------------------------------------------------------------------ GQ (gate quality)
    print(
        f"[{prefix}][GQ]"
        f" gate_mean={gate_mean:.4f}"
        f" gate_entF={gate_entF:.4f}"
        f" gate_maxF={gate_maxF:.4f}"
        f" gate_sumF={gate_sumF:.4f}"
    )

    # ------------------------------------------------------------------ WGATE (decomp diagnostics)
    g_raw_mean = decomp_stats.get("g_raw_mean", nan)
    g_eff_mean = decomp_stats.get("g_eff_mean", nan)
    n_energy = decomp_stats.get("n_energy", nan)
    print(
        f"[{prefix}][WGATE]"
        f" w_mean={w_mean:.4f} w_max={w_max:.4f}"
        f" g_raw_mean={g_raw_mean:.4f} g_eff_mean={g_eff_mean:.4f}"
        f" n_energy={n_energy:.6e}"
    )

    # ------------------------------------------------------------------ AUX
    aux_total = aux_stats.get("aux_total", nan)
    L_easy = aux_stats.get("L_easy", nan)
    L_white = aux_stats.get("L_white", nan)
    L_js = aux_stats.get("L_js", nan)
    L_w1 = aux_stats.get("L_w1", nan)
    L_min = aux_stats.get("L_min", nan)
    L_e_tv = aux_stats.get("L_e_tv", nan)
    L_sparse = aux_stats.get("L_sparse", nan)
    L_u_tv = aux_stats.get("L_u_tv", nan)
    print(
        f"[{prefix}][AUX]"
        f" aux_total={aux_total:.6e}"
        f" L_easy={L_easy:.6e} L_white={L_white:.6e}"
        f" L_js={L_js:.6e} L_w1={L_w1:.6e}"
        f" L_min={L_min:.6e} L_e_tv={L_e_tv:.6e}"
        f" L_sparse={L_sparse:.6e} L_u_tv={L_u_tv:.6e}"
    )

    # ------------------------------------------------------------------ NRATIO
    ratio_n_bc_mean = aux_stats.get("ratio_n_bc_mean", nan)
    ratio_n_bc_min  = aux_stats.get("ratio_n_bc_min",  nan)
    ratio_n_bc_max  = aux_stats.get("ratio_n_bc_max",  nan)
    loss_n_ratio_budget = aux_stats.get("loss_n_ratio_budget", nan)
    print(
        f"[{prefix}][NRATIO]"
        f" ratio_budget_mean={ratio_n_bc_mean:.4f}"
        f" ratio_budget_min={ratio_n_bc_min:.4f}"
        f" ratio_budget_max={ratio_n_bc_max:.4f}"
        f" loss_n_ratio_budget={loss_n_ratio_budget:.6e}"
    )

    # ------------------------------------------------------------------ NPRED
    pred_n_loss_dbg = aux_stats.get("pred_n_loss", nan)
    n_pred_fut = getattr(nm, "_last_n_pred_future", None) if nm is not None else None
    n_hist_time = getattr(nm, "_last_n_hist_time", None) if nm is not None else None
    n_true_ref  = getattr(nm, "_last_n_time", None) if nm is not None else None
    x_ref       = getattr(nm, "_last_x_time", None) if nm is not None else None
    ratio_n_pred = nan
    ratio_n_true_pred = nan
    if n_pred_fut is not None and x_ref is not None:
        x_e = float(x_ref.pow(2).mean().item()) + eps
        ratio_n_pred = float(n_pred_fut.detach().pow(2).mean().item()) / x_e
    if n_hist_time is not None and x_ref is not None:
        x_e = float(x_ref.pow(2).mean().item()) + eps
        ratio_n_true_pred = float(n_hist_time.detach().pow(2).mean().item()) / x_e
    print(
        f"[{prefix}][NPRED]"
        f" pred_n_loss={pred_n_loss_dbg:.6e}"
        f" ratio_n_pred={ratio_n_pred:.4f}"
        f" ratio_n_hist={ratio_n_true_pred:.4f}"
    )

    # ------------------------------------------------------------------ NPRED_TEACHER
    if nm is not None and cfg.future_mode == "pred":
        t_hist          = getattr(nm, "_last_teacher_T_hist", 0)
        t_full          = getattr(nm, "_last_teacher_T_full", 0)
        mask_hist_rate  = getattr(nm, "_last_teacher_mask_hist_rate", nan)
        mask_full_rate  = getattr(nm, "_last_teacher_mask_full_rate", nan)
        print(
            f"[{prefix}][NPRED_TEACHER]"
            f" T_hist={t_hist} T_full={t_full}"
            f" mask_hist_rate={mask_hist_rate:.4f}"
            f" mask_full_rate={mask_full_rate:.4f}"
        )

    # ------------------------------------------------------------------ SPIKE (SAN spike-robust stats)
    if nm is not None and hasattr(nm, "get_last_spike_stats"):
        spike_stats = nm.get_last_spike_stats()
        print(
            f"[{prefix}][SPIKE]"
            f" spike_rate={spike_stats['spike_rate']:.6f}"
            f" spike_thr_mean={spike_stats['spike_thr_mean']:.6e}"
            f" clip_frac={spike_stats['clip_frac']:.6f}"
            f" sigma_min_frac={spike_stats['sigma_min_frac']:.6f}"
        )

    # ------------------------------------------------------------------ ORACLE (oracle gate ablation)
    if nm is not None and hasattr(nm, "_last_oracle_rate"):
        oracle_rate      = getattr(nm, "_last_oracle_rate",      nan)
        oracle_trig_rate = getattr(nm, "_last_oracle_trig_rate", nan)
        gate_mode_str    = getattr(nm, "gate_mode",              "learned")
        print(
            f"[{prefix}][ORACLE]"
            f" gate_mode={gate_mode_str}"
            f" oracle_rate={oracle_rate:.6f}"
            f" trig_rate={oracle_trig_rate:.6f}"
        )

    # ------------------------------------------------------------------ TFBG (TF-background norm)
    if nm is not None and hasattr(nm, "get_last_stats"):
        tfbg_stats = nm.get_last_stats()
        print(
            f"[{prefix}][TFBG]"
            f" scale_min={tfbg_stats['scale_min']:.6f}"
            f" scale_max={tfbg_stats['scale_max']:.6f}"
            f" scale_mean={tfbg_stats['scale_mean']:.6f}"
            f" B_std={tfbg_stats['B_std']:.6f}"
        )

    # ------------------------------------------------------------------ GRAD
    gi = grad_info or {}
    print(
        f"[{prefix}][GRAD]"
        f" gate_grad_norm={gi.get('gate_grad_norm', nan):.6e}"
        f" gate_param_norm={gi.get('gate_param_norm', nan):.6f}"
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

    # Separate learning rate for gate (0 means use base_lr)
    gate_lr = float(cfg.gate_lr) if cfg.gate_lr > 0 else base_lr

    gate_params: list[torch.nn.Parameter] = []
    nm = getattr(model, "nm", None)
    if nm is not None:
        gate_mod = getattr(nm, "gate", None)
        if gate_mod is not None:
            gate_params = [p for p in gate_mod.parameters() if p.requires_grad]

    gate_ids = {id(p) for p in gate_params}
    remaining = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in gate_ids
    ]

    groups: list[dict[str, object]] = []
    if remaining:
        groups.append({"params": remaining, "weight_decay": cfg.weight_decay, "lr": base_lr})
    if gate_params:
        groups.append({"params": gate_params, "weight_decay": gate_wd, "lr": gate_lr})

    return Adam(groups, lr=base_lr)


def _get_gate_params(model: nn.Module) -> list:
    """Return gate params from model.nm."""
    nm = getattr(model, "nm", None)
    if nm is None:
        return []
    gate_mod = getattr(nm, "gate", None)
    if gate_mod is None:
        return []
    return [p for p in gate_mod.parameters() if p.requires_grad]


def train_one_epoch(model, loader, optimizer, cfg, scaler, epoch_idx: int):
    model.train()
    loss_fn = nn.MSELoss()
    losses = []
    task_losses = []
    aux_losses = []
    # Aux stats accumulators
    aux_stat_keys = ["aux_total", "L_easy", "L_white", "L_js", "L_w1", "L_min", "L_e_tv",
                     "ratio_n_bc_mean", "ratio_n_bc_min", "ratio_n_bc_max", "loss_n_ratio_budget",
                     "pred_n_loss", "L_sparse", "L_u_tv"]
    aux_stat_vals: dict[str, list[float]] = {k: [] for k in aux_stat_keys}

    has_nm_aux = (
        hasattr(model, "nm")
        and model.nm is not None
        and hasattr(model.nm, "get_last_aux_stats")
    )

    aux_scale = _aux_scale(cfg, epoch_idx)
    print(f"aux_loss_scale: {aux_scale:.6f}")

    gate_params = _get_gate_params(model)
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
            if hasattr(model.nm, "loss"):
                aux_loss = model.nm.loss(true)

            loss = task_loss + aux_loss * aux_scale

            # pred_N_loss: hard term, NOT scaled by aux_loss_scale
            if (
                cfg.future_mode == "pred"
                and cfg.n_pred_weight > 0.0
                and hasattr(model, "nm")
                and model.nm is not None
                and hasattr(model.nm, "get_last_n_pred_future")
            ):
                n_pred_future = model.nm.get_last_n_pred_future()
                if n_pred_future is not None:
                    n_true_future = model.nm.teacher_n_future(batch_x, batch_y)
                    pred_n_loss_t = ((n_pred_future - n_true_future) ** 2).mean()
                    loss = loss + cfg.n_pred_weight * pred_n_loss_t
                    model.nm._last_pred_n_loss = float(pred_n_loss_t.detach().item())

            # First-batch gradient collection
            if not first_batch_done:
                gate_snap_norms = [p.data.norm().item() for p in gate_params]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            if not first_batch_done:
                gate_grad_norm = _grad_norm_or_nan(gate_params)
                gate_param_norm = _norm_or_nan(gate_params)

            optimizer.step()

            if not first_batch_done:
                grad_info = {
                    "gate_grad_norm": gate_grad_norm,
                    "gate_param_norm": gate_param_norm,
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
            if has_nm_aux:
                s = model.nm.get_last_aux_stats()
                for k in aux_stat_keys:
                    aux_stat_vals[k].append(float(s.get(k, 0.0)))

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

    train_stat: dict[str, float] = {
        "task_loss": float(np.mean(task_losses)) if task_losses else 0.0,
        "aux_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
    }
    if has_nm_aux and aux_stat_vals["aux_total"]:
        for k in aux_stat_keys:
            train_stat[k] = float(np.mean(aux_stat_vals[k]))

    return float(np.mean(losses)) if losses else 0.0, gate_stats_str, train_stat


@torch.no_grad()
def evaluate(model, loader, cfg, scaler, debug_prefix: str | None = None):
    model.eval()
    metrics = _build_metrics(torch.device(cfg.device))
    for metric in metrics.values():
        metric.reset()

    has_nm_aux = (
        hasattr(model, "nm")
        and model.nm is not None
        and hasattr(model.nm, "loss")
        and hasattr(model.nm, "get_last_aux_stats")
    )
    aux_stat_keys = ["aux_total", "L_easy", "L_white", "L_js", "L_w1", "L_min", "L_e_tv",
                     "ratio_n_bc_mean", "ratio_n_bc_min", "ratio_n_bc_max", "loss_n_ratio_budget",
                     "pred_n_loss", "L_sparse", "L_u_tv"]
    aux_stat_vals: dict[str, list[float]] = {k: [] for k in aux_stat_keys}

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

        if has_nm_aux:
            model.nm.loss()
            s = model.nm.get_last_aux_stats()
            for k in aux_stat_keys:
                aux_stat_vals[k].append(float(s.get(k, 0.0)))

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
    if has_nm_aux and aux_stat_vals["aux_total"]:
        for k in aux_stat_keys:
            results[k] = float(np.mean(aux_stat_vals[k]))
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

    # Auto-calibrate trigger mask thresholds from training data
    # Only run when both delta_E_mask and delta_P_mask are unset (== 0)
    if cfg.auto_thresholds and cfg.delta_E_mask == 0.0 and cfg.delta_P_mask == 0.0:
        nm = getattr(model, "nm", None)
        if nm is not None and hasattr(nm, "delta_E_mask"):
            delta_E_mask, delta_P_mask = calibrate_thresholds(
                model, dataloader.train_loader, cfg, max_batches=200
            )
            cfg.delta_E_mask = delta_E_mask
            cfg.delta_P_mask = delta_P_mask
            nm.delta_E_mask = delta_E_mask
            nm.delta_P_mask = delta_P_mask
            print(
                f"[Calibration] trigger_q={cfg.trigger_q}"
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
    if cfg.gate_arch == "lowrank_sparse" and cfg.gate_sparse_l1_weight <= 0:
        raise ValueError(
            "gate_arch=lowrank_sparse requires --gate-sparse-l1-weight > 0 "
            "to keep sparse residual constrained. Set e.g. 1e-3."
        )
    if cfg.gate_lowrank_time_ks % 2 == 0 or cfg.gate_lowrank_freq_ks % 2 == 0:
        raise ValueError(
            f"gate_lowrank_time_ks and gate_lowrank_freq_ks must be odd for symmetric padding. "
            f"Got time_ks={cfg.gate_lowrank_time_ks}, freq_ks={cfg.gate_lowrank_freq_ks}."
        )
    if cfg.gate_arch in ("lowrank", "lowrank_sparse"):
        print(
            f"[LowRankGate] gate_arch={cfg.gate_arch}"
            f" rank={cfg.gate_lowrank_rank}"
            f" time_ks={cfg.gate_lowrank_time_ks}"
            f" freq_ks={cfg.gate_lowrank_freq_ks}"
            f" use_bias={cfg.gate_lowrank_use_bias}"
            f" sparse_l1_weight={cfg.gate_sparse_l1_weight}"
            f" u_tv_weight={cfg.gate_lowrank_u_tv_weight}"
        )

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

        # Build aux info strings
        train_stat_str = ""
        if train_stat:
            train_stat_str = f" | task={train_stat.get('task_loss', 0.0):.6f}"
            if "aux_total" in train_stat:
                train_stat_str += (
                    f" aux_total={train_stat['aux_total']:.6e}"
                    f" (L_easy={train_stat.get('L_easy', 0.0):.3e}"
                    f" L_white={train_stat.get('L_white', 0.0):.3e}"
                    f" L_js={train_stat.get('L_js', 0.0):.3e}"
                    f" L_w1={train_stat.get('L_w1', 0.0):.3e}"
                    f" L_min={train_stat.get('L_min', 0.0):.3e}"
                    f" L_e_tv={train_stat.get('L_e_tv', 0.0):.3e}"
                    f" L_sparse={train_stat.get('L_sparse', 0.0):.3e}"
                    f" L_u_tv={train_stat.get('L_u_tv', 0.0):.3e})"
                )
        val_stat_str = ""
        if "aux_total" in val_metrics:
            val_stat_str = f" | aux_total={val_metrics['aux_total']:.6e}"

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
