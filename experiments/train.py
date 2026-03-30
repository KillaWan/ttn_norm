from __future__ import annotations

import argparse
import json
import os
import sys
import dataclasses
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
from ttn_norm.normalizations import DishTS, FAN, LFAN, No, RevIN, SAN, SANMS, TFBackgroundNorm, WaveBandNormB
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

    # Enlarge loader window for wavband context patches
    if (
        cfg.norm_type.lower() in {"wavband", "wavband_b"}
        and cfg.wav_ctx_patches > 0
    ):
        loader_window = cfg.window + cfg.wav_ctx_patches * cfg.wav_patch_len
    else:
        loader_window = cfg.window

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
            window=loader_window,
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
            window=loader_window,
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
    fan_ablation_mode: str = "original"   # "original" | "low_only" | "topk_exclude_low"
    # DishTS
    dish_init: str = "uniform"   # "standard" | "avg" | "uniform"
    # SAN (seasonal adaptive normalization)
    san_period_len: int = 12
    san_stride: int = 0
    san_base_stride: int = 0
    san_force_extra_levels: int = -1
    # SANMS (multi-scale SAN)
    san_ms_scales: str = "1,2,4"
    san_ms_tau: float = 1.0
    san_ms_sigma_min: float = 1e-3
    san_ms_lambda_std: float = 1.0
    san_ms_ent_weight: float = 0.0
    # SAN TTN (overlap-consistency calibrator, eval-only) options
    # OSTN: overlap-to-statistics test-time normalization
    ostn_enabled:                       bool  = False
    ostn_hidden_dim:                    int   = 64
    ostn_num_layers:                    int   = 2
    ostn_dropout:                       float = 0.0
    ostn_pos_dim:                       int   = 16
    ostn_use_patchwise_overlap_summary: bool  = True
    ostn_alpha_l1:                      float = 1e-3
    ostn_overlap_weight:                float = 1.0
    ostn_stats_weight:                  float = 1.0
    ostn_logsigma_min:                  float = -6.0
    ostn_logsigma_max:                  float = 6.0
    ostn_reset_each_eval:               bool  = True

    # OSTN stage2 finetuning settings
    ostn_stage2_train:                  bool  = False
    ostn_stage2_epochs:                 int   = 20
    ostn_stage2_lr:                     float = 1e-3
    ostn_stage2_weight_decay:           float = 0.0
    ostn_stage2_patience:               int   = 5
    ostn_stage2_min_epochs:             int   = 5
    ostn_stage2_metric:                 str   = "val_mse"

    # Oracle stats injection modes (eval only, for upper-bound analysis)
    # san_oracle_norm:      true mean + true std
    # san_oracle_mean_only: true mean + predicted std
    # san_oracle_std_only:  predicted mean + true std
    san_oracle_norm:      bool = False
    san_oracle_mean_only: bool = False
    san_oracle_std_only:  bool = False
    # Old compat params silently ignored
    ttn_calib_batches: int = 200
    ttn_backcast_patches: int = 3
    ttn_pad_mode: str = "replicate"
    ttn_mean_loss_weight: float = 0.1
    ttn_stage2_only: bool = False
    ttn_debug_print: bool = False
    ttn_use_direct_head: bool = True
    ttn_use_delta2: bool = True
    ttn_gate_hidden_dim: int = 64
    ttn_stats_loss_weight: float = 0.5
    ttn_base_stats_loss_weight: float = 0.25
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
    # WaveBandNormB (norm_type="wavband_b" or "wavband")
    wav_wavelet: str = "haar"        # "haar" | "db2"
    wav_levels: int = 3              # SWT decomposition levels
    wav_mid_levels: str = "2,3"      # comma-separated 1-indexed level indices
    wav_high_levels: str = "1"       # comma-separated 1-indexed level indices
    wav_patch_len: int = 24          # patch length (window & pred_len must be divisible)
    wav_cond: str = "rho_h"          # "none" | "rho_h" | "rho_all"
    wav_sigma_min: float = 1e-3
    wav_stats_loss_weight: float = 0.1
    wav_oracle_mode: str = "none"    # "none" | "eval_stats" | "train_eval_stats" | "train_oracle"
    wav_pred_use_wave: bool = False  # if True, append normalized low/mid waveforms to predictor input
    wav_ctx_patches: int = 0         # extra history patches fed to predictor (loader window enlarged)
    wav_hard_aux_scale: float = 0.0  # >0: wavband aux loss uses this fixed scale instead of aux_scale
    wav_stage_train: bool = False    # if True, stage1 trains only predictor; stage2 trains jointly
    wav_pretrain_epochs: int = 10    # number of stage1 epochs (predictor-only pretraining)
    wav_pretrain_aux_scale: float = 1.0  # aux loss scale in stage1
    wav_pred_lr: float = 1e-3        # predictor-specific learning rate (separate optimizer group)
    wav_pred_weight_decay: float = 0.0   # predictor-specific weight decay
    wav_pad_mode: str = "reflect"    # SWT boundary padding: "reflect" | "replicate" | "circular"
    wav_use_soft_split: bool = False  # if True, use learnable soft band split with band_logits
    wav_gate_tau: float = 1.0         # soft-split temperature (smaller → harder split)
    wav_split_reg_weight: float = 1e-2  # monotone regularisation weight for band-split gates
    wav_pretrain_use_early_stop: bool = True   # enable early stop during stage1 pretraining
    wav_pretrain_patience: int = 8             # stage1 early stop patience (epochs)
    wav_pretrain_min_epochs: int = 10          # minimum stage1 epochs before early stop kicks in
    wav_pretrain_metric: str = "val_aux_total" # metric to monitor for stage1 early stop
    wav_pretrain_delta: float = 0.0            # minimum improvement to reset patience counter
    wav_oracle_prob_start: float = 1.0         # starting oracle probability in stage2 (epoch 0 of stage2)
    wav_oracle_prob_end: float = 0.0           # ending oracle probability in stage2
    wav_oracle_anneal_epochs: int = 50         # number of stage2 epochs over which oracle prob anneals
    wav_freeze_pred_in_stage2: bool = True     # freeze predictor+band_logits during stage2 joint training
    wav_pretrain_stop_to_stage2: bool = True   # if True, stage1 early-stop immediately advances to stage2
    # LFAN (norm_type="lfan")
    lfan_k_low:          int   = 8
    lfan_burst_smooth:   int   = 5
    lfan_burst_mix:      float = 0.7
    lfan_gate_segments:  int   = 16
    lfan_remove_quiet:   float = 0.98
    lfan_remove_burst:   float = 0.05
    lfan_gate_gamma:     float = 3.0
    lfan_gate_tau:       float = 1.0
    lfan_remove_floor:   float = 0.0
    lfan_sigma_min:      float = 1e-5
    lfan_fan_equiv:      bool  = False
    lfan_hidden_dim:     int   = 64
    lfan_loss_low_coeff: float = 1.0
    lfan_loss_low_shape: float = 0.5
    lfan_loss_mu:        float = 0.2
    lfan_loss_sigma:     float = 0.2
    lfan_loss_res:       float = 0.2


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
            ablation_mode=cfg.fan_ablation_mode,
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
            stride=cfg.san_stride,
            base_stride=cfg.san_base_stride,
            force_extra_levels=cfg.san_force_extra_levels,
            enc_in=num_features,
            ostn_enabled=cfg.ostn_enabled,
            ostn_hidden_dim=cfg.ostn_hidden_dim,
            ostn_num_layers=cfg.ostn_num_layers,
            ostn_dropout=cfg.ostn_dropout,
            ostn_pos_dim=cfg.ostn_pos_dim,
            ostn_use_patchwise_overlap_summary=cfg.ostn_use_patchwise_overlap_summary,
            ostn_alpha_l1=cfg.ostn_alpha_l1,
            ostn_overlap_weight=cfg.ostn_overlap_weight,
            ostn_stats_weight=cfg.ostn_stats_weight,
            ostn_logsigma_min=cfg.ostn_logsigma_min,
            ostn_logsigma_max=cfg.ostn_logsigma_max,
            ostn_reset_each_eval=cfg.ostn_reset_each_eval,
        )

    elif _nt in {"sanms", "san_ms"}:
        _scales = tuple(int(x) for x in cfg.san_ms_scales.split(",") if x.strip())
        norm_model = SANMS(
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            period_len=cfg.san_period_len,
            enc_in=num_features,
            scales=_scales,
            tau=cfg.san_ms_tau,
            sigma_min=cfg.san_ms_sigma_min,
            lambda_std=cfg.san_ms_lambda_std,
            ent_weight=cfg.san_ms_ent_weight,
        )

    elif _nt in {"wavband_b", "wavband"}:
        _wav_mid  = tuple(int(x) for x in cfg.wav_mid_levels.split(",")  if x.strip())
        _wav_high = tuple(int(x) for x in cfg.wav_high_levels.split(",") if x.strip())
        norm_model = WaveBandNormB(
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            enc_in=num_features,
            wavelet=cfg.wav_wavelet,
            levels=cfg.wav_levels,
            mid_levels=_wav_mid,
            high_levels=_wav_high,
            patch_len=cfg.wav_patch_len,
            cond=cfg.wav_cond,
            sigma_min=cfg.wav_sigma_min,
            stats_loss_weight=cfg.wav_stats_loss_weight,
            pred_use_wave=cfg.wav_pred_use_wave,
            ctx_patches=cfg.wav_ctx_patches,
            pad_mode=cfg.wav_pad_mode,
            wav_use_soft_split=cfg.wav_use_soft_split,
            wav_gate_tau=cfg.wav_gate_tau,
            wav_split_reg_weight=cfg.wav_split_reg_weight,
        )
        # Adjust backbone enc_in to include conditioning channels
        num_features += norm_model.cond_channels

    elif _nt == "tf_bg":
        norm_model = TFBackgroundNorm(
            n_fft=cfg.tfbg_n_fft,
            hop_length=cfg.tfbg_hop,
            time_kernel=cfg.tfbg_time_kernel,
            freq_kernel=cfg.tfbg_freq_kernel,
            bmax=cfg.tfbg_bmax,
        )

    elif _nt == "lfan":
        norm_model = LFAN(
            seq_len=cfg.window,
            pred_len=cfg.pred_len,
            enc_in=num_features,
            k_low=cfg.lfan_k_low,
            burst_smooth=cfg.lfan_burst_smooth,
            burst_mix=cfg.lfan_burst_mix,
            gate_segments=cfg.lfan_gate_segments,
            remove_quiet=cfg.lfan_remove_quiet,
            remove_burst=cfg.lfan_remove_burst,
            gate_gamma=cfg.lfan_gate_gamma,
            gate_tau=cfg.lfan_gate_tau,
            remove_floor=cfg.lfan_remove_floor,
            sigma_min=cfg.lfan_sigma_min,
            fan_equiv=cfg.lfan_fan_equiv,
            hidden_dim=cfg.lfan_hidden_dim,
            loss_low_coeff=cfg.lfan_loss_low_coeff,
            loss_low_shape=cfg.lfan_loss_low_shape,
            loss_mu=cfg.lfan_loss_mu,
            loss_sigma=cfg.lfan_loss_sigma,
            loss_res=cfg.lfan_loss_res,
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


@torch.no_grad()
def _maybe_update_ostn_cache(
    model: nn.Module,
    pred: torch.Tensor,
    cfg: TrainConfig,
) -> None:
    """Update OSTN stream cache from the current batch's final prediction.

    Must be called AFTER model.forward() returns during eval.
    pred: (B, pred_len, C) — denormalized prediction.
    """
    nm = getattr(model, "nm", None)
    if nm is None:
        return
    if not getattr(nm, "ostn_enabled", False):
        return
    if model.training:
        return
    if not hasattr(nm, "update_ostn_stream_cache"):
        return
    nm.update_ostn_stream_cache(pred)



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

    # For wavband_b, skip LocalTF-specific diagnostic blocks (they print NaN)
    _is_wavband_dbg = cfg.norm_type.lower() in {"wavband", "wavband_b"}
    if _is_wavband_dbg:
        # Only print wavband-relevant blocks; skip LocalTF blocks
        if nm is not None and hasattr(nm, "get_last_diag"):
            d = nm.get_last_diag()
            print(
                f"[{prefix}][WAVBAND]"
                f" sigL={d.get('sig_L_mean', nan):.4f}"
                f" sigDj={d.get('sig_Dj_mean', nan):.4f}"
                f" rhoH_mean={d.get('rho_H_mean', nan):.4f}"
                f" rhoH_p10={d.get('rho_H_p10', nan):.4f}"
                f" rhoH_p90={d.get('rho_H_p90', nan):.4f}"
                f" L_stats={d.get('L_stats', nan):.6f}"
                f" L_split={d.get('L_split', nan):.6f}"
                f" oracle_recon_mse={d.get('oracle_recon_mse', nan):.2e}"
            )
        if nm is not None and hasattr(nm, "get_split_stats"):
            ss = nm.get_split_stats()
            print(
                f"[{prefix}][WAVBAND_GATES]"
                f" g_mean={ss['g_mean']:.4f}"
                f" g_min={ss['g_min']:.4f}"
                f" g_max={ss['g_max']:.4f}"
                f" split_reg={ss['split_reg']:.6f}"
            )
            if prefix == "TRAIN_DBG":
                g_vec_str = " ".join(f"{g:.4f}" for g in ss["g_vec"])
                print(f"[{prefix}][WAVBAND_GATES] g_vec=[{g_vec_str}]")
        return

    # Fetch summary dicts once (non-wavband path)
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

    # ------------------------------------------------------------------ OSTN diagnostics
    if nm is not None and getattr(nm, "ostn_enabled", False) and hasattr(nm, "get_last_ostn_stats"):
        ostn_st = nm.get_last_ostn_stats()
        print(
            f"[{prefix}][OSTN]"
            f" enabled={ostn_st['enabled']}"
            f" applied={ostn_st['applied']}"
            f" alpha_mean={ostn_st['alpha_mean']:.4f}"
            f" alpha_max={ostn_st['alpha_max']:.4f}"
            f" delta_mu_abs={ostn_st['delta_mu_abs_mean']:.4f}"
            f" delta_lsig_abs={ostn_st['delta_logsigma_abs_mean']:.4f}"
            f" base_olap={ostn_st['base_overlap_loss']:.6f}"
            f" refined_olap={ostn_st['refined_overlap_loss']:.6f}"
            f" base_mu={ostn_st['base_mu_abs_mean']:.4f}"
            f" refined_mu={ostn_st['refined_mu_abs_mean']:.4f}"
            f" base_sigma={ostn_st['base_sigma_mean']:.4f}"
            f" refined_sigma={ostn_st['refined_sigma_mean']:.4f}"
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

    # ------------------------------------------------------------------ WAVBAND (non-wavband path: skip)

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

    # Predictor-specific parameter group (wavband staged training)
    pred_params = _wav_pred_params(model)

    excluded_ids = {id(p) for p in gate_params} | {id(p) for p in pred_params}
    remaining = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in excluded_ids
    ]

    groups: list[dict[str, object]] = []
    if remaining:
        groups.append({"params": remaining, "weight_decay": cfg.weight_decay, "lr": base_lr})
    if gate_params:
        groups.append({"params": gate_params, "weight_decay": gate_wd, "lr": gate_lr})
    if pred_params:
        groups.append({
            "params": pred_params,
            "lr": float(cfg.wav_pred_lr),
            "weight_decay": float(cfg.wav_pred_weight_decay),
        })

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


def _maybe_set_oracle_future(
    model: nn.Module, cfg: TrainConfig, y_future: torch.Tensor
) -> None:
    """If wav_oracle_mode requires it, pass true future stats to WaveBandNormB.

    y_future: (B, pred_len, C) ground-truth future tensor (unscaled original space).
    """
    nm = getattr(model, "nm", None)
    if nm is not None and hasattr(nm, "set_oracle_future"):
        nm.set_oracle_future(y_future, cfg.wav_oracle_mode)


def _maybe_clear_oracle_future(model: nn.Module, cfg: TrainConfig) -> None:
    """Remove oracle future tensor from WaveBandNormB after inference."""
    nm = getattr(model, "nm", None)
    if nm is not None and hasattr(nm, "clear_oracle_future"):
        nm.clear_oracle_future()


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Enable or disable gradient computation for all parameters of a module."""
    for p in module.parameters():
        p.requires_grad = flag


def _wav_pred_params(model: nn.Module) -> list:
    """Return predictor + band_logits parameters from model.nm that require grad."""
    nm = getattr(model, "nm", None)
    if nm is None:
        return []
    params: list = []
    predictor = getattr(nm, "predictor", None)
    if predictor is not None:
        params.extend(p for p in predictor.parameters() if p.requires_grad)
    # Include learnable band split logits if trainable
    band_logits = getattr(nm, "band_logits", None)
    if band_logits is not None and isinstance(band_logits, torch.nn.Parameter) and band_logits.requires_grad:
        params.append(band_logits)
    return params


def _ostn_params(model: nn.Module) -> list:
    """Return OSTN corrector parameters only."""
    nm = getattr(model, "nm", None)
    if nm is None or not hasattr(nm, "ostn_corrector"):
        return []
    return [p for p in model.nm.ostn_corrector.parameters() if p.requires_grad]


def _freeze_all_except_wav_predictor(model: nn.Module) -> None:
    """Freeze the entire model, then unfreeze model.nm.predictor and band_logits."""
    _set_requires_grad(model, False)
    nm = getattr(model, "nm", None)
    if nm is not None:
        predictor = getattr(nm, "predictor", None)
        if predictor is not None:
            _set_requires_grad(predictor, True)
        band_logits = getattr(nm, "band_logits", None)
        if band_logits is not None and isinstance(band_logits, torch.nn.Parameter):
            band_logits.requires_grad = True


def _freeze_all_except_ostn_corrector(model: nn.Module) -> None:
    """Freeze entire model, then unfreeze only OSTN corrector."""
    _set_requires_grad(model, False)
    nm = getattr(model, "nm", None)
    if nm is not None and hasattr(nm, "ostn_corrector"):
        _set_requires_grad(nm.ostn_corrector, True)


@torch.no_grad()
def _wavband_val_aux(model: nn.Module, loader, cfg: TrainConfig) -> float:
    """Compute mean wavband predictor aux loss on *loader* (no task loss).

    Used for stage1 early-stop monitoring.  Returns mean aux loss; 0.0 if
    model.nm has no ``loss`` method.
    """
    nm = getattr(model, "nm", None)
    if nm is None or not hasattr(nm, "loss"):
        return 0.0
    model.eval()
    ctx_len = cfg.wav_ctx_patches * cfg.wav_patch_len
    vals: list[float] = []
    for batch_data in loader:
        bx = batch_data[0].to(cfg.device).float()
        by = batch_data[1].to(cfg.device).float()
        if ctx_len > 0 and hasattr(nm, "set_ctx_history"):
            nm.set_ctx_history(bx[:, :ctx_len, :])
            nm.normalize(bx[:, ctx_len:, :])
            nm.clear_ctx_history()
        else:
            nm.normalize(bx)
        vals.append(float(nm.loss(by).item()))
    return float(np.mean(vals)) if vals else 0.0


def _oracle_prob(cfg: TrainConfig, stage2_epoch: int) -> float:
    """Linearly anneal oracle probability from wav_oracle_prob_start to wav_oracle_prob_end."""
    anneal = max(int(cfg.wav_oracle_anneal_epochs), 1)
    t = min(stage2_epoch, anneal) / float(anneal)
    return float(cfg.wav_oracle_prob_start + (cfg.wav_oracle_prob_end - cfg.wav_oracle_prob_start) * t)


def train_one_epoch(model, loader, optimizer, cfg, scaler, epoch_idx: int,
                    oracle_prob: float = 1.0, in_stage1: bool = False,
                    freeze_wav_pred: bool = False):
    model.train()
    loss_fn = nn.MSELoss()
    losses = []
    task_losses = []
    aux_losses = []
    aux_to_task_ratios = []
    # Aux stats accumulators
    aux_stat_keys = ["aux_total", "L_easy", "L_white", "L_js", "L_w1", "L_min", "L_e_tv",
                     "ratio_n_bc_mean", "ratio_n_bc_min", "ratio_n_bc_max", "loss_n_ratio_budget",
                     "pred_n_loss", "L_sparse", "L_u_tv"]
    # LFAN debug scalar accumulators
    _nm_lfan_tr = getattr(model, "nm", None)
    _has_lfan_dbg = _nm_lfan_tr is not None and hasattr(_nm_lfan_tr, "get_debug_scalars")
    _lfan_dbg_accum: dict[str, list[float]] = {}
    aux_stat_vals: dict[str, list[float]] = {k: [] for k in aux_stat_keys}

    has_nm_aux = (
        hasattr(model, "nm")
        and model.nm is not None
        and hasattr(model.nm, "get_last_aux_stats")
    )

    # ── Stage freeze / unfreeze ───────────────────────────────────────────────
    _in_stage1 = bool(in_stage1)
    if _in_stage1:
        _freeze_all_except_wav_predictor(model)
        print(f"[stage1] epoch {epoch_idx}: predictor-only training (frozen backbone)")
    else:
        _set_requires_grad(model, True)
        # In stage2: re-freeze only band_logits (predictor stays trainable)
        if freeze_wav_pred:
            _nm_fp = getattr(model, "nm", None)
            if _nm_fp is not None:
                _bl_fp = getattr(_nm_fp, "band_logits", None)
                if _bl_fp is not None and isinstance(_bl_fp, torch.nn.Parameter):
                    _bl_fp.requires_grad = False

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

            # ctx split for wavband extended predictor history
            ctx_len = cfg.wav_ctx_patches * cfg.wav_patch_len
            _is_wavband = cfg.norm_type.lower() in {"wavband", "wavband_b"}
            if ctx_len > 0 and _is_wavband:
                x_ctx  = batch_x[:, :ctx_len, :]
                x_main = batch_x[:, ctx_len:, :]
                x_enc_main = batch_x_enc[:, ctx_len:, :]
                if model.nm is not None and hasattr(model.nm, "set_ctx_history"):
                    model.nm.set_ctx_history(x_ctx)
                if model.is_former:
                    label_len = min(cfg.label_len or (cfg.window // 2), x_main.size(1))
                    dec_inp, dec_inp_enc = _make_dec_inputs(
                        x_main, batch_y_enc, x_enc_main, label_len
                    )
            else:
                x_main, x_enc_main = batch_x, batch_x_enc

            # ── Stage 1: predictor-only pretraining ──────────────────────────
            if _in_stage1:
                if model.nm is not None and hasattr(model.nm, "normalize"):
                    model.nm.normalize(x_main)
                aux_loss = torch.tensor(0.0, device=batch_x.device)
                if model.nm is not None and hasattr(model.nm, "loss"):
                    aux_loss = model.nm.loss(batch_y)
                loss = cfg.wav_pretrain_aux_scale * aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                if ctx_len > 0 and _is_wavband:
                    if model.nm is not None and hasattr(model.nm, "clear_ctx_history"):
                        model.nm.clear_ctx_history()
                task_loss = torch.tensor(0.0, device=batch_x.device)
                losses.append(loss.item())
                task_losses.append(task_loss.item())
                aux_losses.append(aux_loss.item())
                aux_to_task_ratios.append(0.0)
                pbar.update(batch_x.size(0))
                pbar.set_postfix(
                    stage="pretrain",
                    aux=f"{aux_loss.item():.4f}",
                    total=f"{loss.item():.4f}",
                )
                continue

            # ── Stage 2 / joint: full model forward ──────────────────────────
            # UB-2 / train_oracle: oracle future stats used during training forward pass
            # oracle_prob < 1.0 anneals towards predictor-only denorm
            _use_oracle = (
                cfg.wav_oracle_mode in {"train_eval_stats", "train_oracle"}
                and torch.rand(1).item() < oracle_prob
            )
            if _use_oracle:
                _maybe_set_oracle_future(model, cfg, batch_y)
            pred = model(x_main, x_enc_main, dec_inp, dec_inp_enc)
            if _use_oracle:
                _maybe_clear_oracle_future(model, cfg)
            if ctx_len > 0 and _is_wavband:
                if model.nm is not None and hasattr(model.nm, "clear_ctx_history"):
                    model.nm.clear_ctx_history()
            true = batch_y
            if cfg.invtrans_loss:
                pred = scaler.inverse_transform(pred)
                true = origin_y

            task_loss = loss_fn(pred, true)
            aux_loss = torch.tensor(0.0, device=task_loss.device)
            if hasattr(model.nm, "loss"):
                aux_loss = model.nm.loss(true)

            if _is_wavband and cfg.wav_hard_aux_scale > 0.0:
                loss = task_loss + cfg.wav_hard_aux_scale * aux_loss
            else:
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
            aux_to_task_ratios.append(
                float((aux_loss.detach().item() * aux_scale) / (task_loss.detach().item() + 1e-8))
            )
            if has_nm_aux:
                s = model.nm.get_last_aux_stats()
                for k in aux_stat_keys:
                    aux_stat_vals[k].append(float(s.get(k, 0.0)))
            if _has_lfan_dbg:
                _ds = _nm_lfan_tr.get_debug_scalars()
                for k, v in _ds.items():
                    _lfan_dbg_accum.setdefault(k, []).append(float(v))

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
        "aux_to_task_ratio": float(np.mean(aux_to_task_ratios)) if aux_to_task_ratios else 0.0,
    }
    if has_nm_aux and aux_stat_vals["aux_total"]:
        for k in aux_stat_keys:
            train_stat[k] = float(np.mean(aux_stat_vals[k]))
    if _has_lfan_dbg and _lfan_dbg_accum:
        for k, vs in _lfan_dbg_accum.items():
            train_stat[k] = float(np.mean(vs))

    return float(np.mean(losses)) if losses else 0.0, gate_stats_str, train_stat


def train_ostn_stage2_one_epoch(model, loader, optimizer, cfg, scaler, epoch_idx: int):
    """Stage2 OSTN corrector fine-tuning loop.

    - model.eval(), but OSTN corrector train mode
    - freeze all params except model.nm.ostn_corrector
    - use adjacent within-batch pairs (i and i+1)
    - skip batches with batch_size <= 1
    - average B-1 pair losses per batch
    """
    model.eval()
    nm = getattr(model, "nm", None)
    if nm is None or not hasattr(nm, "ostn_train_loss") or not hasattr(nm, "ostn_corrector"):
        raise RuntimeError("model.nm must have ostn_train_loss and ostn_corrector for OSTN stage2")

    if not _ostn_params(model):
        raise RuntimeError("No OSTN corrector parameters found for stage2 training")

    nm.ostn_corrector.train()
    _freeze_all_except_ostn_corrector(model)

    total_loss = 0.0
    batch_count = 0

    for batch_x, batch_y, origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()

        B = batch_x.size(0)
        if B <= 1:
            continue

        loss_accum = 0.0
        pair_cnt = 0

        for i in range(B - 1):
            x_t = batch_x[i : i + 1]
            y_t = batch_y[i : i + 1]
            x_t1 = batch_x[i + 1 : i + 2]
            y_t1 = batch_y[i + 1 : i + 2]

            loss_pair = nm.ostn_train_loss(x_t, y_t, x_t1, y_t1, prev_overlap_summary=None)
            loss_accum = loss_accum + loss_pair
            pair_cnt += 1

        if pair_cnt == 0:
            continue

        batch_loss = loss_accum / float(pair_cnt)

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        total_loss += float(batch_loss.item())
        batch_count += 1

    return float(total_loss / batch_count) if batch_count > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, cfg, scaler, debug_prefix: str | None = None):
    model.eval()
    # Reset TTN eval state (prev-batch overlap cache) at the start of each phase
    _nm_ostn_ev = getattr(model, "nm", None)
    if _nm_ostn_ev is not None and hasattr(_nm_ostn_ev, "reset_ostn_eval_state"):
        _nm_ostn_ev.reset_ostn_eval_state()
    # Reset FAN frequency-bin stats at the start of each evaluation phase
    _nm_fan = getattr(model, "nm", None)
    _is_fan = _nm_fan is not None and hasattr(_nm_fan, "reset_freq_stats")
    if _is_fan:
        _nm_fan.reset_freq_stats()
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

    # LFAN debug scalar accumulators
    _nm_lfan_ev = getattr(model, "nm", None)
    _has_lfan_ev = _nm_lfan_ev is not None and hasattr(_nm_lfan_ev, "get_debug_scalars")
    _lfan_ev_accum: dict[str, list[float]] = {}
    _lfan_last_text: str = ""

    first_batch_done = False
    for batch_x, batch_y, origin_y, batch_x_enc, batch_y_enc in loader:
        batch_x = batch_x.to(cfg.device).float()
        batch_y = batch_y.to(cfg.device).float()
        batch_x_enc = batch_x_enc.to(cfg.device).float()
        batch_y_enc = batch_y_enc.to(cfg.device).float()
        origin_y = origin_y.to(cfg.device).float()

        dec_inp, dec_inp_enc = None, None

        # ctx split for wavband extended predictor history
        ctx_len = cfg.wav_ctx_patches * cfg.wav_patch_len
        _is_wavband = cfg.norm_type.lower() in {"wavband", "wavband_b"}
        if ctx_len > 0 and _is_wavband:
            x_ctx  = batch_x[:, :ctx_len, :]
            x_main = batch_x[:, ctx_len:, :]
            x_enc_main = batch_x_enc[:, ctx_len:, :]
            if model.nm is not None and hasattr(model.nm, "set_ctx_history"):
                model.nm.set_ctx_history(x_ctx)
            if model.is_former:
                label_len = min(cfg.label_len or (cfg.window // 2), x_main.size(1))
                dec_inp, dec_inp_enc = _make_dec_inputs(
                    x_main, batch_y_enc, x_enc_main, label_len
                )
        else:
            x_main, x_enc_main = batch_x, batch_x_enc
            if model.is_former:
                label_len = min(cfg.label_len or (cfg.window // 2), batch_x.size(1))
                dec_inp, dec_inp_enc = _make_dec_inputs(
                    batch_x, batch_y_enc, batch_x_enc, label_len
                )

        # Oracle future stats used during evaluation (only explicit eval_stats mode)
        if cfg.wav_oracle_mode == "eval_stats":
            _maybe_set_oracle_future(model, cfg, batch_y)
        # SAN oracle norm: inject true future patch stats (upper-bound test)
        _nm_oracle = getattr(model, "nm", None)
        _oracle_mode = (
            "both"      if cfg.san_oracle_norm      else
            "mean_only" if cfg.san_oracle_mean_only else
            "std_only"  if cfg.san_oracle_std_only  else
            None
        )
        if _oracle_mode is not None and _nm_oracle is not None and hasattr(_nm_oracle, "set_oracle_stats"):
            _nm_oracle.set_oracle_stats(batch_y, mode=_oracle_mode)
        pred = model(x_main, x_enc_main, dec_inp, dec_inp_enc)
        if cfg.wav_oracle_mode == "eval_stats":
            _maybe_clear_oracle_future(model, cfg)
        if _oracle_mode is not None and _nm_oracle is not None and hasattr(_nm_oracle, "clear_oracle_stats"):
            _nm_oracle.clear_oracle_stats()
        # OSTN: update stream cache from current final prediction
        _maybe_update_ostn_cache(model, pred.detach(), cfg)
        if ctx_len > 0 and _is_wavband:
            if model.nm is not None and hasattr(model.nm, "clear_ctx_history"):
                model.nm.clear_ctx_history()

        if has_nm_aux:
            model.nm.loss()
            s = model.nm.get_last_aux_stats()
            for k in aux_stat_keys:
                aux_stat_vals[k].append(float(s.get(k, 0.0)))
        if _has_lfan_ev:
            try:
                _nm_lfan_ev.loss(batch_y)
            except Exception:
                pass
            _ds = _nm_lfan_ev.get_debug_scalars()
            for k, v in _ds.items():
                _lfan_ev_accum.setdefault(k, []).append(float(v))
            if hasattr(_nm_lfan_ev, "get_debug_text"):
                _lfan_last_text = _nm_lfan_ev.get_debug_text()

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

    # Wavband-specific diagnostics (band-split gate stats, monotone reg)
    _nm_eval = getattr(model, "nm", None)
    if _nm_eval is not None and hasattr(_nm_eval, "get_split_stats"):
        ss = _nm_eval.get_split_stats()
        results["wav_g_mean"]  = ss["g_mean"]
        results["wav_g_min"]   = ss["g_min"]
        results["wav_g_max"]   = ss["g_max"]
        results["wav_L_split"] = ss["split_reg"]
    if _nm_eval is not None and hasattr(_nm_eval, "get_last_diag"):
        d = _nm_eval.get_last_diag()
        results.setdefault("wav_L_split", d.get("L_split", 0.0))

    # OSTN diagnostics
    if _nm_eval is not None and getattr(_nm_eval, "ostn_enabled", False) and hasattr(_nm_eval, "get_last_ostn_stats"):
        ostn_st = _nm_eval.get_last_ostn_stats()
        _pfx = f"[{debug_prefix}]" if debug_prefix is not None else ""
        print(
            f"{_pfx}[OSTN]"
            f" enabled={ostn_st['enabled']}"
            f" applied={ostn_st['applied']}"
            f" alpha_mean={ostn_st['alpha_mean']:.4f}"
            f" alpha_max={ostn_st['alpha_max']:.4f}"
            f" delta_mu_abs={ostn_st['delta_mu_abs_mean']:.4f}"
            f" delta_lsig_abs={ostn_st['delta_logsigma_abs_mean']:.4f}"
            f" base_olap={ostn_st['base_overlap_loss']:.6f}"
            f" refined_olap={ostn_st['refined_overlap_loss']:.6f}"
            f" base_mu={ostn_st['base_mu_abs_mean']:.4f}"
            f" refined_mu={ostn_st['refined_mu_abs_mean']:.4f}"
            f" base_sigma={ostn_st['base_sigma_mean']:.4f}"
            f" refined_sigma={ostn_st['refined_sigma_mean']:.4f}"
        )

    # FAN frequency-bin statistics (collected over the full eval pass)
    if _is_fan:
        results.update(_nm_fan.get_freq_stats())

    # FAN / LFAN averaged diagnostics (both expose get_debug_scalars())
    if _has_lfan_ev and _lfan_ev_accum:
        for k, vs in _lfan_ev_accum.items():
            results[k] = float(np.mean(vs))
        if debug_prefix is not None:
            if "fan_main_mse" in results:
                print(
                    f"[{debug_prefix}][FAN_DBG]"
                    f" fan_main_mse={results['fan_main_mse']:.6f}"
                    f" fan_res_mse={results['fan_res_mse']:.6f}"
                    f" fan_main_energy_ratio={results['fan_main_energy_ratio']:.4f}"
                    f" fan_res_energy_ratio={results['fan_res_energy_ratio']:.4f}"
                )
            if _lfan_last_text:
                print(f"[{debug_prefix}][LFAN] {_lfan_last_text}")

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
    if not cfg.run_name and cfg.norm_type.lower() == "fan":
        run_name = f"{run_name}_{cfg.fan_ablation_mode}"
    if not cfg.run_name and cfg.norm_type.lower() == "san":
        _san_base_stride = cfg.san_base_stride if cfg.san_base_stride > 0 else cfg.san_period_len
        run_name = (
            f"{run_name}_p{cfg.san_period_len}"
            f"_bs{_san_base_stride}"
            f"_hs{cfg.san_stride}"
            f"_l{cfg.san_force_extra_levels}"
        )
    result_root = cfg.result_dir
    if cfg.norm_type.lower() in {"none", "baseline", "no"}:
        result_root = os.path.join(cfg.result_dir, cfg.baseline_subdir)
    os.makedirs(result_root, exist_ok=True)
    result_path = os.path.join(result_root, f"{run_name}.json")
    best_checkpoint_path = os.path.join(result_root, f"{run_name}_best.pth")
    _wavband_metrics_path = os.path.join(result_root, f"{run_name}_wavband_epoch_metrics.jsonl")

    print(f"Train config: {cfg}")
    print(f"[OracleEval] wav_oracle_mode={cfg.wav_oracle_mode} eval_uses_oracle={cfg.wav_oracle_mode == 'eval_stats'}")
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
    _is_wavband_main = cfg.norm_type.lower() in {"wavband", "wavband_b"}
    if _is_wavband_main and cfg.wav_stage_train:
        if cfg.wav_pretrain_metric not in {"val_aux_total"}:
            raise ValueError(
                f"wav_pretrain_metric must be 'val_aux_total'; got {cfg.wav_pretrain_metric!r}"
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

    # Stage1 early-stop state
    _s1_best_aux = float("inf")
    _s1_no_improve = 0
    _s1_stopped_early = False
    _s1_end_epoch = int(cfg.wav_pretrain_epochs)   # may be shortened by early stop
    _pred_best_path = os.path.join(result_root, f"{run_name}_pred_best.pth")

    for epoch in range(cfg.epochs):
        # ── If stage1 stopped early and stop_to_stage2 requested, advance end epoch ──
        _in_stage1_main = (
            _is_wavband_main
            and cfg.wav_stage_train
            and epoch < _s1_end_epoch
        )
        if _s1_stopped_early and cfg.wav_pretrain_stop_to_stage2 and _in_stage1_main:
            _s1_end_epoch = epoch
            _in_stage1_main = False
            print(f"[stage1] stopped early → switching to stage2 at epoch {epoch}")

        # ── Stage boundary: rebuild optimizer + scheduler + reset early-stop ──
        if (
            _is_wavband_main
            and cfg.wav_stage_train
            and epoch == _s1_end_epoch
            and _s1_end_epoch > 0
        ):
            print(f"[stage transition] epoch {epoch}: entering joint training — "
                  "rebuilding optimizer, scheduler, resetting early-stop state")
            _set_requires_grad(model, True)
            # Load stage1 best predictor checkpoint (before any freezing)
            _nm_tr = getattr(model, "nm", None)
            if _nm_tr is not None and os.path.exists(_pred_best_path):
                _s1_ckpt = torch.load(_pred_best_path, map_location=cfg.device)
                _pred_tr = getattr(_nm_tr, "predictor", None)
                if _pred_tr is not None and "predictor" in _s1_ckpt:
                    _pred_tr.load_state_dict(_s1_ckpt["predictor"])
                if "band_logits" in _s1_ckpt:
                    _nm_tr.band_logits.data.copy_(_s1_ckpt["band_logits"].to(cfg.device))
                print(f"[stage2] loaded stage1 best predictor → {_pred_best_path}")
            # In stage2: freeze only band_logits; predictor stays trainable with reduced lr
            if _nm_tr is not None:
                _bl_s2 = getattr(_nm_tr, "band_logits", None)
                if _bl_s2 is not None and isinstance(_bl_s2, torch.nn.Parameter):
                    _bl_s2.requires_grad = False
                    print(f"[stage2] band_logits frozen")
            # Set predictor lr = backbone_lr * 0.1 for stage2
            cfg.wav_pred_lr = cfg.lr * 0.1
            cfg.wav_pred_weight_decay = 0.0
            print(f"[stage2] predictor lr={cfg.wav_pred_lr:.2e} wd=0.0 (trainable)")
            optimizer = _build_optimizer(model, cfg)
            remaining_epochs = cfg.epochs - epoch
            scheduler = CosineAnnealingLR(optimizer, T_max=max(remaining_epochs, 1))
            best_monitor = float("inf")
            best_epoch = -1
            best_val_at_best = float("inf")
            best_test_at_best = float("inf")
            ema_val_mse = None
            no_improve_count = 0

        # Compute oracle annealing probability (stage2 only)
        if _in_stage1_main or not _is_wavband_main or not cfg.wav_stage_train:
            _oracle_p = 1.0
        else:
            _stage2_epoch = epoch - _s1_end_epoch
            _oracle_p = _oracle_prob(cfg, _stage2_epoch)

        _freeze_pred = (
            not _in_stage1_main
            and _is_wavband_main
            and cfg.wav_stage_train
            and cfg.wav_freeze_pred_in_stage2
        )
        # Reset FAN stats before the training pass
        _nm_fan_ep = getattr(model, "nm", None)
        _is_fan_ep = _nm_fan_ep is not None and hasattr(_nm_fan_ep, "reset_freq_stats")
        if _is_fan_ep:
            _nm_fan_ep.reset_freq_stats()

        result = train_one_epoch(
            model, dataloader.train_loader, optimizer, cfg, scaler,
            epoch_idx=epoch, oracle_prob=_oracle_p,
            in_stage1=_in_stage1_main, freeze_wav_pred=_freeze_pred,
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

        # Collect FAN stats for the just-completed training pass
        _train_fan_stats = _nm_fan_ep.get_freq_stats() if _is_fan_ep else {}

        # ── Stage1: val_aux early stop, then skip full val/test evaluation ────
        if _in_stage1_main:
            val_aux = _wavband_val_aux(model, dataloader.val_loader, cfg)
            print(
                f"Epoch: {epoch + 1} [stage1-pretrain]"
                f" train_loss: {train_loss:.6f}"
                f" val_aux: {val_aux:.6f}"
            )
            # Save best predictor checkpoint
            s1_improved = val_aux < (_s1_best_aux - cfg.wav_pretrain_delta)
            if s1_improved:
                _s1_best_aux = val_aux
                _s1_no_improve = 0
                _nm_s1 = getattr(model, "nm", None)
                if _nm_s1 is not None:
                    _s1_state = {}
                    _pred_s1 = getattr(_nm_s1, "predictor", None)
                    if _pred_s1 is not None:
                        _s1_state["predictor"] = _pred_s1.state_dict()
                    _bl_s1 = getattr(_nm_s1, "band_logits", None)
                    if _bl_s1 is not None:
                        _s1_state["band_logits"] = _bl_s1.data.clone()
                    if _s1_state:
                        torch.save(_s1_state, _pred_best_path)
                        print(f"[stage1] predictor checkpoint saved → {_pred_best_path}")
            elif cfg.wav_pretrain_use_early_stop and (epoch + 1) >= cfg.wav_pretrain_min_epochs:
                _s1_no_improve += 1
                print(
                    f"[stage1] EarlyStopping counter: {_s1_no_improve} out of {cfg.wav_pretrain_patience}"
                )
                if _s1_no_improve >= cfg.wav_pretrain_patience:
                    _s1_stopped_early = True
                    if cfg.wav_pretrain_stop_to_stage2:
                        _s1_end_epoch = epoch + 1
                        print(
                            f"[stage1] early stop triggered at epoch {epoch + 1},"
                            f" _s1_end_epoch={_s1_end_epoch}"
                        )
                    else:
                        print(f"[stage1] Early stopping after epoch {epoch + 1}")
            scheduler.step()
            continue

        # Reset OSTN stream cache before each eval phase
        _nm_ostn = getattr(model, "nm", None)
        if _nm_ostn is not None and hasattr(_nm_ostn, "reset_ostn_eval_state"):
            _nm_ostn.reset_ostn_eval_state()

        val_metrics = evaluate(model, dataloader.val_loader, cfg, scaler, debug_prefix="VAL_DBG")

        if _nm_ostn is not None and hasattr(_nm_ostn, "reset_ostn_eval_state"):
            _nm_ostn.reset_ostn_eval_state()

        test_metrics = evaluate(model, dataloader.test_loader, cfg, scaler)

        if _is_wavband_main:
            # ── Wavband: 1 train line + 1 val line ───────────────────────────
            _nm_ep = getattr(model, "nm", None)
            _diag = _nm_ep.get_last_diag() if _nm_ep is not None and hasattr(_nm_ep, "get_last_diag") else {}
            _ss   = _nm_ep.get_split_stats() if _nm_ep is not None and hasattr(_nm_ep, "get_split_stats") else {}
            print(
                f"Epoch {epoch + 1} [train]"
                f" loss={train_loss:.6f}"
                f" task={train_stat.get('task_loss', 0.0):.6f}"
                f" aux={train_stat.get('aux_loss', 0.0):.6f}"
                f" aux_to_task={train_stat.get('aux_to_task_ratio', 0.0):.6f}"
            )
            print(
                f"Epoch {epoch + 1} [val]"
                f" mse={val_metrics['mse']:.6f} mae={val_metrics['mae']:.6f}"
                f" | sigL={_diag.get('sig_L_mean', float('nan')):.4f}"
                f" sigDj={_diag.get('sig_Dj_mean', float('nan')):.4f}"
                f" L_stats={_diag.get('L_stats', float('nan')):.4f}"
                f" L_split={_diag.get('L_split', float('nan')):.4f}"
                f" g_mean={_ss.get('g_mean', float('nan')):.4f}"
            )
            # Write per-epoch metrics to JSONL file
            _epoch_row = {
                "epoch": epoch + 1,
                "stage": "stage1" if _in_stage1_main else "stage2",
                "train_loss":    train_loss,
                "train_task":    train_stat.get("task_loss", 0.0),
                "train_aux":     train_stat.get("aux_loss", 0.0),
                "train_aux_to_task": train_stat.get("aux_to_task_ratio", 0.0),
                "val_mse":       val_metrics["mse"],
                "val_mae":       val_metrics["mae"],
                "val_rmse":      val_metrics.get("rmse", float("nan")),
                "test_mse":      test_metrics["mse"],
                "test_mae":      test_metrics["mae"],
                "test_rmse":     test_metrics.get("rmse", float("nan")),
                "sig_L_mean":    _diag.get("sig_L_mean", float("nan")),
                "sig_Dj_mean":   _diag.get("sig_Dj_mean", float("nan")),
                "rho_H_mean":    _diag.get("rho_H_mean", float("nan")),
                "L_stats":       _diag.get("L_stats", float("nan")),
                "L_split":       _diag.get("L_split", float("nan")),
                "g_mean":        _ss.get("g_mean", float("nan")),
                "g_min":         _ss.get("g_min", float("nan")),
                "g_max":         _ss.get("g_max", float("nan")),
                "g_vec":         _ss.get("g_vec", []),
                "split_reg":     _ss.get("split_reg", float("nan")),
            }
            with open(_wavband_metrics_path, "a", encoding="utf-8") as _mf:
                _mf.write(json.dumps(_epoch_row) + "\n")
        else:
            # ── Non-wavband: standard verbose output ──────────────────────────
            train_stat_str = ""
            if train_stat:
                train_stat_str = f" | task={train_stat.get('task_loss', 0.0):.6f}"
                train_stat_str += f" aux_to_task={train_stat.get('aux_to_task_ratio', 0.0):.6f}"
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
            if _train_fan_stats:
                print(
                    f"Epoch: {epoch + 1} [fan_train]"
                    f" selected_bin_mean={_train_fan_stats['selected_bin_mean']:.2f}"
                    f" selected_bin_std={_train_fan_stats['selected_bin_std']:.2f}"
                    f" selected_low_ratio={_train_fan_stats['selected_low_ratio']:.4f}"
                )
            # FAN epoch-averaged prediction quality diagnostics
            if "fan_main_mse" in train_stat:
                print(
                    f"Epoch: {epoch + 1} [fan_dbg]"
                    f" fan_main_mse={train_stat['fan_main_mse']:.6f}"
                    f" fan_res_mse={train_stat['fan_res_mse']:.6f}"
                    f" fan_main_energy_ratio={train_stat['fan_main_energy_ratio']:.4f}"
                    f" fan_res_energy_ratio={train_stat['fan_res_energy_ratio']:.4f}"
                )
            # LFAN epoch-averaged training diagnostics
            _lfan_tr_keys = [k for k in train_stat if k.startswith("lf_")]
            if _lfan_tr_keys:
                _core = ["lf_low_energy_ratio_x", "lf_removed_energy_ratio_x",
                         "lf_high_score_fraction", "lf_score_mean", "lf_corr_score_remove",
                         "lf_removed_coeff_mae", "lf_removed_shape_mae",
                         "lf_pred_removed_energy_ratio", "lf_true_removed_energy_ratio"]
                parts = " ".join(
                    f"{k}={train_stat[k]:.4f}" for k in _core if k in train_stat
                )
                print(f"Epoch: {epoch + 1} [lfan_train] {parts}")
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

    # Load the best training checkpoint before final evaluation.
    if os.path.exists(best_checkpoint_path):
        try:
            model.load_state_dict(torch.load(best_checkpoint_path, map_location=cfg.device))
        except RuntimeError as exc:
            print(
                f"[WARN] Failed to reload best checkpoint from {best_checkpoint_path}: {exc}"
            )
            print(
                "[WARN] This usually means another run wrote an incompatible SAN checkpoint to the same path. "
                "Continuing with the in-memory model from the current run."
            )

    # Reset OSTN stream cache before final test
    _nm_final = getattr(model, "nm", None)
    if _nm_final is not None and hasattr(_nm_final, "reset_ostn_eval_state"):
        _nm_final.reset_ostn_eval_state()

    test_metrics = evaluate(model, dataloader.test_loader, cfg, scaler)
    print(
        "test_results: "
        f"{{'mae': {test_metrics['mae']:.6f}, "
        f"'mape': {test_metrics['mape']:.6f}, "
        f"'mse': {test_metrics['mse']:.6f}, "
        f"'rmse': {test_metrics['rmse']:.6f}}}"
    )

    # Oracle-norm upper bound analysis (three modes)
    _nm_oracle_final = getattr(model, "nm", None)
    if _nm_oracle_final is not None and hasattr(_nm_oracle_final, "set_oracle_stats"):
        _oracle_runs = [
            ("oracle_mean_only (true mean + pred std)", "mean_only",
             dataclasses.replace(cfg, san_oracle_mean_only=True, san_oracle_norm=False, san_oracle_std_only=False, ostn_enabled=False)),
            ("oracle_std_only  (pred mean + true std)", "std_only",
             dataclasses.replace(cfg, san_oracle_std_only=True,  san_oracle_norm=False, san_oracle_mean_only=False, ostn_enabled=False)),
            ("oracle_both      (true mean + true std)", "both",
             dataclasses.replace(cfg, san_oracle_norm=True, san_oracle_mean_only=False, san_oracle_std_only=False, ostn_enabled=False)),
        ]
        for label, _mode, _ocfg in _oracle_runs:
            if hasattr(_nm_oracle_final, "reset_ostn_eval_state"):
                _nm_oracle_final.reset_ostn_eval_state()
            _om = evaluate(model, dataloader.test_loader, _ocfg, scaler)
            print(
                f"{label}: "
                f"mse={_om['mse']:.6f}  mae={_om['mae']:.6f}  "
                f"Δmse={_om['mse'] - test_metrics['mse']:+.6f}  "
                f"Δmae={_om['mae'] - test_metrics['mae']:+.6f}"
            )

    _norm_plot_path = os.path.join(result_root, f"{run_name}_norm_stats.png")
    plot_norm_stats(model, dataloader, cfg, _norm_plot_path)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "dataset": cfg.dataset_type,
                "backbone": cfg.backbone_type,
                "norm_type": cfg.norm_type,
                "fan_ablation_mode": cfg.fan_ablation_mode,
                "selected_bin_mean":  test_metrics.get("selected_bin_mean"),
                "selected_bin_std":   test_metrics.get("selected_bin_std"),
                "selected_low_ratio": test_metrics.get("selected_low_ratio"),
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


def plot_norm_stats(model, dataloader, cfg, save_path: str) -> None:
    """Compare SAN predicted future window stats vs oracle stats across train/val/test.

    oracle : per-window mean/std computed directly from batch_y (ground-truth future)
    predicted : per-window mean/std from nm._refined_pred_stats by default, plus
    a separate level-0 base predictor curve from nm._base_pred_stats.

    Both are in the same space (after global scaler).  SAN is supervised on exactly
    these quantities, so the two lines should overlap in the training split and diverge
    in val/test when distribution shifts.

    Each point = one batch, value = mean over (B, N_pred, C).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[NORM_STATS] matplotlib not available, skipping plot.")
        return

    nm = getattr(model, "nm", None)
    if nm is None or not hasattr(nm, "normalize") or not hasattr(nm, "_pred_stats"):
        print("[NORM_STATS] no SAN normalization module, skipping plot.")
        return

    C = getattr(nm, "channels", getattr(nm, "enc_in", None))
    if C is None or not hasattr(nm, "_extract_windows") or not hasattr(nm, "_compute_window_stats"):
        print("[NORM_STATS] SAN window-stat helpers not found, skipping plot.")
        return

    device = torch.device(cfg.device)

    pred_base_mean_vals, pred_refined_mean_vals, oracle_mean_vals = [], [], []
    pred_base_std_vals, pred_refined_std_vals, oracle_std_vals = [], [], []
    split_boundaries = []   # batch indices where a new split begins

    splits = [
        ("train", dataloader.train_loader),
        ("val",   dataloader.val_loader),
        ("test",  dataloader.test_loader),
    ]

    model.eval()
    with torch.no_grad():
        total = 0
        for split_name, loader in splits:
            split_boundaries.append(total)
            # Reset OSTN stream cache so it doesn't bleed across splits during viz pass
            if hasattr(nm, "reset_ostn_eval_state"):
                nm.reset_ostn_eval_state()
            for batch_x, batch_y, _origin_y, _batch_x_enc, _batch_y_enc in loader:
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()

                nm.normalize(batch_x)

                ps_refined = getattr(nm, "_refined_pred_stats", nm._pred_stats)
                ps_base = getattr(nm, "_base_pred_stats", None)
                if ps_refined is None:
                    continue

                pred_refined_mean_vals.append(float(ps_refined[:, :, :C].mean().item()))
                pred_refined_std_vals.append(float(ps_refined[:, :, C:].mean().item()))

                if ps_base is None:
                    pred_base_mean_vals.append(float("nan"))
                    pred_base_std_vals.append(float("nan"))
                else:
                    pred_base_mean_vals.append(float(ps_base[:, :, :C].mean().item()))
                    pred_base_std_vals.append(float(ps_base[:, :, C:].mean().item()))

                # Oracle: extract sliding windows, compute window-level stats
                y_w = nm._extract_windows(batch_y)
                y_mean, y_std = nm._compute_window_stats(y_w)
                oracle_mean_vals.append(float(y_mean.mean().item()))
                oracle_std_vals.append( float(y_std.mean().item()))
                total += 1

    if not pred_refined_mean_vals:
        print("[NORM_STATS] no data collected, skipping plot.")
        return

    xs          = np.arange(len(pred_refined_mean_vals))
    split_ends  = split_boundaries[1:] + [len(xs)]   # end index for each split
    split_names = ["train", "val", "test"]
    split_colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, (ax_mean, ax_std) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(
        f"SAN Predicted vs Oracle Window Stats"
        f" — {cfg.norm_type} | {cfg.dataset_type} | pred={cfg.pred_len}",
        fontsize=13,
    )

    def _draw_splits(ax):
        for i, (s, e) in enumerate(zip(split_boundaries, split_ends)):
            ax.axvspan(s, e, alpha=0.07, color=split_colors[i], zorder=0)
            ax.axvline(s, color=split_colors[i], linestyle="--",
                       linewidth=0.9, zorder=1)
            mid = (s + e) / 2
            ax.text(mid, ax.get_ylim()[1], split_names[i],
                    ha="center", va="top", fontsize=9,
                    color=split_colors[i], fontweight="bold")

    # ---- Mean panel ----
    ax_mean.plot(xs, oracle_mean_vals, color="steelblue",  linewidth=0.9,
                 alpha=0.9, label="oracle mean")
    ax_mean.plot(xs, pred_base_mean_vals, color="darkorange", linewidth=0.9,
                 alpha=0.8, label="base mean", linestyle="--")
    ax_mean.plot(xs, pred_refined_mean_vals, color="seagreen", linewidth=1.0,
                 alpha=0.9, label="refined mean")
    ax_mean.set_ylabel("Window Mean", fontsize=10)
    ax_mean.legend(fontsize=9, loc="upper right")
    ax_mean.grid(True, alpha=0.25)
    _draw_splits(ax_mean)

    # ---- Std panel ----
    ax_std.plot(xs, oracle_std_vals, color="steelblue",  linewidth=0.9,
                alpha=0.9, label="oracle std")
    ax_std.plot(xs, pred_base_std_vals, color="darkorange", linewidth=0.9,
                alpha=0.8, label="base std", linestyle="--")
    ax_std.plot(xs, pred_refined_std_vals, color="seagreen", linewidth=1.0,
                alpha=0.9, label="refined std")
    ax_std.set_ylabel("Window Std", fontsize=10)
    ax_std.set_xlabel("Batch index", fontsize=10)
    ax_std.legend(fontsize=9, loc="upper right")
    ax_std.grid(True, alpha=0.25)
    _draw_splits(ax_std)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[NORM_STATS] plot saved → {save_path}")


if __name__ == "__main__":
    main()
