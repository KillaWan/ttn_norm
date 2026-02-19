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


def _build_dataloader(cfg):
    _ensure_fan_on_path()
    from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader

    dataset = _parse_type(cfg.dataset_type)(root=cfg.data_path)
    scaler = _parse_scaler(cfg.scaler_type)(device=cfg.device)
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
    return dataloader, dataset, scaler


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
    aux_loss_scale: float = 1.0
    aux_loss_min_scale: float = 0.2
    aux_loss_decay_start_epoch: int = 4
    aux_loss_decay_epochs: int = 16

    scaler_type: str = "StandarScaler"
    scale_in_train: bool = False
    train_ratio: float = 0.7
    val_ratio: float = 0.2
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
    predict_n_time: bool = True
    pred_hidden_dim: int = 64
    pred_dropout: float = 0.1
    pred_loss_weight: float = 1.0
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


def build_model(cfg: TrainConfig, num_features: int) -> TTNModel:
    if cfg.norm_type.lower() in {"none", "baseline", "no"}:
        norm_model: nn.Module | None = nn.Identity()
    else:
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
        )
    label_len = cfg.label_len or (cfg.window // 2)
    label_len = min(label_len, cfg.window)
    backbone_kwargs = _build_backbone_kwargs(cfg, num_features, label_len)
    if cfg.backbone_kwargs:
        extra = json.loads(cfg.backbone_kwargs)
        backbone_kwargs.update(extra)
    return TTNModel(
        backbone_type=cfg.backbone_type,
        backbone_kwargs=backbone_kwargs,
        norm_model=norm_model,
        is_former=True if cfg.force_former else None,
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


def train_one_epoch(model, loader, optimizer, cfg, scaler, epoch_idx: int):
    model.train()
    loss_fn = nn.MSELoss()
    losses = []
    aux_scale = _aux_scale(cfg, epoch_idx)
    print(f"aux_loss_scale: {aux_scale:.6f}")
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
            loss = task_loss + aux_loss * aux_scale

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            losses.append(loss.item())
            pbar.update(batch_x.size(0))
            pbar.set_postfix(loss=loss.item())
    
    # Get gate statistics from model
    gate_stats_str = ""
    if hasattr(model, "nm") and model.nm is not None and hasattr(model.nm, "get_last_gate_stats"):
        stats = model.nm.get_last_gate_stats()
        gate_stats_str = (
            f" | gate: mean={stats['gate_mean']:.4f}, "
            f"sumF={stats['gate_sum_f']:.4f}, maxF={stats['gate_max_f']:.4f}, "
            f"entF={stats['gate_ent_f']:.4f}"
        )
    
    return float(np.mean(losses)) if losses else 0.0, gate_stats_str


@torch.no_grad()
def evaluate(model, loader, cfg, scaler):
    model.eval()
    metrics = _build_metrics(torch.device(cfg.device))
    for metric in metrics.values():
        metric.reset()
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

    return {name: float(metric.compute()) for name, metric in metrics.items()}


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
    dataloader, dataset, scaler = _build_dataloader(cfg)

    model = build_model(cfg, dataset.num_features).to(cfg.device)
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
        if isinstance(result, tuple):
            train_loss, gate_info = result
        else:
            train_loss = result
            gate_info = ""
        val_metrics = evaluate(model, dataloader.val_loader, cfg, scaler)
        test_metrics = evaluate(model, dataloader.test_loader, cfg, scaler)

        print(f"Epoch: {epoch + 1} Traininng loss : {train_loss:.6f}{gate_info}")
        print(
            "vali_results: "
            f"{{'mae': {val_metrics['mae']:.6f}, "
            f"'mape': {val_metrics['mape']:.6f}, "
            f"'mse': {val_metrics['mse']:.6f}, "
            f"'rmse': {val_metrics['rmse']:.6f}}}"
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
