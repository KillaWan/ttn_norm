from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from ttn_norm.experiments.train import TrainConfig, _build_dataloader, _set_seed, build_model


def _ensure_fan_on_path() -> None:
    fan_root = os.path.join(_ROOT, "FAN")
    if fan_root not in sys.path:
        sys.path.append(fan_root)


def _fan_main_freq_part(x: torch.Tensor, k: int, rfft: bool = True):
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)
    k_values = torch.topk(xf.abs(), k, dim=1)
    indices = k_values.indices
    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask
    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()
    norm_input = x - x_filtered
    return norm_input, x_filtered, xf, indices


def _avg_gate_spectrum(g_local: torch.Tensor) -> torch.Tensor:
    # g_local: (B, C, F, T)
    return g_local.mean(dim=(0, 1, 3))


def _avg_freq_energy_rfft(xf: torch.Tensor) -> torch.Tensor:
    # xf: (B, F, C) complex
    return xf.abs().mean(dim=(0, 2))


def _avg_freq_energy_stft(x_tf: torch.Tensor) -> torch.Tensor:
    # x_tf: (B, C, F, T) complex
    return x_tf.abs().mean(dim=(0, 1, 3))


def _topk_hist(indices: torch.Tensor, freq_bins: int) -> np.ndarray:
    # indices: (B, K, C) on freq dim
    idx = indices.reshape(-1).cpu().numpy()
    hist = np.bincount(idx, minlength=freq_bins).astype(np.float64)
    hist = hist / max(hist.sum(), 1.0)
    return hist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", default="ETTh1")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--freq", default=None)
    parser.add_argument("--num-batches", type=int, default=4)

    parser.add_argument("--localtf-ckpt", required=True)
    parser.add_argument("--fan-ckpt", required=True)
    parser.add_argument("--fan-topk", type=int, default=None)
    parser.add_argument("--fan-rfft", action="store_true", default=True)
    parser.add_argument("--result-dir", default="./results/compare")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    cfg = TrainConfig(
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        device=args.device,
        seed=args.seed,
        window=args.window,
        pred_len=args.pred_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        label_len=args.label_len,
        freq=args.freq,
        norm_type="localtf",
        backbone_type="SCINet",
        gate_threshold=args.gate_threshold,
        gate_temperature=args.gate_temperature,
    )

    dataloader, dataset, _ = _build_dataloader(cfg)

    # LocalTF model
    localtf_model = build_model(cfg, dataset.num_features).to(cfg.device)
    localtf_state = torch.load(args.localtf_ckpt, map_location=cfg.device)
    localtf_model.load_state_dict(localtf_state)
    localtf_model.eval()

    # FAN model
    _ensure_fan_on_path()
    from torch_timeseries.models.SCINet import SCINet
    from torch_timeseries.normalizations.FAN import FAN
    from torch_timeseries.norm_experiments.Model import Model

    fan_topk_map = {
        "ETTh1": 4,
        "ETTh2": 3,
        "ETTm1": 11,
        "ETTm2": 5,
        "ExchangeRate": 2,
        "Electricity": 3,
        "Traffic": 30,
        "Weather": 2,
    }
    fan_topk = (
        args.fan_topk if args.fan_topk is not None else fan_topk_map.get(args.dataset_type, 4)
    )

    scinet_kwargs = dict(
        output_len=cfg.pred_len,
        input_len=cfg.window,
        input_dim=dataset.num_features,
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
    f_model = SCINet(**scinet_kwargs)
    n_model = FAN(
        seq_len=cfg.window,
        pred_len=cfg.pred_len,
        enc_in=dataset.num_features,
        freq_topk=fan_topk,
        rfft=args.fan_rfft,
    )
    fan_model = Model("SCINet", f_model, n_model).to(cfg.device)
    fan_state = torch.load(args.fan_ckpt, map_location=cfg.device)
    fan_model.load_state_dict(fan_state)
    fan_model.eval()

    gate_sums = None
    gate_count = 0
    fan_hist_sum = None
    fan_energy_sum = None
    local_energy_sum = None

    loader = dataloader.test_loader
    for i, (batch_x, *_rest) in enumerate(loader):
        if i >= args.num_batches:
            break
        batch_x = batch_x.to(cfg.device).float()

        # LocalTF gate spectrum
        _, state = localtf_model.nm.normalize(batch_x, return_state=True)
        g_avg = _avg_gate_spectrum(state.g_local)  # (F)
        g_avg = g_avg.detach().cpu()
        if gate_sums is None:
            gate_sums = g_avg.clone()
        else:
            gate_sums += g_avg
        gate_count += 1

        # FAN topk histogram + energy
        _, _, xf, indices = _fan_main_freq_part(batch_x, fan_topk, rfft=args.fan_rfft)
        hist = _topk_hist(indices, xf.shape[1])
        energy = _avg_freq_energy_rfft(xf).detach().cpu().numpy()

        if fan_hist_sum is None:
            fan_hist_sum = hist
            fan_energy_sum = energy
        else:
            fan_hist_sum += hist
            fan_energy_sum += energy

        # LocalTF energy
        x_tf = localtf_model.nm.stft(batch_x)
        local_energy = _avg_freq_energy_stft(x_tf).detach().cpu().numpy()
        if local_energy_sum is None:
            local_energy_sum = local_energy
        else:
            local_energy_sum += local_energy

    gate_avg = (gate_sums / max(gate_count, 1)).numpy()
    fan_hist_avg = fan_hist_sum / max(gate_count, 1)
    fan_energy_avg = fan_energy_sum / max(gate_count, 1)
    local_energy_avg = local_energy_sum / max(gate_count, 1)

    os.makedirs(args.result_dir, exist_ok=True)
    out = {
        "config": asdict(cfg),
        "localtf_ckpt": args.localtf_ckpt,
        "fan_ckpt": args.fan_ckpt,
        "fan_topk": fan_topk,
        "gate_avg": gate_avg.tolist(),
        "fan_topk_hist": fan_hist_avg.tolist(),
        "fan_freq_energy": fan_energy_avg.tolist(),
        "local_freq_energy": local_energy_avg.tolist(),
    }
    suffix = args.output_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    out_path = os.path.join(
        args.result_dir,
        f"filter_compare_{args.dataset_type}_P{args.pred_len}{suffix}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved filter analysis to {out_path}")


if __name__ == "__main__":
    main()
