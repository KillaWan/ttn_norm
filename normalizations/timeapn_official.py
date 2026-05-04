"""timeapn_official.py — Self-contained port of the official TimeAPN (APN) module.

Ported from:
  models/APN.py          — APN, Statics_MLP, norm_sliding, norm, normalize, de_normalize
  utils/learnable_wavelet.py — DWT1D (learnable QMF filter bank)
  utils/TCN.py           — TemporalConvNet, TemporalBlock, Chomp1d

Design notes
------------
- All heavy optional deps (pytorch_wavelets, scipy) are avoided.
  pywt is used ONLY at __init__ time to read wavelet filter coefficients; it is
  not imported at module-level so the module loads even without pywt.
- DWT1D is a self-contained learnable QMF bank derived from the original
  learnable_wavelet.py.  It requires only PyTorch and pywt (at init time).
- Statics_MLP matches the official architecture, including the known std_r1 /
  std_r discrepancy that is faithfully reproduced.
- TCN channels follow the official data_path-dependent branches when a
  data_path/dataset hint is provided:
    traffic -> [512, 1024, 1024, 512, enc_in]
    elec    -> [256, 512, 1024, 512, enc_in]
    wea     -> [32, 64, 32, enc_in]
    else    -> [16, 32, 64, 32, enc_in]

Interface expected by SANRouteNorm
-----------------------------------
  apn = OfficialAPN(seq_len, pred_len, enc_in, kernel_len, hkernel_len,
                    j, learnable, wavelet, dr, pd_model, pd_ff, pe_layers)

  # normalize: input (B,T,C) → normalized (B,T,C), pred_ms tuple, seq_ms tuple
  norm_x, pred_ms, seq_ms = apn.normalize(x)
  # pred_ms = (pred_m, pred_s) each (B, C, pred_len) channel-first
  # seq_ms  = (seq_m,  seq_s)  each (B, C, seq_len)  channel-first

  # de_normalize: input (B,T,C), station_pred (B,T,2C) → (B,T,C)
  y_out = apn.de_normalize(y_norm, station_pred_tensor)

  # norm_sliding: (B,C,T) → (x_center, (m, phi_f))  channel-first
  xc, (m, phi) = apn.norm_sliding(x_BCT)
"""
from __future__ import annotations

import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DWT1D — learnable QMF analysis / synthesis filter bank
# (ported from utils/learnable_wavelet.py; no pytorch_wavelets dependency)
# ---------------------------------------------------------------------------

def official_tcn_channels(enc_in: int, data_path: Optional[str] = None) -> List[int]:
    """Return the official TimeAPN TCN channel schedule for a dataset hint.

    The original ``models/APN.py`` branches on ``configs.data_path`` using
    substring checks.  ``ttn_norm`` usually has a dataset name rather than the
    exact CSV path, so this helper accepts either.
    """
    key = (data_path or "").lower()
    if "traffic" in key:
        return [512, 1024, 1024, 512, enc_in]
    if "elec" in key:
        return [256, 512, 1024, 512, enc_in]
    if "wea" in key:
        return [32, 64, 32, enc_in]
    return [16, 32, 64, 32, enc_in]

def _dwt_coeff_len(n: int, filt_len: int) -> int:
    """Compute DWT coefficient array length (mode='zero')."""
    return (n + filt_len - 1) // 2


def _afb1d(x: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor) -> torch.Tensor:
    """1-D analysis filter bank via stride-2 conv2d on dim=3.

    x:  (B, C, 1, T)
    h0, h1: (1, 1, 1, L) filters
    returns lohi: (B, 2C, 1, T//2+…) — even channels = lo, odd = hi
    """
    N = x.shape[3]
    L = h0.shape[3]
    s = (1, 2)
    h = torch.cat([h0, h1] * x.shape[1], dim=0)  # (2C, 1, 1, L)
    outsize = _dwt_coeff_len(N, L)
    p = 2 * (outsize - 1) - N + L
    if p % 2 == 1:
        x = F.pad(x, (0, 1, 0, 0))
    pad = (0, p // 2)
    return F.conv2d(x, h, padding=pad, stride=s, groups=x.shape[1])


def _sfb1d(
    lo: torch.Tensor,
    hi: torch.Tensor,
    g0: torch.Tensor,
    g1: torch.Tensor,
) -> torch.Tensor:
    """1-D synthesis filter bank via stride-2 conv_transpose2d on dim=3."""
    L = g0.shape[3]
    s = (1, 2)
    g0 = g0.repeat(lo.shape[1], 1, 1, 1)
    g1 = g1.repeat(lo.shape[1], 1, 1, 1)
    pad = (0, L - 2)
    return (
        F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=lo.shape[1])
        + F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=hi.shape[1])
    )


class DWT1D(nn.Module):
    """Learnable 1-D DWT (QMF bank) — ported from official learnable_wavelet.py.

    Args:
        J:          Decomposition levels (default 1).
        wave:       Wavelet name understood by pywt (default 'bior3.5').
        learnable:  Whether filter coefficients are learnable parameters.
    """

    # Hardcoded bior3.5 dec_lo for environments without pywt at runtime.
    _BIOR35_DEC_LO = [
        -0.013810679320049757,  0.04143203796014927,  0.052480581416189075,
        -0.26792717880896527,  -0.07181553246425873,  0.966747552403483,
         0.966747552403483,    -0.07181553246425873, -0.26792717880896527,
         0.052480581416189075,  0.04143203796014927, -0.013810679320049757,
    ]

    def __init__(self, J: int = 1, wave: str = "bior3.5", learnable: bool = True):
        super().__init__()
        self.J = J
        self.mode = "zero"
        hac = self._get_dec_lo(wave)
        self.N = len(hac)
        self.hac = nn.Parameter(hac, requires_grad=learnable)

    @staticmethod
    def _get_dec_lo(wave: str) -> torch.Tensor:
        try:
            import pywt  # optional at runtime
            coeffs = list(pywt.Wavelet(wave).dec_lo)
        except Exception:
            # Fall back to hardcoded bior3.5
            coeffs = DWT1D._BIOR35_DEC_LO
        return torch.tensor(coeffs, dtype=torch.float32)

    def _filters(self):
        hac = self.hac
        miu = torch.ones_like(hac)
        miu[::2] = -1
        hdc = hac.flip(-1) * miu
        h0 = hac.view(1, 1, 1, -1)
        h1 = hdc.view(1, 1, 1, -1)
        return h0, h1

    def forward(self, x, Inverse: int = 0):
        h0, h1 = self._filters()
        if Inverse == 0:
            # Analysis: (B, C, T) → x0 (B, C, T//2), highs list
            x0, highs = x, []
            for _ in range(self.J):
                # reshape for conv2d
                x0_4d = x0[:, :, None, :]  # (B, C, 1, T)
                lohi = _afb1d(x0_4d, h0, h1)  # (B, 2C, 1, T')
                x0_4d_new = lohi[:, ::2, :, :].contiguous()   # even = lo
                x1_4d = lohi[:, 1::2, :, :].contiguous()       # odd  = hi
                x0 = x0_4d_new[:, :, 0, :]  # (B, C, T')
                x1 = x1_4d[:, :, 0, :]      # (B, C, T')
                highs.append(x1)
            return x0, highs
        else:
            # Synthesis: (x0, highs) → (B, C, T)
            x0, highs = x
            for x1 in reversed(highs):
                if x1 is None:
                    x1 = torch.zeros_like(x0)
                if x0.shape[-1] > x1.shape[-1]:
                    x0 = x0[..., :-1]
                lo = x0[:, :, None, :]
                hi = x1[:, :, None, :]
                out = _sfb1d(lo, hi, h0, h1)  # (B, C, 1, T)
                x0 = out[:, :, 0, :]
            return x0


# ---------------------------------------------------------------------------
# TCN — Temporal Convolutional Network
# (copied verbatim from utils/TCN.py)
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, 1, dilation, padding, dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: nn.Module,
        drop_rate: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            activation,
            nn.Linear(d_ff, d_model, bias=bias),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Statics_MLP — ported verbatim from APN.py
# ---------------------------------------------------------------------------

class Statics_MLP(nn.Module):
    """Channel-shared MLP predicting future statistics (mean, phase).

    All inputs and outputs are channel-first: (B, C, T).

    Architecture matches official APN.py Statics_MLP exactly, including
    the known std_r1 / std_r discrepancy in the forward pass.

    Args:
        seq_len:    Historical input length (T dimension).
        d_model:    Hidden dimension (pd_model).
        d_ff:       FFN hidden dimension (pd_ff).
        enc_in:     Number of channels (C).
        pred_len:   Prediction horizon.
        drop_rate:  Dropout probability.
        layer:      Number of FFN layers in mean_ffn.
        tcn_channels: TCN channel list for std_ffn.  Defaults to the official
                    data_path-dependent branch when data_path is provided.
        data_path: Dataset/path hint used to select official TCN channels.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int,
        enc_in: int,
        pred_len: int,
        drop_rate: float = 0.1,
        bias: bool = False,
        layer: int = 1,
        tcn_channels: Optional[List[int]] = None,
        data_path: Optional[str] = None,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = official_tcn_channels(enc_in, data_path)
        self.tcn_channels = list(tcn_channels)

        project = nn.Sequential(
            nn.Linear(seq_len, d_model, bias=bias), nn.Dropout(drop_rate)
        )
        self.m_project  = copy.deepcopy(project)
        self.s_project  = copy.deepcopy(project)
        self.mean_proj  = copy.deepcopy(project)
        self.std_proj   = copy.deepcopy(project)
        self.mean_proj1 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(drop_rate)
        )
        self.m_concat = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Dropout(drop_rate)
        )
        self.s_concat = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Dropout(drop_rate)
        )
        ffn_tcn = TemporalConvNet(
            num_inputs=enc_in, num_channels=self.tcn_channels, kernel_size=3, dropout=drop_rate
        )
        ffn1 = nn.Sequential(
            *[FFN(d_model, d_ff, nn.LeakyReLU(), drop_rate, bias) for _ in range(layer)]
        )
        self.mean_ffn = copy.deepcopy(ffn1)
        self.std_ffn  = copy.deepcopy(ffn_tcn)
        self.mean_pred = nn.Linear(d_model, pred_len, bias=bias)
        self.std_pred  = nn.Linear(d_model, pred_len, bias=bias)
        self.down_sampling_window = 2
        self.down_sampling_layers = 3
        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    seq_len // (self.down_sampling_window ** i),
                    pred_len,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        x2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mean: (B, C, T)  — sliding mean (seq_m)
            std:  (B, C, T)  — sliding phase (seq_s / phi_f)
            x:    (B, C, T)  — raw input (optional)
            x2:   (B, C, T)  — secondary raw input (optional, for DWT branch)

        Returns:
            (pred_mean, pred_std): each (B, C, pred_len)
        """
        m_all = mean.mean(dim=-1, keepdim=True)  # (B, C, 1)
        s_all = std.mean(dim=-1, keepdim=True)   # (B, C, 1)
        mean_r = mean - m_all
        std_r  = std  - s_all
        mean_r = self.mean_proj(mean_r)  # (B, C, d_model)
        std_r  = self.std_proj(std_r)    # (B, C, d_model)

        if x is not None:
            m_orig = self.m_project(x - m_all)
            s_inp  = (
                torch.abs(torch.fft.fft(x)) if x2 is None else x2 - s_all
            )
            s_ori  = self.s_project(s_inp)
            mean_r, std_r1 = (  # NOTE: std_r1 assigned but std_r used below — faithful port
                self.m_concat(torch.cat([m_orig, mean_r], dim=-1)),
                self.s_concat(torch.cat([s_ori,  std_r ], dim=-1)),
            )
            # std_r is still self.std_proj(std - s_all) — official code behaviour

        # std_r shape: (B, C, d_model); treat as (N=B, channels=C, L=d_model) for TCN
        mean_r = self.mean_ffn(mean_r)   # (B, C, d_model)
        std_r  = self.std_ffn(std_r)     # (B, C, d_model) via TCN
        mean_r = self.mean_pred(mean_r)  # (B, C, pred_len)
        std_r  = self.std_pred(std_r)    # (B, C, pred_len)
        mean_out = mean_r + m_all        # broadcasts m_all (B,C,1) → (B,C,pred_len)
        std_out  = std_r  + s_all
        return mean_out, std_out


# ---------------------------------------------------------------------------
# OfficialAPN — main module
# ---------------------------------------------------------------------------

class OfficialAPN(nn.Module):
    """Official TimeAPN normalisation module.

    Ported from models/APN.py (class APN) with the following changes:
    - Removed scipy / pytorch_wavelets / ADF / S_Mamba dependencies.
    - DWT1D is self-contained (pywt used only at __init__ for filter init).
    - Statics_MLP tcn_channels configurable (defaults to [16,32,64,32,enc_in]).
    - station_type hard-wired to 'adaptive' (the only mode used here).

    All tensor shapes follow the official channel-first convention
    (B, C, T) inside the module; (B, T, C) at the public interface.

    Args:
        seq_len:      Historical window length.
        pred_len:     Prediction horizon.
        enc_in:       Number of channels.
        kernel_len:   Sliding-window kernel size for norm_sliding (default 7).
        hkernel_len:  Sliding-window kernel for DWT-band norm_sliding (default 5).
        j:            DWT decomposition levels (0 = no DWT).
        learnable:    Learnable DWT filter coefficients.
        wavelet:      Wavelet name for DWT1D (default 'bior3.5').
        dr:           Dropout rate for Statics_MLP.
        pd_model:     Hidden dimension d_model for Statics_MLP (default 128).
        pd_ff:        FFN hidden dimension d_ff for Statics_MLP (default 128).
        pe_layers:    Number of FFN layers in mean_ffn (default 2).
        tcn_channels: TCN channel list for std_ffn.  If omitted, selected from
                      data_path/dataset using the official APN.py branches.
        data_path:    Dataset/path hint used for official TCN channel selection.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_len: int = 7,
        hkernel_len: int = 5,
        j: int = 1,
        learnable: bool = True,
        wavelet: str = "bior3.5",
        dr: float = 0.05,
        pd_model: int = 128,
        pd_ff: int = 128,
        pe_layers: int = 2,
        tcn_channels: Optional[List[int]] = None,
        data_path: Optional[str] = None,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.enc_in    = enc_in
        self.kernel    = kernel_len
        self.hkernel   = hkernel_len
        self.j         = j

        # Padding for sliding window (replication)
        self.pad  = nn.ReplicationPad1d(
            (kernel_len // 2, kernel_len // 2 - ((kernel_len + 1) % 2))
        )
        self.hpad = nn.ReplicationPad1d(
            (hkernel_len // 2, hkernel_len // 2 - ((hkernel_len + 1) % 2))
        )

        # DWT (j=0 → skipped)
        if j > 0:
            self.dwt = DWT1D(J=j, wave=wavelet, learnable=learnable)
            self.dwt_ratio = nn.Parameter(
                torch.clamp(torch.zeros(1, enc_in, 1), min=0.0, max=1.0)
            )
        else:
            self.dwt = None
            self.dwt_ratio = None

        # Shared Statics_MLP
        if tcn_channels is None:
            tcn_channels = official_tcn_channels(enc_in, data_path)
        self.tcn_channels = list(tcn_channels)
        self.mlp = Statics_MLP(
            seq_len=seq_len,
            d_model=pd_model,
            d_ff=pd_ff,
            enc_in=enc_in,
            pred_len=pred_len,
            drop_rate=dr,
            layer=pe_layers,
            tcn_channels=self.tcn_channels,
            data_path=data_path,
        )

    # ------------------------------------------------------------------
    # Core: sliding-window normalisation
    # ------------------------------------------------------------------

    def norm_sliding(
        self,
        x: torch.Tensor,
        kernel: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sliding-window mean subtraction + FFT phase extraction.

        Exactly matches official APN.norm_sliding / exp_main.norm_sliding.

        Args:
            x:      (B, C, T) channel-first.
            kernel: override for kernel size (uses self.kernel by default).

        Returns:
            x_center: (B, C, T) — mean-centred series.
            (m, phi_f): each (B, C, T) — sliding mean and FFT phase.
        """
        if kernel is None:
            kernel, pad = self.kernel, self.pad
        else:
            pad = self.hpad

        x_window = x.unfold(-1, kernel, 1)       # (B, C, T-K+1, K)
        m = x_window.mean(dim=-1)                # (B, C, T-K+1)
        s = x_window.std(dim=-1)                 # (B, C, T-K+1)
        m, s = pad(m), pad(s)                    # (B, C, T)
        x_center = x - m                         # (B, C, T)  [no / s]

        F_w   = torch.fft.fft(x_center, dim=2)  # (B, C, T) complex
        phi_f = torch.angle(F_w)                 # (B, C, T) real

        return x_center, (m, phi_f)

    # ------------------------------------------------------------------
    # Full normalisation (time + DWT branches)
    # ------------------------------------------------------------------

    def norm(
        self,
        x: torch.Tensor,
        predict: bool = True,
    ) -> Tuple[torch.Tensor, Tuple, Tuple]:
        """Full normalisation matching official APN.norm.

        Args:
            x:       (B, C, T) channel-first.
            predict: If False, skip mlp and DWT prediction (returns dummy pred).

        Returns:
            norm_x:  (B, C, T) normalised input.
            seq_ms:  (seq_m, seq_s) each (B, C, T).
            pred_ms: (pred_m, pred_s) each (B, C, pred_len).
        """
        norm_x, (seq_m, seq_s) = self.norm_sliding(x)

        if not predict:
            dummy = torch.zeros(
                x.shape[0], x.shape[1], self.pred_len,
                device=x.device, dtype=x.dtype,
            )
            return norm_x, (seq_m, seq_s), (dummy, dummy)

        mov_m, mov_s = self.mlp(seq_m, seq_s, x)

        if self.j > 0 and self.dwt is not None:
            ac, dc_list = self.dwt(x)          # ac: (B, C, T/2), dc_list: [(B,C,T/2)]
            norm_ac, (mac, sac) = self.norm_sliding(ac, kernel=self.hkernel)
            norm_dc_list, m_list, s_list = [], [], []
            for dc in dc_list:
                norm_dc, (mdc, sdc) = self.norm_sliding(dc, kernel=self.hkernel)
                norm_dc_list.append(norm_dc)
                m_list.append(mdc)
                s_list.append(sdc)

            # Merge bands and predict
            ac_m_merged = self.dwt([mac, m_list], 1)   # (B, C, T)
            ac_s_merged = self.dwt([sac, s_list], 1)
            ac_x_merged = self.dwt([ac, dc_list], 1)
            pred_m, pred_s = self.mlp(ac_m_merged, ac_s_merged, ac_x_merged)

            dwt_r = self.dwt_ratio            # (1, C, 1)
            mov_r = 1.0 - dwt_r
            norm_x = norm_x * mov_r + self.dwt([norm_ac, norm_dc_list], 1) * dwt_r
            pred_m = mov_m * mov_r + pred_m * dwt_r
            pred_s = mov_s * mov_r + pred_s * dwt_r
        else:
            pred_m, pred_s = mov_m, mov_s

        return norm_x, (seq_m, seq_s), (pred_m, pred_s)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def normalize(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple, Tuple]:
        """Normalise input sequence.

        Args:
            x: (B, T, C)

        Returns:
            norm_x:  (B, T, C) normalised input.
            pred_ms: (pred_m, pred_s) each (B, C, pred_len).
            seq_ms:  (seq_m,  seq_s)  each (B, C, seq_len).
        """
        x_BCT = x.transpose(-1, -2)                  # (B, C, T)
        norm_BCT, seq_ms, pred_ms = self.norm(x_BCT)
        return norm_BCT.transpose(-1, -2), pred_ms, seq_ms

    def de_normalize(
        self,
        y: torch.Tensor,
        station_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Phase-compensation denormalisation.

        Exactly matches official APN.de_normalize.

        Args:
            y:            (B, pred_len, C) model output (normalised domain).
            station_pred: (B, pred_len, 2*C) — cat([pred_m, pred_s], channel).
                          Can be pre-built with build_station_pred_tensor().

        Returns:
            (B, pred_len, C)
        """
        bs, L, dim = y.shape
        half = station_pred.shape[-1] // 2
        mean  = station_pred[..., :half]   # (B, pred_len, C)
        phase = station_pred[..., half:]   # (B, pred_len, C)  ← pred_phase or seq_s_y + pred_phase

        B1, T1, N1 = y.size()
        groups = B1 * N1

        # Reshape y to [1, B*C, T]
        y_flat = y.permute(0, 2, 1).reshape(1, groups, T1)

        # Build phase-compensation kernel via IFFT
        # phase: (B, T, C) → transpose → (B, C, T)
        kernel_raw = torch.fft.ifft(
            torch.exp(1j * phase.transpose(-1, -2)), dim=-1
        ).transpose(-1, -2)              # (B, T, C) complex
        kernel = (
            torch.abs(kernel_raw)
            .permute(0, 2, 1)
            .reshape(groups, 1, T1)
        )
        kernel = torch.flip(kernel, dims=[2])   # flip for correlation = convolution

        out = F.conv1d(
            y_flat, kernel, groups=groups, padding=T1 // 2
        )[:, :, : self.pred_len]               # [1, groups, pred_len]

        output = out.view(B1, N1, T1).permute(0, 2, 1) + mean  # (B, pred_len, C)
        return output.reshape(bs, L, dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def build_station_pred_tensor(
        self,
        pred_m: torch.Tensor,
        pred_s: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate pred_m and pred_s into (B, pred_len, 2C) tensor.

        pred_m, pred_s: (B, C, pred_len) channel-first → output (B, pred_len, 2C).
        """
        return torch.cat([pred_m, pred_s], dim=1).transpose(-1, -2)

    def norm_sliding_for_loss(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Wrapper: (B, T, C) → norm_sliding in (B, C, T) domain.

        Returns:
            x_center: (B, T, C)
            (m, phi_f): each (B, C, T)
        """
        x_BCT = x.transpose(-1, -2)
        xc_BCT, (m, phi_f) = self.norm_sliding(x_BCT)
        return xc_BCT.transpose(-1, -2), (m, phi_f)
