import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.center = center
        self.normalized = normalized
        self.onesided = onesided

        if window == "hann":
            window_tensor = torch.hann_window(self.win_length)
        else:
            raise ValueError(f"Unsupported window type: {window}")

        self.register_buffer("window", window_tensor, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        batch_size, length, channels = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * channels, length)
        xf = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        freq_bins, time_bins = xf.shape[-2], xf.shape[-1]
        return xf.reshape(batch_size, channels, freq_bins, time_bins)

    def inverse(self, xf: torch.Tensor, length: int | None = None) -> torch.Tensor:
        # xf: (B, C, F, TT)
        batch_size, channels, freq_bins, time_bins = xf.shape
        xf = xf.reshape(batch_size * channels, freq_bins, time_bins)
        x = torch.istft(
            xf,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=length,
        )
        return x.reshape(batch_size, channels, -1).permute(0, 2, 1)

    def time_bins(self, length: int) -> int:
        pad = self.n_fft // 2 if self.center else 0
        length = length + 2 * pad
        return int((length - self.n_fft) // self.hop_length + 1)
