import torch
from torch import nn

from .spectral_ops import ISTFT


class ISTFTHead(nn.Module):
    """Predict complex STFT coefficients and reconstruct the waveform."""

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        magnitude, phase = x.chunk(2, dim=1)
        magnitude = torch.exp(magnitude.clamp(max=4.6))
        spectrum = magnitude * (torch.cos(phase) + 1j * torch.sin(phase))
        return self.istft(spectrum)
