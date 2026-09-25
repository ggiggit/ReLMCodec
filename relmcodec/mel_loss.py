"""Seven-resolution log-Mel reconstruction loss used by ReLMCodec."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class MultiResolutionMelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        mel_bins = [5, 10, 20, 40, 80, 160, 320]
        window_lengths = [32, 64, 128, 256, 512, 1024, 2048]
        self.transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=window,
                    hop_length=window // 4,
                    n_mels=bins,
                    power=1.0,
                    center=True,
                    norm="slaney",
                    mel_scale="slaney",
                )
                for bins, window in zip(mel_bins, window_lengths)
            ]
        )

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if generated.ndim == 3:
            generated = generated.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)
        length = min(generated.size(-1), target.size(-1))
        generated, target = generated[..., :length], target[..., :length]
        loss = generated.new_zeros(())
        for transform in self.transforms:
            generated_mel = transform(generated).clamp_min(1e-5).log10()
            target_mel = transform(target).clamp_min(1e-5).log10()
            loss = loss + F.l1_loss(generated_mel, target_mel)
        return loss
