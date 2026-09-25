"""Trainable acoustic encoder extracted from the public XCodec2 implementation."""

from __future__ import annotations

import torch
from torch import nn

from ..vendor.xcodec2.vq.codec_encoder import CodecEncoder_Transformer


class XCodec2AcousticRetrainEncoder(nn.Module):
    """Randomly initialized XCodec2 convolutional encoder at 50 frames/s."""

    hidden_dim = 1024

    def __init__(self):
        super().__init__()
        self.codec_encoder = CodecEncoder_Transformer()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"Expected audio with shape (batch, samples), got {audio.shape}")
        return self.codec_encoder(audio.unsqueeze(1))
