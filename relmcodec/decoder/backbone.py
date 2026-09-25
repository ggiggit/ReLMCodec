"""Vocos backbones used by the PAPA predictor and waveform decoder."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .modules import AdaLayerNorm, ConvNeXtBlock


def _activation(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


class ResidualPositionBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = _group_norm(channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(_activation(self.norm1(x)))
        residual = self.conv2(self.dropout(_activation(self.norm2(residual))))
        return x + residual


class PositionAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _group_norm(channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        query = self.q(normalized).transpose(1, 2)
        key = self.k(normalized)
        value = self.v(normalized)
        weights = torch.bmm(query, key) * (key.size(1) ** -0.5)
        weights = torch.softmax(weights, dim=-1).transpose(1, 2)
        return x + self.proj_out(torch.bmm(value, weights))


class VocosBackbone(nn.Module):
    """Twelve-layer ConvNeXt decoder backbone with a positional front end."""

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, 7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = (
            AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
            if self.adanorm
            else nn.LayerNorm(dim, eps=1e-6)
        )
        scale = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=scale,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        # Match the released training initialization: initialize the ConvNeXt
        # backbone before attaching the positional front end.
        self.apply(self._init_weights)
        self.pos_net = nn.Sequential(
            ResidualPositionBlock(dim),
            ResidualPositionBlock(dim),
            PositionAttention(dim),
            ResidualPositionBlock(dim),
            ResidualPositionBlock(dim),
            _group_norm(dim),
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, bandwidth_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.pos_net(self.embed(x))
        if self.adanorm:
            if bandwidth_id is None:
                raise ValueError("bandwidth_id is required for adaptive normalization")
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for block in self.convnext:
            x = block(x, cond_embedding_id=bandwidth_id)
        return self.final_layer_norm(x.transpose(1, 2))


class VocosLightBackbone(nn.Module):
    """ConvNeXt residual predictor used by PAPA."""

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        output_dim: Optional[int] = None,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, 7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        scale = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.output_proj = nn.Linear(dim, output_dim) if output_dim is not None else None
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        x = self.final_layer_norm(x.transpose(1, 2))
        return self.output_proj(x) if self.output_proj is not None else x
