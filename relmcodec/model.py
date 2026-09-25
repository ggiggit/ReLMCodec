from __future__ import annotations

import torch
from torch import nn

from .checkpoint import load_checkpoint, load_model_state, remap_lightning_state
from .config import ModelConfig, QuantizerConfig
from .decoder import ISTFTHead, VocosBackbone, VocosLightBackbone
from .encoder import W2vBertEncoder, XCodec2AcousticRetrainEncoder
from .quantizer import VectorQuantizerV2

ReLMCodecConfig = ModelConfig


def papa_fuse(semantic: torch.Tensor, residual: torch.Tensor, alpha) -> torch.Tensor:
    if semantic.shape != residual.shape:
        raise ValueError(f"PAPA operands must have equal shapes: {semantic.shape} vs {residual.shape}")
    return semantic + alpha * residual


class ReLMCodec(nn.Module):
    """ReLMCodec with Pre-quantization Anchor-Preserving Adaptation (PAPA)."""

    def __init__(
        self,
        config: ModelConfig,
        quantizer_config: QuantizerConfig | None = None,
        ssl_encoder: nn.Module | None = None,
        acoustic_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        qcfg = quantizer_config or QuantizerConfig()
        self.ssl_encoder = ssl_encoder or W2vBertEncoder(config.w2vbert_path, config.ssl_layer)
        self.acoustic_encoder = acoustic_encoder or XCodec2AcousticRetrainEncoder()
        self.papa_adapter = VocosLightBackbone(
            input_channels=2048,
            dim=config.preencoder_dim,
            intermediate_dim=config.preencoder_intermediate_dim,
            num_layers=config.preencoder_layers,
            output_dim=1024,
        )
        self.register_buffer("alpha", torch.tensor(float(config.alpha)), persistent=True)
        self.quantizer = VectorQuantizerV2(
            dim=1024,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            num_quantizers=1,
            decay=qcfg.decay,
            kmeans_init=qcfg.kmeans_init,
            kmeans_iters=qcfg.kmeans_iters,
            threshold_ema_dead_code=qcfg.threshold_ema_dead_code,
            threshold_ema_dead_code_end=qcfg.threshold_ema_dead_code_end,
            threshold_decay_steps=qcfg.threshold_decay_steps,
            commitment_weight=qcfg.commitment_weight,
            data_pool_size=qcfg.data_pool_size,
            enable_expire=qcfg.enable_expire,
            l2_normalize=qcfg.l2_normalize,
        )
        self.decoder = VocosBackbone(
            input_channels=1024,
            dim=config.decoder_dim,
            intermediate_dim=config.decoder_intermediate_dim,
            num_layers=config.decoder_layers,
        )
        self.waveform_head = ISTFTHead(
            dim=config.decoder_dim, n_fft=config.n_fft, hop_length=config.hop_length
        )

    def pre_quantize(self, audio: torch.Tensor, return_components: bool = False):
        semantic = self.ssl_encoder(audio)
        acoustic = self.acoustic_encoder(audio)
        length = min(semantic.size(1), acoustic.size(1))
        semantic = semantic[:, :length]
        acoustic = acoustic[:, :length]
        residual = self.papa_adapter(
            torch.cat([semantic, acoustic], dim=-1).transpose(1, 2)
        )
        representation = papa_fuse(semantic, residual, self.alpha)
        if return_components:
            return representation, semantic, acoustic, residual
        return representation

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self.quantizer.encode(self.pre_quantize(audio)).squeeze(0)

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
        if indices.ndim == 2:
            indices = indices.unsqueeze(0)
        quantized = self.quantizer.decode(indices)
        return self.waveform_head(self.decoder(quantized.transpose(1, 2)))

    def forward(self, audio: torch.Tensor, return_features: bool = False):
        z = self.pre_quantize(audio)
        quantized, indices, commitment = self.quantizer(z)
        waveform = self.waveform_head(self.decoder(quantized.transpose(1, 2)))
        output = {
            "waveform": waveform,
            "indices": indices.squeeze(0),
            "commitment_loss": commitment,
        }
        if return_features:
            output.update(pre_quantization=z, quantized=quantized)
        return output

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        config: ModelConfig,
        quantizer_config: QuantizerConfig | None = None,
        map_location="cpu",
        source_format: str = "release",
    ):
        model = cls(config, quantizer_config)
        checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if source_format == "lightning" or ("state_dict" in checkpoint and "model" not in checkpoint):
            state = remap_lightning_state(state)
            target_keys = set(model.state_dict())
            extras = set(state) - target_keys
            normalizer_keys = {"normalizer.mean", "normalizer.std"}
            if extras - normalizer_keys:
                raise RuntimeError(f"Unexpected checkpoint keys: {sorted(extras - normalizer_keys)}")
            if extras:
                if extras != normalizer_keys or not torch.all(state["normalizer.mean"] == 0) or not torch.all(state["normalizer.std"] == 1):
                    raise RuntimeError("Checkpoint uses a non-identity feature normalizer")
            state = {key: value for key, value in state.items() if key in target_keys}
        load_model_state(model, state, strict=True)
        return model
