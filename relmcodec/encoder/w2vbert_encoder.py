"""Frozen W2v-BERT 2.0 main-path encoder."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel


class W2vBertEncoder(nn.Module):
    """Extract one frozen W2v-BERT 2.0 hidden layer at 50 Hz."""

    def __init__(self, model_name: str = "models/w2v-bert-2.0", layer_idx: int = 17):
        super().__init__()
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("layer_idx must identify one non-negative hidden layer")
        # The bundled codec checkpoint contains the complete frozen SSL encoder.
        # Construct its architecture from the small local config; strict checkpoint
        # loading below supplies every tensor, with no network or model cache needed.
        model_config = Wav2Vec2BertConfig.from_pretrained(model_name, local_files_only=True)
        self.model = Wav2Vec2BertModel(model_config)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)
        if layer_idx > self.model.config.num_hidden_layers:
            raise ValueError(
                f"layer_idx={layer_idx} exceeds {self.model.config.num_hidden_layers} layers"
            )
        self.layer_idx = layer_idx
        self.hidden_dim = self.model.config.hidden_size
        self.sample_rate = 16000
        self.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        audio_list = [sample.detach().cpu().numpy() for sample in audio]
        inputs = self.feature_extractor(
            audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            do_normalize_per_mel_bins=True,
        )
        input_features = inputs.input_features.to(audio.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(audio.device)
        with torch.no_grad():
            outputs = self.model(
                input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[self.layer_idx]

    @staticmethod
    def get_output_length(input_length: int) -> int:
        return input_length // 320
