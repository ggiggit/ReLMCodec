"""Small, strict configuration schema for published inference checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    w2vbert_path: str = "models/w2v-bert-2.0"
    ssl_layer: int = 17
    codebook_size: int = 8192
    codebook_dim: int = 8
    alpha: float = 0.1
    sample_rate: int = 16000
    hop_length: int = 320
    preencoder_dim: int = 384
    preencoder_intermediate_dim: int = 2048
    preencoder_layers: int = 12
    decoder_dim: int = 1024
    decoder_intermediate_dim: int = 4096
    decoder_layers: int = 12
    n_fft: int = 1280


@dataclass
class QuantizerConfig:
    decay: float = 0.99
    l2_normalize: bool = False
    kmeans_init: bool = True
    kmeans_iters: int = 200
    commitment_weight: float = 1.0
    threshold_ema_dead_code: float = 0.1
    threshold_ema_dead_code_end: float = 0.0001
    threshold_decay_steps: int = -1
    data_pool_size: int = 524288
    enable_expire: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)

    def validate(self) -> None:
        if self.model.codebook_size not in (8192, 65536):
            raise ValueError("codebook_size must be 8192 or 65536")
        if self.model.codebook_dim != 8 or self.model.sample_rate != 16000:
            raise ValueError("Published weights require eight-dimensional codes and 16 kHz audio")
        if self.model.hop_length != 320 or self.model.ssl_layer != 17:
            raise ValueError("Published weights require hop_length=320 and W2v-BERT layer 17")
        if self.quantizer.l2_normalize:
            raise ValueError("Published checkpoints use Euclidean VQ assignment; set l2_normalize: false")

    def to_dict(self):
        return asdict(self)


def _merge(cls, values):
    known = set(cls.__dataclass_fields__)
    extra = set(values) - known
    if extra:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(extra)}")
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = raw or {}
    extra = set(raw) - {"model", "quantizer"}
    if extra:
        raise ValueError(f"Unknown inference config sections: {sorted(extra)}")
    config = ExperimentConfig(
        model=_merge(ModelConfig, raw.get("model", {})),
        quantizer=_merge(QuantizerConfig, raw.get("quantizer", {})),
    )
    local_model = Path(__file__).resolve().parents[1] / config.model.w2vbert_path
    if not Path(config.model.w2vbert_path).is_absolute() and local_model.is_dir():
        config.model.w2vbert_path = str(local_model)
    config.validate()
    return config
