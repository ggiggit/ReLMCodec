"""Read inference-only ReLMCodec checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(path: str | Path, map_location="cpu"):
    # The published files contain tensors and plain metadata only. mmap avoids
    # holding a second 3 GB copy of the state dict in host memory.
    return torch.load(str(path), map_location=map_location, weights_only=True, mmap=True)


def load_model_state(model, state, strict: bool = True):
    return model.load_state_dict(state, strict=strict)


def remap_lightning_state(state):
    """Map a historical research checkpoint into the inference module names."""
    prefixes = {
        "encoder.ssl_encoder.": "ssl_encoder.",
        "encoder.acoustic_encoder.": "acoustic_encoder.",
        "pre_encoder.": "papa_adapter.",
        "backbone.": "decoder.",
        "head.": "waveform_head.",
    }
    remapped = {}
    for original_key, value in state.items():
        key = original_key.removeprefix("model.")
        if key == "pre_encoder_alpha":
            key = "alpha"
        for source, target in prefixes.items():
            if key.startswith(source):
                key = target + key[len(source):]
                break
        if key in remapped:
            raise ValueError(f"Duplicate remapped tensor: {key}")
        remapped[key] = value
    return remapped
