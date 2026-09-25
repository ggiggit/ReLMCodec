from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def load_audio(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio).mean(dim=1)
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio file: {path}")
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().float().cpu().squeeze().numpy(), sample_rate)


def match_waveform_lengths(reference: torch.Tensor, generated: torch.Tensor):
    length = min(reference.shape[-1], generated.shape[-1])
    return reference[..., :length], generated[..., :length]
