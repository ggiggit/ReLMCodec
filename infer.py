#!/usr/bin/env python3
"""Encode, decode, or reconstruct audio with a release checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from relmcodec.audio import load_audio, save_audio
from relmcodec.config import load_config
from relmcodec.model import ReLMCodec


def main():
    parser = argparse.ArgumentParser(description="Encode, decode, or reconstruct 16 kHz speech")
    parser.add_argument("command", choices=("reconstruct", "encode", "decode"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--w2vbert")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.w2vbert:
        cfg.model.w2vbert_path = args.w2vbert
    device = torch.device(args.device)
    model = ReLMCodec.from_checkpoint(
        args.checkpoint, cfg.model, cfg.quantizer,
        map_location="cpu",
    ).to(device).eval()

    if args.command == "decode":
        payload = torch.load(args.input, map_location=device, weights_only=True)
        indices = payload["indices"] if isinstance(payload, dict) else payload
        with torch.inference_mode():
            waveform = model.decode(indices.to(device))[0]
        save_audio(args.output, waveform, cfg.model.sample_rate)
        return

    audio = load_audio(args.input, cfg.model.sample_rate).unsqueeze(0).to(device)
    with torch.inference_mode():
        indices = model.encode(audio)
        if args.command == "reconstruct":
            waveform = model.decode(indices)[0]
            save_audio(args.output, waveform, cfg.model.sample_rate)
        else:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"indices": indices.cpu(), "sample_rate": cfg.model.sample_rate,
                 "hop_length": cfg.model.hop_length, "source_samples": audio.shape[-1]},
                output,
            )


if __name__ == "__main__":
    main()
