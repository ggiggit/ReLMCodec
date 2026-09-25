"""Verify a bundled checkpoint with a one-second encode/decode round trip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relmcodec.config import load_config
from relmcodec.model import ReLMCodec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("8k", "64k"), default="64k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / f"relmcodec_{args.variant}.yaml")
    checkpoint = root / "models" / f"relmcodec_{args.variant}.pt"
    model = ReLMCodec.from_checkpoint(checkpoint, config.model, config.quantizer).to(args.device).eval()
    audio = 0.05 * torch.randn(1, 16000, device=args.device, generator=torch.Generator(device=args.device).manual_seed(1234))
    with torch.inference_mode():
        indices = model.encode(audio)
        reconstructed = model.decode(indices)
        forward = model(audio)
    assert indices.ndim == 2 and 45 <= indices.shape[-1] <= 50
    assert 0 <= int(indices.min()) and int(indices.max()) < config.model.codebook_size
    assert reconstructed.ndim == 2 and bool(torch.isfinite(reconstructed).all())
    assert torch.equal(indices, forward["indices"])
    result = {
        "variant": args.variant,
        "checkpoint": str(checkpoint),
        "frames": int(indices.shape[-1]),
        "output_samples": int(reconstructed.shape[-1]),
        "code_min": int(indices.min()),
        "code_max": int(indices.max()),
        "finite": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
