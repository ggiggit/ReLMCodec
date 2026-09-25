"""Score saved reconstructions with the original UTMOS-demo Score API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--utmos-dir", type=Path, required=True, help="Official UTMOS-demo directory")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = args.utmos_dir / "epoch=3-step=7459.ckpt"
    if not (args.utmos_dir / "score.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("UTMOS-demo must contain score.py and epoch=3-step=7459.ckpt")
    sys.path.insert(0, str(args.utmos_dir.resolve()))
    from score import Score
    scorer = Score(ckpt_path=str(checkpoint), input_sample_rate=16000, device=str(args.device))
    paths = sorted(args.audio_dir.rglob("*.wav"))
    if not paths:
        raise ValueError(f"No .wav files in {args.audio_dir}")
    rows = []
    for path in tqdm(paths, desc="UTMOS"):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz: {path}")
        tensor = torch.from_numpy(audio.mean(axis=1)).to(args.device)
        with torch.inference_mode():
            score = scorer.score(tensor)
        if torch.is_tensor(score):
            score = score.detach().cpu().numpy()
        rows.append({"file": str(path), "utmos": float(np.asarray(score).reshape(-1)[0])})
    values = [row["utmos"] for row in rows]
    result = {"count": len(rows), "mean": float(np.mean(values)), "std": float(np.std(values)), "per_file": rows}
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_file"}, indent=2))


if __name__ == "__main__":
    main()
