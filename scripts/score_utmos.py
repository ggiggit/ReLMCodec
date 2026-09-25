"""Score saved reconstructions with the original UTMOS-demo Score API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from relmcodec.results import TEST_SPLITS, save_json, split_result_path


def save_utmos_results(path: Path, rows: list[dict]) -> dict:
    values = [row["utmos"] for row in rows]
    result = {"split": "test-all", "count": len(rows), "mean": float(np.mean(values)), "std": float(np.std(values)), "per_file": rows}
    save_json(path, result)
    for split in TEST_SPLITS:
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        split_values = [row["utmos"] for row in subset]
        save_json(split_result_path(path, split), {
            "split": split,
            "count": len(subset),
            "mean": float(np.mean(split_values)),
            "std": float(np.std(split_values)),
            "per_file": subset,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--utmos-dir", type=Path, required=True, help="Official UTMOS-demo directory")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    utmos_dir = args.utmos_dir.resolve()
    checkpoint = utmos_dir / "epoch=3-step=7459.ckpt"
    required = ("score.py", "epoch=3-step=7459.ckpt", "wav2vec_small.pt")
    missing = [name for name in required if not (utmos_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"UTMOS-demo is missing: {', '.join(missing)}")
    sys.path.insert(0, str(utmos_dir))
    from omegaconf import _utils as omegaconf_utils
    from score import Score

    if not hasattr(omegaconf_utils, "is_primitive_type"):
        # Fairseq's historical checkpoint converter calls this removed helper.
        omegaconf_utils.is_primitive_type = lambda _: True
    previous_dir = Path.cwd()
    previous_load_setting = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    try:
        os.chdir(utmos_dir)
        # Official Lightning/Fairseq checkpoints contain config objects, not just tensors.
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        scorer = Score(ckpt_path=str(checkpoint), input_sample_rate=16000, device=str(args.device))
    finally:
        os.chdir(previous_dir)
        if previous_load_setting is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_load_setting
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
        split = path.parent.name if path.parent.name in TEST_SPLITS else "other"
        rows.append({"file": str(path), "split": split, "utmos": float(np.asarray(score).reshape(-1)[0])})
    result = save_utmos_results(args.result_json, rows)
    print(json.dumps({key: value for key, value in result.items() if key != "per_file"}, indent=2))


if __name__ == "__main__":
    main()
