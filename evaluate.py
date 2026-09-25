#!/usr/bin/env python3
"""Reconstruct a manifest and score the paper's objective speech metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from relmcodec.audio import load_audio, save_audio
from relmcodec.config import load_config
from relmcodec.data import read_filelist
from relmcodec.mel_loss import MultiResolutionMelSpectrogramLoss
from relmcodec.model import ReLMCodec


class WhisperTranscriber:
    """Match the original English Whisper-Large-v3 WER protocol."""

    def __init__(self, model_name: str, device: torch.device):
        import whisper
        from whisper.normalizers import EnglishTextNormalizer

        self.whisper = whisper
        self.model = whisper.load_model(model_name, device=str(device))
        self.options = whisper.DecodingOptions(language="en", without_timestamps=True)
        self.normalizer = EnglishTextNormalizer()
        self.device = device

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray) -> str:
        audio = self.whisper.pad_or_trim(torch.from_numpy(waveform).float())
        mel = self.whisper.log_mel_spectrogram(audio, self.model.dims.n_mels).to(self.device)
        return self.normalizer(self.model.decode(mel.unsqueeze(0), self.options)[0].text)


class SpeakerEncoder:
    """Official WavLM-Large-SV and ECAPA-TDNN speaker embedding wrapper."""

    def __init__(self, model_dir: Path, device: torch.device):
        checkpoint = model_dir / "wavlm_large_finetune.pth"
        if not checkpoint.is_file() or not (model_dir / "verification.py").is_file():
            raise FileNotFoundError(f"Expected verification.py and wavlm_large_finetune.pth in {model_dir}")
        sys.path.insert(0, str(model_dir.resolve()))
        try:
            from verification import init_model
        finally:
            sys.path.pop(0)
        self.model = init_model("wavlm_large", checkpoint=str(checkpoint)).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(waveform, device=self.device).float().unsqueeze(0)
        return self.model(tensor)[0]


def transcript_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(" ")
        result[key] = value
    return result


def summarize(rows: list[dict], metric: str) -> dict | None:
    values = [row[metric] for row in rows if metric in row and np.isfinite(row[metric])]
    if not values:
        return None
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--filelist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=0, help="0 evaluates the complete manifest")
    parser.add_argument("--whisper", help="OpenAI Whisper model name or local .pt path; enables WER")
    parser.add_argument("--transcripts", type=Path, help="LibriSpeech 'utterance-id TEXT' lines")
    parser.add_argument("--wavlm-sv", type=Path, help="Official UniSpeech speaker-verification directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.whisper and not args.transcripts:
        parser.error("--whisper requires --transcripts")

    paths = read_filelist(args.filelist)
    if args.max_files:
        paths = paths[: args.max_files]
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = ReLMCodec.from_checkpoint(args.checkpoint, cfg.model, cfg.quantizer).to(device).eval()
    mel = MultiResolutionMelSpectrogramLoss(cfg.model.sample_rate).to(device).eval()
    from pesq import pesq
    from pystoi import stoi

    whisper = WhisperTranscriber(args.whisper, device) if args.whisper else None
    speaker = SpeakerEncoder(args.wavlm_sv, device) if args.wavlm_sv else None
    transcripts = transcript_map(args.transcripts)
    if whisper and any(Path(path).stem not in transcripts for path in paths):
        raise ValueError("Some manifest utterances have no transcript")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path_string in tqdm(paths, desc="Reconstructing"):
        path = Path(path_string)
        reference = load_audio(path, cfg.model.sample_rate)
        with torch.inference_mode():
            generated = model(reference.unsqueeze(0).to(device))["waveform"][0]
        length = min(reference.numel(), generated.numel())
        reference = reference[:length]
        generated = generated[:length]
        generated_np = generated.detach().cpu().float().numpy()
        reference_np = reference.numpy()
        split = "test-clean" if "test-clean" in path.parts else "test-other" if "test-other" in path.parts else "other"
        output_path = args.output_dir / split / f"{path.stem}.wav"
        save_audio(output_path, generated, cfg.model.sample_rate)
        row = {"file": str(path), "split": split, "output": str(output_path), "source_samples": int(reference.numel())}
        with torch.inference_mode():
            row["mel"] = float(mel(generated[None], reference.to(device)[None]))
        try:
            row["pesq"] = float(pesq(cfg.model.sample_rate, reference_np, generated_np, "wb"))
        except Exception as error:
            row["pesq_error"] = str(error)
        try:
            row["stoi"] = float(stoi(reference_np, generated_np, cfg.model.sample_rate, extended=False))
        except Exception as error:
            row["stoi_error"] = str(error)
        if speaker:
            ref_embedding = speaker(reference_np)
            gen_embedding = speaker(generated_np)
            row["sim"] = float(F.cosine_similarity(ref_embedding[None], gen_embedding[None]))
        if whisper:
            from jiwer import wer

            reference_text = whisper.normalizer(transcripts[path.stem])
            row["wer"] = float(wer(reference_text, whisper(generated_np)))
        rows.append(row)

    metrics = ("mel", "pesq", "stoi", "sim", "wer")
    result = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "filelist": str(args.filelist),
        "num_files": len(rows),
        "metrics": {name: summarize(rows, name) for name in metrics if summarize(rows, name) is not None},
        "split_metrics": {
            split: {name: summarize(subset, name) for name in metrics if summarize(subset, name) is not None}
            for split in ("test-clean", "test-other")
            if (subset := [row for row in rows if row["split"] == split])
        },
        "per_file": rows,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_file"}, indent=2))


if __name__ == "__main__":
    main()
