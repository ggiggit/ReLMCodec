#!/usr/bin/env python3
"""Reconstruct a manifest and score the paper's objective speech metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

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
    """Transcribe complete English utterances with the official Whisper model."""

    def __init__(self, model_name: str, device: torch.device):
        import whisper
        from whisper.normalizers import EnglishTextNormalizer

        self.model = whisper.load_model(model_name, device=str(device))
        self.normalizer = EnglishTextNormalizer()
        self.device = device

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray) -> str:
        result = self.model.transcribe(
            waveform.astype(np.float32, copy=False),
            language="en",
            task="transcribe",
            fp16=self.device.type == "cuda",
            verbose=None,
        )
        return self.normalizer(result["text"])


class SpeakerEncoder:
    """Official WavLM-Large-SV and ECAPA-TDNN speaker embedding wrapper."""

    def __init__(self, model_dir: Path, device: torch.device):
        checkpoint = model_dir / "wavlm_large_finetune.pth"
        if not checkpoint.is_file() or not (model_dir / "models" / "ecapa_tdnn.py").is_file():
            raise FileNotFoundError(f"Expected models/ecapa_tdnn.py and wavlm_large_finetune.pth in {model_dir}")
        # The s3prl revision used by UniSpeech imports this legacy torchaudio helper.
        import torchaudio.functional as audio_functional

        if not hasattr(audio_functional, "magphase"):
            audio_functional.magphase = lambda x, power=1.0: (x.abs().pow(power), torch.angle(x))
        sys.path.insert(0, str(model_dir.resolve()))
        try:
            from models.ecapa_tdnn import ECAPA_TDNN_SMALL
            from s3prl.upstream.wavlm.expert import UpstreamExpert
        finally:
            sys.path.pop(0)

        # The official SV file already contains all upstream WavLM parameters.
        # Build its s3prl expert locally; s3prl's old hub URL has expired.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        prefix = "feature_extract.model."
        upstream_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        config = {
            "extractor_mode": "layer_norm",
            "encoder_layers": 24,
            "encoder_embed_dim": 1024,
            "encoder_ffn_embed_dim": 4096,
            "encoder_attention_heads": 16,
            "layer_norm_first": True,
            "normalize": True,
            "relative_position_embedding": True,
            "num_buckets": 320,
            "max_distance": 800,
            "gru_rel_pos": True,
        }
        original_load = torch.load

        def load_inline(path, *args, **kwargs):
            if path == "__wavlm_inline__":
                return {"cfg": config, "model": upstream_state}
            return original_load(path, *args, **kwargs)

        with patch.object(torch, "load", side_effect=load_inline):
            upstream = UpstreamExpert("__wavlm_inline__")
        with patch.object(torch.hub, "load", return_value=upstream):
            self.model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")
        state.pop("loss_calculator.projection.weight", None)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(device).eval()
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
