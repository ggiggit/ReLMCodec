"""Score WER and speaker similarity on saved LibriSpeech reconstructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate import SpeakerEncoder, WhisperTranscriber, summarize, transcript_map
from relmcodec.audio import load_audio
from relmcodec.data import read_filelist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filelist", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--whisper", help="Whisper model name or local .pt path")
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--wavlm-sv", type=Path)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.whisper and not args.wavlm_sv:
        parser.error("Provide --whisper and/or --wavlm-sv")
    if args.whisper and not args.transcripts:
        parser.error("--whisper requires --transcripts")

    paths = read_filelist(args.filelist)
    if args.max_files:
        paths = paths[: args.max_files]
    transcripts = transcript_map(args.transcripts)
    if args.whisper and any(Path(path).stem not in transcripts for path in paths):
        parser.error("Some manifest utterances have no transcript")
    device = torch.device(args.device)
    whisper = WhisperTranscriber(args.whisper, device) if args.whisper else None
    speaker = SpeakerEncoder(args.wavlm_sv, device) if args.wavlm_sv else None

    rows = []
    for source in tqdm(paths, desc="Scoring reconstructions"):
        reference_path = Path(source)
        split = "test-clean" if "test-clean" in reference_path.parts else "test-other" if "test-other" in reference_path.parts else "other"
        output_path = args.audio_dir / split / f"{reference_path.stem}.wav"
        generated, rate = sf.read(output_path, dtype="float32", always_2d=True)
        if rate != 16000:
            raise ValueError(f"Expected 16 kHz: {output_path}")
        generated = generated.mean(axis=1)
        row = {"file": str(reference_path), "split": split, "output": str(output_path)}
        if speaker:
            reference = load_audio(reference_path, 16000).numpy()
            length = min(len(reference), len(generated))
            a = speaker(reference[:length])
            b = speaker(generated[:length])
            row["sim"] = float(F.cosine_similarity(a[None], b[None]).item())
        if whisper:
            from jiwer import wer

            text = whisper.normalizer(transcripts[reference_path.stem])
            row["wer"] = float(wer(text, whisper(generated)))
        rows.append(row)

    names = ("wer", "sim")
    result = {
        "filelist": str(args.filelist),
        "audio_dir": str(args.audio_dir),
        "num_files": len(rows),
        "metrics": {name: summarize(rows, name) for name in names if summarize(rows, name) is not None},
        "split_metrics": {
            split: {name: summarize(subset, name) for name in names if summarize(subset, name) is not None}
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
