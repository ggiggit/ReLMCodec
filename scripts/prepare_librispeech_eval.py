"""Build deterministic reconstruction manifests from LibriSpeech test splits."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing test-clean and test-other")
    parser.add_argument("--output-dir", type=Path, default=Path("data/filelists"))
    parser.add_argument("--limit-per-split", type=int, default=0, help="0 means all utterances")
    args = parser.parse_args()

    root = args.root.resolve()
    if (root / "LibriSpeech").is_dir():
        root = root / "LibriSpeech"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    transcripts = {}
    for split in ("test-clean", "test-other"):
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(split_root)
        paths = sorted(split_root.rglob("*.flac"))
        if args.limit_per_split:
            paths = paths[: args.limit_per_split]
        if not paths:
            raise RuntimeError(f"No FLAC files under {split_root}")
        lines = [str(path.resolve()) for path in paths]
        (args.output_dir / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        combined.extend(lines)
        for transcript_file in sorted(split_root.rglob("*.trans.txt")):
            for line in transcript_file.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition(" ")
                transcripts[key] = value
        print(f"{split}: {len(paths)} utterances")

    missing = [Path(path).stem for path in combined if Path(path).stem not in transcripts]
    if missing:
        raise RuntimeError(f"Missing transcripts for {len(missing)} files, e.g. {missing[:3]}")
    (args.output_dir / "test-all.txt").write_text("\n".join(combined) + "\n", encoding="utf-8")
    (args.output_dir / "transcripts.txt").write_text(
        "\n".join(f"{Path(path).stem} {transcripts[Path(path).stem]}" for path in combined) + "\n",
        encoding="utf-8",
    )
    print(f"test-all: {len(combined)} utterances; manifests in {args.output_dir}")


if __name__ == "__main__":
    main()
