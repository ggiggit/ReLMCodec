"""Download and verify the published inference checkpoints."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {
    "8k": ("relmcodec_8k.pt", "b55ee1aa176a7a9b3e00b952a07f4932c1770103e6e6e4e4bd91e0d6345d3335"),
    "64k": ("relmcodec_64k.pt", "951abb8e1b8af372c963b9ca360c3246ee9ed255a6c0f251018fe3223846231a"),
}
REPOSITORIES = {
    "hf": "hf-wzx1205/ReLMCodec",
    "modelscope": "wanzixiang/ReLMCodec",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(source: str, relative_path: str) -> Path:
    if source == "hf":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("Install the hub client: pip install -e '.[hub-hf]'") from exc
        return Path(
            hf_hub_download(
                repo_id=REPOSITORIES[source],
                filename=relative_path,
                local_dir=ROOT,
                force_download=True,
            )
        )
    try:
        from modelscope_hub import HubApi
    except ImportError as exc:
        raise SystemExit("Install the hub client: pip install -e '.[hub-ms]'") from exc
    return Path(
        HubApi().download_file(
            REPOSITORIES[source], "model", relative_path, local_dir=ROOT, force=True
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=tuple(REPOSITORIES), default="hf")
    parser.add_argument("--variant", choices=("8k", "64k", "all"), default="all")
    args = parser.parse_args()

    variants = tuple(WEIGHTS) if args.variant == "all" else (args.variant,)
    for variant in variants:
        filename, expected = WEIGHTS[variant]
        destination = ROOT / "models" / filename
        if destination.is_file() and sha256(destination) == expected:
            print(f"{variant}: verified existing {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(f"{variant}: local file failed SHA-256; downloading a fresh copy")
        downloaded = fetch(args.source, f"models/{filename}")
        if downloaded.resolve() != destination.resolve():
            downloaded.replace(destination)
        actual = sha256(destination)
        if actual != expected:
            raise RuntimeError(f"{variant}: SHA-256 mismatch ({actual}); expected {expected}")
        print(f"{variant}: verified {destination} ({actual})")


if __name__ == "__main__":
    main()
