"""Download and verify the published inference checkpoints."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {
    "8k": ("relmcodec_8k.pt", "b55ee1aa176a7a9b3e00b952a07f4932c1770103e6e6e4e4bd91e0d6345d3335"),
    "64k": ("relmcodec_64k.pt", "951abb8e1b8af372c963b9ca360c3246ee9ed255a6c0f251018fe3223846231a"),
    "8k-stage1": ("relmcodec_8k_stage1.pt", "d662a733f87b8e51eb7f8e3ddecd831112f4694dec6dddb687e5ff4a5ea223fc"),
    "64k-stage1": ("relmcodec_64k_stage1.pt", "9b351388e91fe60622d197271d886060546b3c78bd53ef6c05740f5db7902e03"),
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
    parser.add_argument("--variant", choices=(*WEIGHTS, "all"), default="all",
                        help="all downloads the two primary, perceptually trained checkpoints")
    args = parser.parse_args()

    variants = ("8k", "64k") if args.variant == "all" else (args.variant,)
    for variant in variants:
        filename, expected = WEIGHTS[variant]
        destination = ROOT / "models" / filename
        if destination.is_file() and sha256(destination) == expected:
            print(f"{variant}: verified existing {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(f"{variant}: local file failed verification; downloading a fresh copy")
        downloaded = fetch(args.source, f"models/{filename}")
        if downloaded.resolve() != destination.resolve():
            downloaded.replace(destination)
        actual = sha256(destination)
        if actual != expected:
            raise RuntimeError(f"{variant}: downloaded file failed verification")
        print(f"{variant}: verified {destination}")


if __name__ == "__main__":
    main()
