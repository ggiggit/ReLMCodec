"""Audio manifest reader for reconstruction evaluation."""

from __future__ import annotations

from pathlib import Path


def read_filelist(path: str | Path) -> list[str]:
    paths = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = line
        if not Path(candidate).is_file():
            fields = line.split(maxsplit=1)
            candidate = fields[1] if len(fields) == 2 and Path(fields[1]).suffix else fields[0]
        if not Path(candidate).is_file():
            raise FileNotFoundError(f"Audio not found: {candidate}")
        paths.append(candidate)
    if not paths:
        raise ValueError(f"No audio paths found in {path}")
    return paths
