"""Name and write LibriSpeech evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path


TEST_SPLITS = ("test-clean", "test-other")


def split_result_path(all_path: Path, split: str) -> Path:
    if split not in TEST_SPLITS:
        raise ValueError(f"Unknown LibriSpeech split: {split}")
    stem = all_path.stem
    prefix = stem[: -len("test-all")] if stem.endswith("test-all") else f"{stem}_"
    return all_path.with_name(f"{prefix}{split}{all_path.suffix}")


def save_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def save_metric_results(path: Path, result: dict) -> None:
    """Save a combined result and each represented LibriSpeech test split."""
    save_json(path, result)
    for split in TEST_SPLITS:
        subset = [row for row in result["per_file"] if row["split"] == split]
        if not subset:
            continue
        split_metrics = result["split_metrics"][split]
        save_json(split_result_path(path, split), {
            **result,
            "split": split,
            "num_files": len(subset),
            "metrics": split_metrics,
            "split_metrics": {split: split_metrics},
            "per_file": subset,
        })
