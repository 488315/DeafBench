"""Workspace helpers for the installed DeafBench recorder."""

from __future__ import annotations

import importlib.resources as resources
from pathlib import Path


BUNDLED_DATA_PACKAGE = "deafbench.recorder.data"


def resolve_dataset_paths(repo_root: Path, dataset: str = "core-v1") -> tuple[Path, Path]:
    """Return reference and audio paths for a safe benchmark dataset name."""
    if not dataset or dataset in {".", ".."} or any(
        separator in dataset for separator in ("/", "\\", ":")
    ):
        raise ValueError("Invalid dataset name")
    dataset_dir = Path(repo_root) / "benchmarks" / dataset
    return dataset_dir / "references.jsonl", dataset_dir / "audio"


def ensure_dataset_workspace(repo_root: Path, dataset: str = "core-v1") -> tuple[Path, Path]:
    """Seed bundled references when a local benchmark workspace is missing them."""
    references_path, audio_dir = resolve_dataset_paths(repo_root, dataset)
    if references_path.is_file():
        return references_path, audio_dir

    resource = resources.files(BUNDLED_DATA_PACKAGE).joinpath(f"{dataset}.jsonl")
    if not resource.is_file():
        raise FileNotFoundError(
            f"No references found for dataset {dataset}. "
            "Use --references with an existing JSONL file."
        )

    references_path.parent.mkdir(parents=True, exist_ok=True)
    references_path.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")
    return references_path, audio_dir
