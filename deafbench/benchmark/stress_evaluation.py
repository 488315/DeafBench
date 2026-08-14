"""Paired model execution and scoring for prepared stress audio."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, cast

from deafbench import __version__
from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.stress_metrics import summarize_stress_results
from deafbench.benchmark.workspace import (
    atomic_write_json,
    atomic_write_jsonl,
    load_reference_records,
)
from deafbench.parser import parse_jsonl


ModelRunner = Callable[[Path, Path, Path], ModelRunInfo]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_model(left: ModelRunInfo, right: ModelRunInfo) -> bool:
    return (
        left.name,
        left.model_id,
        left.revision,
        left.decoding,
    ) == (
        right.name,
        right.model_id,
        right.revision,
        right.decoding,
    )


def run_stress_evaluation(
    prepared: Path,
    references: Path,
    destination: Path,
    model_runner: ModelRunner,
) -> Mapping[str, Any]:
    """Run one model over paired audio and atomically promote its evidence."""
    if destination.exists():
        raise ValueError("Stress evaluation destination already exists")
    try:
        preparation = json.loads(
            (prepared / "preparation-manifest.json").read_text(encoding="utf-8")
        )
        sample_ids = [sample["id"] for sample in preparation["samples"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("Stress preparation manifest is invalid") from exc
    if (
        preparation.get("lane") != "accessibility-stress-v1"
        or not sample_ids
        or len(sample_ids) != len(set(sample_ids))
        or preparation.get("sample_count") != len(sample_ids)
    ):
        raise ValueError("Stress preparation manifest is invalid")

    references_by_id = {
        cast(str, record["id"]): record
        for record in load_reference_records(references)
    }
    if not set(sample_ids) <= set(references_by_id):
        raise ValueError("Stress preparation references unknown samples")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-evaluate-", dir=destination.parent)
    )
    promoted = False
    try:
        subset_references = staging / "references.jsonl"
        atomic_write_jsonl(
            subset_references,
            [references_by_id[sample_id] for sample_id in sample_ids],
        )
        clean_predictions = staging / "clean-predictions.jsonl"
        stressed_predictions = staging / "stressed-predictions.jsonl"
        clean_model = model_runner(
            prepared / "clean", subset_references, clean_predictions
        )
        stressed_model = model_runner(
            prepared / "stressed", subset_references, stressed_predictions
        )
        if not _same_model(clean_model, stressed_model):
            raise ValueError("Stress lanes used different model configurations")

        reference_records = parse_jsonl(str(subset_references))
        summary = summarize_stress_results(
            reference_records,
            parse_jsonl(str(clean_predictions)),
            parse_jsonl(str(stressed_predictions)),
        )
        result: Mapping[str, Any] = {
            "schema_version": 1,
            "lane": "accessibility-stress-v1",
            "result_kind": "local_stress_observation",
            "evaluator_version": __version__,
            "model": {
                "name": clean_model.name,
                "model_id": clean_model.model_id,
                "revision": clean_model.revision,
                "decoding": dict(clean_model.decoding or {}),
            },
            "performance": {
                "clean": dict(clean_model.performance or {}),
                "stressed": dict(stressed_model.performance or {}),
            },
            "sample_count": len(sample_ids),
            "references_sha256": _sha256(subset_references),
            "preparation_manifest_sha256": _sha256(
                prepared / "preparation-manifest.json"
            ),
            "summary": summary,
        }
        atomic_write_json(staging / "result.json", result)
        os.replace(staging, destination)
        promoted = True
        return result
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
