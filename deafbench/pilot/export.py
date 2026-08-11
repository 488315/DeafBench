"""Aggregate-only artifact export for the customer-run pilot."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from deafbench.pilot.export_scan import assert_export_safe
from deafbench.pilot.manifest import (
    EXECUTION_NOTICE,
    verify_signed_manifest,
    write_signed_manifest,
)


PILOT_MODEL_IDS = (
    "Qwen/Qwen3-ASR-1.7B-hf",
    "nvidia/parakeet-tdt-0.6b-v2",
    "ibm-granite/granite-speech-4.1-2b",
)
_SAFE_CONFIGURATION = frozenset(
    {
        "batch_size",
        "device",
        "dtype",
        "keyword_biasing",
        "language",
        "max_new_tokens",
        "num_beams",
        "timestamps",
        "trust_remote_code",
    }
)


@dataclass(frozen=True)
class CustomerExportResult:
    manifest_sha256: str
    model_count: int
    dataset_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("result manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError("result manifest must be an object")
    return value


def _registry_models(repo_root: Path) -> dict[str, dict[str, object]]:
    registry = _load_json(repo_root / "deafbench" / "model-registry.json")
    models = registry.get("models")
    if not isinstance(models, list):
        raise ValueError("model registry is invalid")
    return {
        str(model.get("model_id")): model
        for model in models
        if isinstance(model, dict)
    }


def _synthetic_evaluation(result: dict[str, object]) -> dict[str, object]:
    evaluations = result.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("result evaluations are invalid")
    matches = [
        evaluation
        for evaluation in evaluations
        if isinstance(evaluation, dict) and evaluation.get("lane") == "synthetic-v2"
    ]
    if len(matches) != 1 or matches[0].get("scope") != "complete":
        raise ValueError("result lacks one complete synthetic-v2 evaluation")
    return matches[0]


def _aggregate_model(
    result: dict[str, object], registry: dict[str, dict[str, object]]
) -> dict[str, object]:
    model = result.get("model")
    if not isinstance(model, dict):
        raise ValueError("result model identity is invalid")
    model_id = str(model.get("model_id"))
    revision = str(model.get("revision"))
    entry = registry.get(model_id)
    if (
        entry is None
        or entry.get("revision") != revision
        or entry.get("intended_lane") != "commercial_candidate"
        or not str(entry.get("commercial_use", "")).startswith("commercial_")
        or result.get("license_classification") != "commercial_candidate"
    ):
        raise ValueError("result model is not pinned to a permitted registry entry")

    evaluation = _synthetic_evaluation(result)
    metrics = evaluation.get("metrics")
    failures = evaluation.get("critical_failures")
    decoding = result.get("decoding")
    if not isinstance(metrics, dict) or not isinstance(failures, list):
        raise ValueError("result aggregate evidence is invalid")
    if not isinstance(decoding, dict):
        raise ValueError("result decoding configuration is invalid")
    failure_counts = Counter(
        str(failure.get("entity_type"))
        for failure in failures
        if isinstance(failure, dict)
    )
    required_metrics = {
        "wer_percent",
        "strict_lexical_recall_percent",
        "canonical_semantic_recall_percent",
        "substitutions",
        "insertions",
        "deletions",
        "local_rtfx",
        "median_latency_ms",
        "peak_vram_bytes",
    }
    if not required_metrics.issubset(metrics):
        raise ValueError("result metrics are incomplete")
    return {
        "model_id": model_id,
        "revision": revision,
        "license_classification": "commercial_candidate",
        "configuration": {
            key: decoding[key] for key in sorted(decoding) if key in _SAFE_CONFIGURATION
        },
        "aggregate_metrics": {
            **{key: metrics[key] for key in sorted(required_metrics)},
            "critical_failures_by_entity_type": dict(sorted(failure_counts.items())),
        },
        "dataset_count": evaluation.get("sample_count"),
        "evaluator_version": result.get("evaluator_revision"),
    }


def _report(models: list[dict[str, object]], dataset_count: int) -> str:
    lines = [
        "# Accessibility-Critical ASR Audit",
        "",
        EXECUTION_NOTICE,
        "",
        f"Dataset count: {dataset_count}",
        "",
        "| Model | WER | Strict recall | Canonical recall | Local RTFx |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        metrics = model["aggregate_metrics"]
        lines.append(
            f"| {model['model_id']} | {metrics['wer_percent']:.1f}% | "
            f"{metrics['strict_lexical_recall_percent']:.1f}% | "
            f"{metrics['canonical_semantic_recall_percent']:.1f}% | "
            f"{metrics['local_rtfx']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Only aggregate metrics are included. Results are not a certification or "
            "a Hugging Face leaderboard result.",
            "",
        ]
    )
    return "\n".join(lines)


def create_customer_export(
    *,
    repo_root: Path,
    result_paths: list[Path],
    output_dir: Path,
    signing_key: Path,
) -> CustomerExportResult:
    """Create a verified aggregate-only export from local result manifests."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("export directory must not already exist")
    if len(result_paths) != len(PILOT_MODEL_IDS):
        raise ValueError("export requires the exact three-model pilot set")
    registry = _registry_models(Path(repo_root))
    loaded = [(_load_json(Path(path)), Path(path)) for path in result_paths]
    aggregates = [_aggregate_model(result, registry) for result, _ in loaded]
    by_id = {str(model["model_id"]): model for model in aggregates}
    paths_by_id = {
        str(result["model"]["model_id"]): path for result, path in loaded
    }
    if set(by_id) != set(PILOT_MODEL_IDS):
        raise ValueError("export requires the exact three-model pilot set")
    ordered = [by_id[model_id] for model_id in PILOT_MODEL_IDS]
    counts = {model.pop("dataset_count") for model in ordered}
    evaluators = {model.pop("evaluator_version") for model in ordered}
    if len(counts) != 1 or len(evaluators) != 1:
        raise ValueError("model results do not share one dataset and evaluator")
    dataset_count = counts.pop()
    evaluator = evaluators.pop()
    if not isinstance(dataset_count, int) or not isinstance(evaluator, str):
        raise ValueError("model result identity is invalid")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix="deafbench-export-"
    ) as temporary:
        staging = Path(temporary) / "artifacts"
        staging.mkdir()
        report_path = staging / "report.md"
        report_path.write_text(
            _report(ordered, dataset_count), encoding="utf-8", newline="\n"
        )
        artifact_hashes = [
            {
                "artifact_type": "local_result_manifest",
                "model_id": str(model["model_id"]),
                "sha256": _sha256(path),
            }
            for model in ordered
            for path in (paths_by_id[str(model["model_id"])],)
        ]
        artifact_hashes.append(
            {"artifact_type": "redacted_report", "sha256": _sha256(report_path)}
        )
        digest = write_signed_manifest(
            staging / "manifest.json",
            payload={
                "schema_version": 1,
                "execution_notice": EXECUTION_NOTICE,
                "evaluator_version": evaluator,
                "dataset_count": dataset_count,
                "models": ordered,
                "artifact_hashes": artifact_hashes,
            },
            key_path=Path(signing_key),
        )
        assert_export_safe(staging)
        if not verify_signed_manifest(staging / "manifest.json"):
            raise ValueError("export manifest signature verification failed")
        staging.replace(destination)
    return CustomerExportResult(digest, len(ordered), dataset_count)
