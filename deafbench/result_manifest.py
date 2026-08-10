"""Validated, deterministic evidence manifests for local ASR evaluations."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from deafbench.model_registry import get_model_license


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40,64}")
_LANE_METRICS = {
    "synthetic-v2": frozenset(
        {
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
    ),
    "hugging-face-compatibility-smoke": frozenset(
        {
            "wer_percent",
            "substitutions",
            "insertions",
            "deletions",
            "local_rtfx",
            "median_latency_ms",
            "peak_vram_bytes",
        }
    ),
}


class ResultManifestError(ValueError):
    """Raised when benchmark result evidence is incomplete or misleading."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultManifestError(f"{label} must be an object")
    return value


def _required(record: Mapping[str, Any], fields: set[str], label: str) -> None:
    missing = fields - record.keys()
    if missing:
        raise ResultManifestError(f"{label} missing: {', '.join(sorted(missing))}")


def validate_result_manifest(payload: object) -> Mapping[str, Any]:
    """Validate one model's local evidence without broadening its claim scope."""
    manifest = _mapping(payload, "result manifest")
    _required(
        manifest,
        {
            "schema_version",
            "status",
            "model",
            "license_classification",
            "evaluator_revision",
            "decoding",
            "corpora",
            "evaluations",
            "claim_boundary",
        },
        "result manifest",
    )
    if manifest["schema_version"] != 1 or manifest["status"] != "smoke_complete":
        raise ResultManifestError("unsupported result manifest state")
    if not isinstance(manifest["claim_boundary"], str) or not manifest[
        "claim_boundary"
    ].strip():
        raise ResultManifestError("claim_boundary must be nonempty")

    model = _mapping(manifest["model"], "model")
    _required(model, {"model_id", "revision"}, "model")
    license_entry = get_model_license(str(model["model_id"]))
    if model["revision"] != license_entry.revision:
        raise ResultManifestError("model revision differs from license registry")
    if manifest["license_classification"] != license_entry.intended_lane:
        raise ResultManifestError("license classification differs from registry")
    if _REVISION.fullmatch(str(manifest["evaluator_revision"])) is None:
        raise ResultManifestError("invalid evaluator revision")
    _mapping(manifest["decoding"], "decoding")

    corpora = manifest["corpora"]
    if not isinstance(corpora, Sequence) or isinstance(corpora, (str, bytes)):
        raise ResultManifestError("corpora must be a nonempty list")
    if not corpora:
        raise ResultManifestError("corpora must be a nonempty list")
    for corpus in corpora:
        record = _mapping(corpus, "corpus")
        _required(record, {"name", "manifest_sha256", "frozen"}, "corpus")
        if _SHA256.fullmatch(str(record["manifest_sha256"])) is None:
            raise ResultManifestError("invalid corpus manifest hash")
        if record["frozen"] is not True:
            raise ResultManifestError("result evidence requires frozen corpora")

    evaluations = manifest["evaluations"]
    if not isinstance(evaluations, Sequence) or isinstance(
        evaluations, (str, bytes)
    ):
        raise ResultManifestError("evaluations must be a list")
    lanes = set()
    for evaluation in evaluations:
        record = _mapping(evaluation, "evaluation")
        _required(
            record,
            {"lane", "scope", "sample_count", "metrics", "critical_failures"},
            "evaluation",
        )
        lane = str(record["lane"])
        if lane not in _LANE_METRICS or lane in lanes:
            raise ResultManifestError(f"invalid or duplicate evaluation lane: {lane}")
        lanes.add(lane)
        expected_scope = "complete" if lane == "synthetic-v2" else "partial"
        if record["scope"] != expected_scope:
            raise ResultManifestError(f"invalid scope for evaluation lane: {lane}")
        if not isinstance(record["sample_count"], int) or record["sample_count"] <= 0:
            raise ResultManifestError(f"invalid sample count for evaluation lane: {lane}")
        metrics = _mapping(record["metrics"], "metrics")
        _required(metrics, set(_LANE_METRICS[lane]), f"{lane} metrics")
        if not isinstance(record["critical_failures"], list):
            raise ResultManifestError("critical_failures must be a list")
    if lanes != set(_LANE_METRICS):
        raise ResultManifestError("both evaluation tracks are required")
    return manifest


def canonical_result_bytes(payload: object) -> bytes:
    """Return stable UTF-8 JSON bytes after validating the evidence."""
    manifest = validate_result_manifest(payload)
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
