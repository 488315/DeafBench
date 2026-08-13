"""Validated, deterministic evidence manifests for local ASR evaluations."""

from __future__ import annotations

import json
import math
import re
import sys
from numbers import Real
from typing import Any, Mapping, Sequence

from deafbench.critical_entities import ENTITY_TYPES
from deafbench.model_registry import get_model_license


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40,64}")
_ACCESSIBILITY_METRICS = frozenset(
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
)
_LANE_METRICS = {
    "synthetic-v2": _ACCESSIBILITY_METRICS,
    "customer-audit": _ACCESSIBILITY_METRICS,
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
_STATUS_LANES = {
    "smoke_complete": frozenset(
        {"synthetic-v2", "hugging-face-compatibility-smoke"}
    ),
    "customer_audit_complete": frozenset({"customer-audit"}),
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


def _validate_metrics(lane: str, metrics: Mapping[str, Any]) -> None:
    expected = _LANE_METRICS[lane]
    if set(metrics) != expected:
        raise ResultManifestError(f"invalid {lane} metric fields")
    count_fields = {"substitutions", "insertions", "deletions", "peak_vram_bytes"}
    recall_fields = {
        "strict_lexical_recall_percent",
        "canonical_semantic_recall_percent",
    }
    for field, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ResultManifestError(f"invalid {field} metric")
        if value < 0 or (
            isinstance(value, int) and value > sys.float_info.max
        ) or (isinstance(value, float) and not math.isfinite(value)):
            raise ResultManifestError(f"invalid {field} metric")
        if field in count_fields and not isinstance(value, int):
            raise ResultManifestError(f"invalid {field} metric")
        if field in recall_fields and value > 100:
            raise ResultManifestError(f"invalid {field} metric")


def _validate_critical_failures(value: object) -> None:
    if not isinstance(value, list):
        raise ResultManifestError("critical_failures must be a list")
    expected = {"id", "term", "entity_type"}
    for item in value:
        failure = _mapping(item, "critical failure")
        if set(failure) != expected:
            raise ResultManifestError("invalid critical failure fields")
        if any(
            not isinstance(failure[field], str) or not failure[field].strip()
            for field in ("id", "term", "entity_type")
        ):
            raise ResultManifestError("critical failure fields must be nonempty strings")
        if failure["entity_type"] not in ENTITY_TYPES | {"UNCLASSIFIED"}:
            raise ResultManifestError("invalid critical failure entity_type")


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
    if (
        isinstance(manifest["schema_version"], bool)
        or not isinstance(manifest["schema_version"], int)
        or manifest["schema_version"] != 1
        or not isinstance(manifest["status"], str)
        or manifest["status"] not in _STATUS_LANES
    ):
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
        expected_scope = (
            "partial" if lane == "hugging-face-compatibility-smoke" else "complete"
        )
        if record["scope"] != expected_scope:
            raise ResultManifestError(f"invalid scope for evaluation lane: {lane}")
        if (
            isinstance(record["sample_count"], bool)
            or not isinstance(record["sample_count"], int)
            or record["sample_count"] <= 0
        ):
            raise ResultManifestError(f"invalid sample count for evaluation lane: {lane}")
        metrics = _mapping(record["metrics"], "metrics")
        _validate_metrics(lane, metrics)
        _validate_critical_failures(record["critical_failures"])
    expected_lanes = _STATUS_LANES[str(manifest["status"])]
    if lanes != expected_lanes:
        if manifest["status"] == "smoke_complete":
            raise ResultManifestError("both evaluation tracks are required")
        raise ResultManifestError("evaluation lanes do not match result state")
    return manifest


def canonical_result_bytes(payload: object) -> bytes:
    """Return stable UTF-8 JSON bytes after validating the evidence."""
    manifest = validate_result_manifest(payload)
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
