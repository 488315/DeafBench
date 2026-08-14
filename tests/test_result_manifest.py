import json
from copy import deepcopy
from pathlib import Path

import pytest

from deafbench.result_manifest import (
    ResultManifestError,
    canonical_result_bytes,
    validate_result_manifest,
)


_PAYLOAD = {
    "schema_version": 1,
    "status": "smoke_complete",
    "model": {
        "model_id": "Qwen/Qwen3-ASR-0.6B-hf",
        "revision": "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
    },
    "license_classification": "commercial_candidate",
    "evaluator_revision": "6f5b294c07a2d7b37094336cec8e01011556850b",
    "decoding": {"language": "English"},
    "corpora": [{"name": "synthetic-v2", "manifest_sha256": "a" * 64, "frozen": True}],
    "evaluations": [
        {
            "lane": "synthetic-v2",
            "scope": "complete",
            "sample_count": 25,
            "metrics": {
                "wer_percent": 1.0,
                "strict_lexical_recall_percent": 2.0,
                "canonical_semantic_recall_percent": 3.0,
                "substitutions": 1,
                "insertions": 2,
                "deletions": 3,
                "local_rtfx": 4.0,
                "median_latency_ms": 5.0,
                "peak_vram_bytes": 6,
            },
            "critical_failures": [],
        },
        {
            "lane": "hugging-face-compatibility-smoke",
            "scope": "partial",
            "sample_count": 2,
            "metrics": {
                "wer_percent": 1.0,
                "substitutions": 1,
                "insertions": 0,
                "deletions": 0,
                "local_rtfx": 4.0,
                "median_latency_ms": 5.0,
                "peak_vram_bytes": 6,
            },
            "critical_failures": [],
        },
    ],
    "verification": {
        "status": "recorded_local_observation",
        "sample_artifacts_in_repository": False,
        "independently_recomputable_from_checkout": False,
    },
    "claim_boundary": "Local compatibility smoke only.",
}


def _customer_payload() -> dict:
    payload = deepcopy(_PAYLOAD)
    payload["status"] = "customer_audit_complete"
    payload["corpora"] = [
        {
            "name": "customer-authorized-audio",
            "manifest_sha256": "b" * 64,
            "frozen": True,
        }
    ]
    payload["evaluations"] = [deepcopy(payload["evaluations"][0])]
    payload["evaluations"][0]["lane"] = "customer-audit"
    payload["claim_boundary"] = "Customer-executed local audit."
    return payload


def _non_speech_payload() -> dict:
    payload = deepcopy(_PAYLOAD)
    payload["status"] = "smoke_complete_with_non_speech"
    payload["corpora"].append(
        {
            "name": "non-speech-v1",
            "manifest_sha256": "c" * 64,
            "frozen": True,
        }
    )
    payload["evaluations"].append(
        {
            "lane": "non-speech-v1",
            "scope": "complete",
            "sample_count": 12,
            "metrics": {
                "wer_percent": 1.0,
                "strict_lexical_recall_percent": 95.0,
                "canonical_semantic_recall_percent": 95.0,
                "substitutions": 2,
                "insertions": 0,
                "deletions": 0,
                "non_speech_recall_percent": 0.0,
                "matched_sound_events": 0,
                "total_sound_events": 19,
                "local_rtfx": 4.0,
                "median_latency_ms": 5.0,
                "peak_vram_bytes": 6,
            },
            "critical_failures": [],
        }
    )
    return payload


def test_result_manifest_requires_both_separate_tracks() -> None:
    payload = deepcopy(_PAYLOAD)
    payload["evaluations"].pop()

    with pytest.raises(ResultManifestError, match="both evaluation tracks"):
        validate_result_manifest(payload)


def test_result_manifest_accepts_complete_customer_audit() -> None:
    assert validate_result_manifest(_customer_payload())["status"] == (
        "customer_audit_complete"
    )


def test_result_manifest_accepts_separate_non_speech_evidence() -> None:
    assert validate_result_manifest(_non_speech_payload())["status"] == (
        "smoke_complete_with_non_speech"
    )


def test_non_speech_result_requires_all_three_evaluation_lanes() -> None:
    payload = _non_speech_payload()
    payload["evaluations"].pop(0)

    with pytest.raises(ResultManifestError, match="lanes do not match"):
        validate_result_manifest(payload)


def test_non_speech_result_rejects_inconsistent_sound_counts() -> None:
    payload = _non_speech_payload()
    payload["evaluations"][2]["metrics"]["matched_sound_events"] = 20

    with pytest.raises(ResultManifestError, match="sound event counts"):
        validate_result_manifest(payload)


def test_customer_audit_manifest_rejects_extra_evaluation_lane() -> None:
    payload = _customer_payload()
    payload["evaluations"].append(deepcopy(_PAYLOAD["evaluations"][1]))

    with pytest.raises(ResultManifestError, match="lanes do not match"):
        validate_result_manifest(payload)


def test_customer_audit_manifest_requires_complete_scope() -> None:
    payload = _customer_payload()
    payload["evaluations"][0]["scope"] = "partial"

    with pytest.raises(ResultManifestError, match="invalid scope"):
        validate_result_manifest(payload)


def test_result_manifest_rejects_unfrozen_corpus() -> None:
    payload = deepcopy(_PAYLOAD)
    payload["corpora"][0]["frozen"] = False

    with pytest.raises(ResultManifestError, match="requires frozen corpora"):
        validate_result_manifest(payload)


def test_result_manifest_rejects_broadened_smoke_scope() -> None:
    payload = deepcopy(_PAYLOAD)
    payload["evaluations"][1]["scope"] = "complete"

    with pytest.raises(ResultManifestError, match="invalid scope"):
        validate_result_manifest(payload)


def test_result_manifest_is_byte_stable() -> None:
    reordered = json.loads(json.dumps(_PAYLOAD, sort_keys=True))

    assert canonical_result_bytes(_PAYLOAD) == canonical_result_bytes(reordered)


def test_whisper_at_result_manifest_is_valid_and_byte_stable() -> None:
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "model-results"
        / "whisper-at-medium-en.json"
    )
    raw = path.read_bytes()
    payload = json.loads(raw)

    assert canonical_result_bytes(payload) == raw


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("claim_boundary"), "result manifest missing"),
        (lambda payload: payload.pop("verification"), "result manifest missing"),
        (lambda payload: payload.update(schema_version=2), "unsupported"),
        (lambda payload: payload.update(schema_version=True), "unsupported"),
        (lambda payload: payload.update(status="draft"), "unsupported"),
        (lambda payload: payload.update(status=[]), "unsupported"),
        (lambda payload: payload.update(claim_boundary=""), "claim_boundary"),
        (
            lambda payload: payload["verification"].update(
                sample_artifacts_in_repository=True
            ),
            "verification state",
        ),
        (
            lambda payload: payload["verification"].update(
                independently_recomputable_from_checkout=True
            ),
            "verification state",
        ),
        (
            lambda payload: payload["verification"].update(status="verified"),
            "verification state",
        ),
        (lambda payload: payload.update(evaluator_revision="main"), "evaluator revision"),
        (lambda payload: payload.update(decoding=[]), "decoding must be an object"),
        (lambda payload: payload.update(corpora="synthetic-v2"), "corpora"),
        (lambda payload: payload.update(corpora=[]), "corpora"),
        (
            lambda payload: payload["corpora"][0].update(manifest_sha256="bad"),
            "corpus manifest hash",
        ),
        (lambda payload: payload.update(evaluations={}), "evaluations must be a list"),
        (
            lambda payload: payload["evaluations"][0].update(lane="unknown"),
            "invalid or duplicate",
        ),
        (
            lambda payload: payload["evaluations"][1].update(lane="synthetic-v2"),
            "invalid or duplicate",
        ),
        (
            lambda payload: payload["evaluations"][0].update(sample_count=0),
            "invalid sample count",
        ),
        (
            lambda payload: payload["evaluations"][0].update(sample_count=True),
            "invalid sample count",
        ),
        (
            lambda payload: payload["evaluations"][0].update(metrics=[]),
            "metrics must be an object",
        ),
        (
            lambda payload: payload["evaluations"][0]["metrics"].pop("wer_percent"),
            "metric fields",
        ),
        (
            lambda payload: payload["evaluations"][0].update(critical_failures={}),
            "critical_failures",
        ),
    ],
)
def test_result_manifest_rejects_incomplete_evidence(mutation, message) -> None:
    payload = deepcopy(_PAYLOAD)
    mutation(payload)

    with pytest.raises(ResultManifestError, match=message):
        validate_result_manifest(payload)


def test_result_manifest_rejects_unknown_model() -> None:
    payload = deepcopy(_PAYLOAD)
    payload["model"]["model_id"] = "unknown/model"

    with pytest.raises(ValueError, match="missing license metadata"):
        validate_result_manifest(payload)


def test_result_manifest_rejects_revision_or_lane_drift() -> None:
    revision = deepcopy(_PAYLOAD)
    revision["model"]["revision"] = "a" * 40
    with pytest.raises(ResultManifestError, match="revision differs"):
        validate_result_manifest(revision)

    lane = deepcopy(_PAYLOAD)
    lane["license_classification"] = "research_only"
    with pytest.raises(ResultManifestError, match="classification differs"):
        validate_result_manifest(lane)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("wer_percent", True, "wer_percent"),
        ("wer_percent", float("nan"), "wer_percent"),
        ("strict_lexical_recall_percent", 101.0, "strict_lexical"),
        ("canonical_semantic_recall_percent", -1.0, "canonical_semantic"),
        ("substitutions", 1.5, "substitutions"),
        ("insertions", -1, "insertions"),
        ("local_rtfx", float("inf"), "local_rtfx"),
        ("median_latency_ms", -1.0, "median_latency_ms"),
        ("peak_vram_bytes", True, "peak_vram_bytes"),
    ],
)
def test_result_manifest_rejects_invalid_metric_values(
    field: str, value: object, message: str
) -> None:
    payload = deepcopy(_PAYLOAD)
    payload["evaluations"][0]["metrics"][field] = value

    with pytest.raises(ResultManifestError, match=message):
        validate_result_manifest(payload)


def test_result_manifest_rejects_oversized_integer_metric() -> None:
    payload = deepcopy(_PAYLOAD)
    payload["evaluations"][0]["metrics"]["substitutions"] = 10**1000

    with pytest.raises(ResultManifestError, match="substitutions"):
        validate_result_manifest(payload)


def test_result_manifest_rejects_extra_metric() -> None:
    payload = deepcopy(_PAYLOAD)
    payload["evaluations"][0]["metrics"]["unreviewed_score"] = 1

    with pytest.raises(ResultManifestError, match="metric fields"):
        validate_result_manifest(payload)


@pytest.mark.parametrize(
    "failure",
    [
        "core-001",
        {"id": "core-001", "term": "8:30", "entity_type": "TIME", "raw": "x"},
        {"id": "", "term": "8:30", "entity_type": "TIME"},
        {"id": "core-001", "term": "8:30", "entity_type": "UNKNOWN"},
    ],
)
def test_result_manifest_rejects_invalid_critical_failure(failure: object) -> None:
    payload = deepcopy(_PAYLOAD)
    payload["evaluations"][0]["critical_failures"] = [failure]

    with pytest.raises(ResultManifestError, match="critical failure"):
        validate_result_manifest(payload)
