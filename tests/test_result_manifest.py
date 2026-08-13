import json
from copy import deepcopy

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


def test_result_manifest_requires_both_separate_tracks() -> None:
    payload = deepcopy(_PAYLOAD)
    payload["evaluations"].pop()

    with pytest.raises(ResultManifestError, match="both evaluation tracks"):
        validate_result_manifest(payload)


def test_result_manifest_accepts_complete_customer_audit() -> None:
    assert validate_result_manifest(_customer_payload())["status"] == (
        "customer_audit_complete"
    )


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("claim_boundary"), "result manifest missing"),
        (lambda payload: payload.update(schema_version=2), "unsupported"),
        (lambda payload: payload.update(status="draft"), "unsupported"),
        (lambda payload: payload.update(claim_boundary=""), "claim_boundary"),
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
            lambda payload: payload["evaluations"][0].update(metrics=[]),
            "metrics must be an object",
        ),
        (
            lambda payload: payload["evaluations"][0]["metrics"].pop("wer_percent"),
            "metrics missing",
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
