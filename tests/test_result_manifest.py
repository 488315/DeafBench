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


def test_result_manifest_requires_both_separate_tracks() -> None:
    payload = deepcopy(_PAYLOAD)
    payload["evaluations"].pop()

    with pytest.raises(ResultManifestError, match="both evaluation tracks"):
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
