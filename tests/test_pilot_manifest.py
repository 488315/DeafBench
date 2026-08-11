import json
from pathlib import Path

import pytest

from deafbench.pilot.manifest import (
    EXECUTION_NOTICE,
    verify_signed_manifest,
    write_signed_manifest,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "execution_notice": EXECUTION_NOTICE,
        "evaluator_version": "a" * 40,
        "dataset_count": 25,
        "models": [
            {
                "model_id": "Qwen/Qwen3-ASR-1.7B-hf",
                "revision": "b" * 40,
                "license_classification": "commercial_candidate",
                "configuration": {
                    "device": "cuda",
                    "max_new_tokens": 256,
                    "trust_remote_code": False,
                },
                "aggregate_metrics": {
                    "wer_percent": 21.0,
                    "strict_lexical_recall_percent": 67.7,
                    "canonical_semantic_recall_percent": 91.9,
                    "substitutions": 43,
                    "insertions": 10,
                    "deletions": 7,
                    "local_rtfx": 9.89,
                    "median_latency_ms": 495.9,
                    "peak_vram_bytes": 4144613888,
                    "critical_failures_by_entity_type": {"CODE": 2},
                },
            }
        ],
        "artifact_hashes": [
            {"artifact_type": "redacted_report", "sha256": "c" * 64}
        ],
    }


def test_manifest_is_signed_and_verifiable_with_embedded_public_key(
    tmp_path: Path,
) -> None:
    export = tmp_path / "export"
    export.mkdir()
    manifest = export / "manifest.json"
    key = tmp_path / "private" / "signing-key.pem"

    digest = write_signed_manifest(manifest, payload=_payload(), key_path=key)

    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["signature"]["algorithm"] == "Ed25519"
    assert len(digest) == 64
    assert key.exists()
    assert not (export / key.name).exists()
    assert verify_signed_manifest(manifest) is True


def test_manifest_signature_rejects_tampering(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    manifest = export / "manifest.json"
    write_signed_manifest(
        manifest, payload=_payload(), key_path=tmp_path / "private-key.pem"
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["dataset_count"] = 26
    manifest.write_text(json.dumps(document), encoding="utf-8")

    assert verify_signed_manifest(manifest) is False


def test_manifest_rejects_private_key_inside_export(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()

    with pytest.raises(ValueError, match="private signing key"):
        write_signed_manifest(
            export / "manifest.json",
            payload=_payload(),
            key_path=export / "signing-key.pem",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(dataset_count=0),
        lambda value: value.update(execution_notice="vendor executed"),
        lambda value: value["models"][0].update(sample_id="private-001"),
        lambda value: value["models"][0].update(license_classification="research_only"),
        lambda value: value["models"][0]["configuration"].update(audio_path="private.wav"),
        lambda value: value["models"][0]["aggregate_metrics"].update(wer_percent="21"),
        lambda value: value["artifact_hashes"][0].update(sha256="bad"),
    ],
)
def test_manifest_contract_fails_closed(mutation, tmp_path: Path) -> None:
    payload = _payload()
    mutation(payload)
    export = tmp_path / "export"
    export.mkdir()

    with pytest.raises(ValueError, match="manifest"):
        write_signed_manifest(
            export / "manifest.json",
            payload=payload,
            key_path=tmp_path / "private-key.pem",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(evaluator_version="unpinned"),
        lambda value: value.update(models=[]),
        lambda value: value["models"][0].update(revision="unpinned"),
        lambda value: value["models"].append(dict(value["models"][0])),
        lambda value: value["models"][0].update(aggregate_metrics={}),
        lambda value: value.update(artifact_hashes=[]),
        lambda value: value["artifact_hashes"][0].update(artifact_type="audio"),
        lambda value: value["artifact_hashes"][0].update(
            model_id="unregistered/model"
        ),
    ],
)
def test_manifest_rejects_unverifiable_boundaries(mutation, tmp_path: Path) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValueError, match="manifest"):
        write_signed_manifest(
            tmp_path / "export" / "manifest.json",
            payload=payload,
            key_path=tmp_path / "private-key.pem",
        )


@pytest.mark.parametrize(
    ("signature_field", "value"),
    [("algorithm", "RSA"), ("unexpected", "field")],
)
def test_manifest_verification_rejects_signature_contract_changes(
    tmp_path: Path, signature_field: str, value: str
) -> None:
    manifest = tmp_path / "export" / "manifest.json"
    write_signed_manifest(
        manifest, payload=_payload(), key_path=tmp_path / "private-key.pem"
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["signature"][signature_field] = value
    manifest.write_text(json.dumps(document), encoding="utf-8")

    assert verify_signed_manifest(manifest) is False
