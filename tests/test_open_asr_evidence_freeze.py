import hashlib
import json
from pathlib import Path

import pytest

from deafbench.leaderboard.evidence import (
    EvidenceIntegrityError,
    verify_evidence_manifest,
)


_ROOT = Path(__file__).parents[1]
_EXPERIMENT = _ROOT / "experiments" / "open-asr"


def test_official_evidence_manifest_matches_frozen_artifacts():
    result = verify_evidence_manifest(
        _EXPERIMENT / "evidence-manifest.json",
        repo_root=_ROOT,
    )

    assert result.artifact_count == 9
    assert result.result_rows == 74_737
    assert result.composite_wer == 5.23
    assert result.hardware_label == "local RTX 4070"


def test_official_evidence_manifest_rejects_modified_artifact(tmp_path):
    manifest_path = _EXPERIMENT / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "changed.json"
    (tmp_path / "changed.json").write_text("changed\n", encoding="utf-8")
    changed_manifest = tmp_path / "manifest.json"
    changed_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EvidenceIntegrityError, match="hash mismatch"):
        verify_evidence_manifest(changed_manifest, repo_root=tmp_path)


def _write_evidence(tmp_path, *, artifact=b"one\ntwo\n"):
    artifact_path = tmp_path / "results.jsonl"
    artifact_path.write_bytes(artifact)
    manifest = {
        "artifacts": [
            {
                "path": "results.jsonl",
                "sha256": hashlib.sha256(artifact).hexdigest(),
                "bytes": len(artifact),
                "rows": 2,
            }
        ],
        "metrics": {"public_seven_set_macro_wer": 5.23},
        "hardware": {"label": "local test hardware"},
        "result_rows": 2,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.update(artifacts=[]), "no artifacts"),
        (lambda manifest: manifest.update(artifacts=["artifact"]), "must be an object"),
        (
            lambda manifest: manifest["artifacts"][0].update(path=""),
            "nonempty string",
        ),
        (
            lambda manifest: manifest["artifacts"][0].update(path="../outside"),
            "escapes repository",
        ),
        (
            lambda manifest: manifest["artifacts"].append(
                dict(manifest["artifacts"][0])
            ),
            "duplicate artifact",
        ),
        (
            lambda manifest: manifest["artifacts"][0].update(bytes=1),
            "size mismatch",
        ),
        (
            lambda manifest: manifest["artifacts"][0].update(rows=1),
            "row mismatch",
        ),
        (lambda manifest: manifest.update(metrics=[]), "metadata is incomplete"),
        (
            lambda manifest: manifest["metrics"].update(
                public_seven_set_macro_wer="5.23"
            ),
            "missing public seven-set",
        ),
        (
            lambda manifest: manifest["hardware"].update(label="RTX 4070"),
            "labeled local",
        ),
        (
            lambda manifest: manifest.update(result_rows=1),
            "aggregate result row count",
        ),
    ],
)
def test_evidence_manifest_rejects_untrusted_metadata(tmp_path, mutate, message):
    manifest, manifest_path = _write_evidence(tmp_path)
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EvidenceIntegrityError, match=message):
        verify_evidence_manifest(manifest_path, repo_root=tmp_path)


def test_evidence_manifest_rejects_missing_artifact(tmp_path):
    _, manifest_path = _write_evidence(tmp_path)
    (tmp_path / "results.jsonl").unlink()

    with pytest.raises(EvidenceIntegrityError, match="missing evidence artifact"):
        verify_evidence_manifest(manifest_path, repo_root=tmp_path)


@pytest.mark.parametrize("content", ["not json", "[]"])
def test_evidence_manifest_rejects_invalid_document(tmp_path, content):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(content, encoding="utf-8")

    with pytest.raises(EvidenceIntegrityError):
        verify_evidence_manifest(manifest_path, repo_root=tmp_path)
