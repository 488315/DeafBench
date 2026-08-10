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
