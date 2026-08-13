import json
from pathlib import Path

import pytest

from deafbench.benchmark.freeze import FrozenCorpusError, verify_frozen_corpus


_ROOT = Path(__file__).parents[1]
_MANIFEST = _ROOT / "benchmarks" / "core-v1" / "freeze-manifest.json"


def test_core_v1_freeze_records_complete_reproducibility_contract():
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["corpus"] == "core-v1"
    assert manifest["status"] == "frozen"
    assert manifest["baseline"] == {
        "canonical_semantic_critical_recall": 90.3,
        "deletions": 33,
        "insertions": 9,
        "strict_lexical_critical_recall": 69.4,
        "substitutions": 33,
        "wer": 26.2,
    }
    assert manifest["generation"]["scene_profile"] == "default-v1"
    assert manifest["generation"]["seed"] == 42
    assert manifest["generation"]["tts"]["package_version"] == "0.8.9"
    assert manifest["model"]["revision"] == (
        "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
    )
    assert manifest["model"]["decoding"] == {
        "beam_size": 5,
        "compute_type": "int8",
        "device": "cpu",
        "language": "en",
    }
    assert manifest["evaluator_commit"] == (
        "236aa9bf293231737b0580f85cb56bca210a2fae"
    )
    assert len(manifest["artifacts"]["audio"]) == 25


def test_checked_out_core_v1_matches_its_freeze_manifest():
    result = verify_frozen_corpus(_MANIFEST, _ROOT)

    assert result.verified_required >= 2
    assert result.verified_optional + len(result.missing_optional) == 29


def test_freeze_verifier_rejects_silent_required_file_change(tmp_path: Path):
    frozen = tmp_path / "frozen.txt"
    frozen.write_text("original", encoding="utf-8")
    manifest = _write_test_manifest(tmp_path, required={"frozen.txt": _sha256(frozen)})
    frozen.write_text("changed", encoding="utf-8")

    with pytest.raises(FrozenCorpusError, match="hash mismatch.*frozen.txt"):
        verify_frozen_corpus(manifest, tmp_path)


def test_freeze_verifier_checks_optional_artifacts_when_present(tmp_path: Path):
    generated = tmp_path / "generated.wav"
    generated.write_bytes(b"audio")
    manifest = _write_test_manifest(
        tmp_path,
        optional={"generated.wav": _sha256(generated)},
    )
    generated.write_bytes(b"modified")

    with pytest.raises(FrozenCorpusError, match="hash mismatch.*generated.wav"):
        verify_frozen_corpus(manifest, tmp_path)


def test_freeze_verifier_can_require_generated_evidence(tmp_path: Path):
    manifest = _write_test_manifest(
        tmp_path,
        optional={"missing.wav": "0" * 64},
    )

    result = verify_frozen_corpus(manifest, tmp_path)
    assert result.missing_optional == ("missing.wav",)

    with pytest.raises(FrozenCorpusError, match="missing frozen artifact"):
        verify_frozen_corpus(manifest, tmp_path, require_optional=True)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_test_manifest(
    root: Path,
    *,
    required: dict[str, str] | None = None,
    optional: dict[str, str] | None = None,
) -> Path:
    manifest = root / "freeze.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": {
                    "required": required or {},
                    "optional": optional or {},
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest
