import json
from pathlib import Path

from deafbench.benchmark.freeze import verify_frozen_corpus


_ROOT = Path(__file__).parents[1]
_CORE_MANIFEST = _ROOT / "benchmarks" / "core-v1" / "freeze-manifest.json"
_V2_MANIFEST = _ROOT / "benchmarks" / "synthetic-v2" / "freeze-manifest.json"


def _audio_hashes(manifest: dict) -> dict[str, str]:
    return {
        Path(record["path"]).name: record["sha256"]
        for record in manifest["artifacts"]["audio"]
        if "/audio-synthetic/" in record["path"]
    }


def test_synthetic_v2_freeze_records_separate_baseline_contract():
    manifest = json.loads(_V2_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["corpus"] == "synthetic-v2"
    assert manifest["status"] == "frozen"
    assert manifest["generation"]["inherited_sample_count"] == 21
    assert manifest["generation"]["replacement_sample_ids"] == [
        "core-001",
        "core-009",
        "core-011",
        "core-016",
    ]
    assert manifest["validation"]["accepted_replacement_count"] == 4
    assert manifest["baseline"] == {
        "canonical_critical_failures": [
            {"id": "core-019", "term": "Office Guest"},
            {"id": "core-019", "term": "alpha seven nine"},
        ],
        "canonical_semantic_critical_recall": 96.8,
        "deletions": 31,
        "insertions": 6,
        "strict_lexical_critical_recall": 69.4,
        "substitutions": 35,
        "wer": 25.2,
    }
    assert manifest["model"]["revision"] == (
        "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
    )
    assert manifest["model"]["decoding"] == {
        "beam_size": 5,
        "compute_type": "int8",
        "device": "cpu",
        "language": "en",
    }


def test_synthetic_v2_matches_its_freeze_manifest():
    result = verify_frozen_corpus(_V2_MANIFEST, _ROOT)

    assert result.verified_required == 4
    assert result.verified_optional == 32
    assert result.missing_optional == ()


def test_only_declared_v2_replacements_change_parent_audio():
    core = json.loads(_CORE_MANIFEST.read_text(encoding="utf-8"))
    v2 = json.loads(_V2_MANIFEST.read_text(encoding="utf-8"))
    core_hashes = _audio_hashes(core)
    v2_hashes = _audio_hashes(v2)
    changed = {
        name
        for name, digest in v2_hashes.items()
        if digest != core_hashes[name]
    }

    assert changed == {
        "core-001.wav",
        "core-009.wav",
        "core-011.wav",
        "core-016.wav",
    }
    assert (
        _ROOT / "benchmarks" / "synthetic-v2" / "references.jsonl"
    ).read_bytes() == (
        _ROOT / "benchmarks" / "core-v1" / "references.jsonl"
    ).read_bytes()
