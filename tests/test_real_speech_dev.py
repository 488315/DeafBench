import hashlib
import json

import pytest

from deafbench.leaderboard.dev_corpus import (
    DEV_DATASET_REVISION,
    DevCorpusError,
    load_dev_contract,
)


def _write_contract(tmp_path, *, split="validation", count=2):
    references = tmp_path / "references.jsonl"
    rows = [
        {"id": "dev-001", "text": "first reference"},
        {"id": "dev-002", "text": "second reference"},
    ]
    references.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "name": "real-speech-dev-v1",
        "purpose": "model_selection_only",
        "source": {
            "dataset_id": "openslr/librispeech_asr",
            "revision": DEV_DATASET_REVISION,
            "config": "clean",
            "split": split,
            "license": "CC-BY-4.0",
        },
        "selection": {"strategy": "ordered_prefix", "count": count},
        "official_evaluation_exclusions": [
            "hf-audio/open_asr_leaderboard:librispeech:test.clean",
            "hf-audio/open_asr_leaderboard:librispeech:test.other",
        ],
        "references_sha256": hashlib.sha256(references.read_bytes()).hexdigest(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, references, manifest


def test_load_dev_contract_accepts_pinned_validation_cohort(tmp_path):
    manifest, references, _ = _write_contract(tmp_path)

    contract = load_dev_contract(manifest, references, expected_count=2)

    assert contract.dataset_id == "openslr/librispeech_asr"
    assert contract.revision == DEV_DATASET_REVISION
    assert contract.config == "clean"
    assert contract.split == "validation"
    assert contract.sample_ids == ("dev-001", "dev-002")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("split", "test", "validation split"),
        ("revision", "0" * 40, "revision"),
        ("license", "unknown", "license"),
    ),
)
def test_load_dev_contract_rejects_source_contract_drift(
    tmp_path, field, value, message
):
    manifest_path, references, manifest = _write_contract(tmp_path)
    manifest["source"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DevCorpusError, match=message):
        load_dev_contract(manifest_path, references, expected_count=2)


def test_load_dev_contract_rejects_reference_hash_mismatch(tmp_path):
    manifest, references, _ = _write_contract(tmp_path)
    references.write_text('{"id":"changed","text":"changed"}\n', encoding="utf-8")

    with pytest.raises(DevCorpusError, match="reference hash"):
        load_dev_contract(manifest, references, expected_count=2)


def test_load_dev_contract_requires_official_test_exclusions(tmp_path):
    manifest_path, references, manifest = _write_contract(tmp_path)
    manifest["official_evaluation_exclusions"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DevCorpusError, match="official test exclusions"):
        load_dev_contract(manifest_path, references, expected_count=2)


def test_load_dev_contract_rejects_incomplete_cohort(tmp_path):
    manifest, references, _ = _write_contract(tmp_path, count=1)

    with pytest.raises(DevCorpusError, match="sample count"):
        load_dev_contract(manifest, references, expected_count=2)
