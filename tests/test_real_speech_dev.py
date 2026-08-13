import hashlib
import io
import json
import wave

import numpy as np
import pytest
import soundfile as sf

from deafbench.leaderboard.dev_corpus import (
    DEV_DATASET_REVISION,
    DevCorpusError,
    load_dev_contract,
    materialize_dev_corpus,
)


def _write_contract(tmp_path, *, split="validation", count=2):
    references = tmp_path / "references.jsonl"
    rows = [
        {
            "id": "dev-001",
            "text": "first reference",
            "source_audio_sha256": "1" * 64,
        },
        {
            "id": "dev-002",
            "text": "second reference",
            "source_audio_sha256": "2" * 64,
        },
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


def _flac_bytes(frequency):
    time = np.arange(1600, dtype=np.float32) / 16_000
    audio = (0.1 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    payload = io.BytesIO()
    sf.write(payload, audio, 16_000, format="FLAC")
    return payload.getvalue()


def _write_materialization_contract(tmp_path):
    encoded = (_flac_bytes(220), _flac_bytes(330))
    manifest_path, references, manifest = _write_contract(tmp_path)
    rows = [json.loads(line) for line in references.read_text().splitlines()]
    for row, payload in zip(rows, encoded, strict=True):
        row["source_audio_sha256"] = hashlib.sha256(payload).hexdigest()
    references.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest["references_sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_rows = [
        {
            "id": row["id"],
            "text": row["text"],
            "audio": {"bytes": payload, "path": f"{row['id']}.flac"},
        }
        for row, payload in zip(rows, encoded, strict=True)
    ]
    return manifest_path, references, source_rows


def test_materialize_dev_corpus_writes_verified_48khz_audio(tmp_path):
    manifest, references, source_rows = _write_materialization_contract(tmp_path)
    destination = tmp_path / "real-speech-dev-v1" / "audio"

    result = materialize_dev_corpus(
        manifest,
        references,
        destination,
        source_rows=source_rows,
        expected_count=2,
    )

    assert result["sample_count"] == 2
    assert result["source"]["revision"] == DEV_DATASET_REVISION
    assert [row["id"] for row in result["audio"]] == ["dev-001", "dev-002"]
    for sample_id in ("dev-001", "dev-002"):
        with wave.open(str(destination / f"{sample_id}.wav"), "rb") as handle:
            assert handle.getframerate() == 48_000
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
    saved = json.loads((destination / "materialization-manifest.json").read_text())
    assert saved == result


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda rows: rows.__setitem__(0, {**rows[0], "text": "wrong"}), "text"),
        (lambda rows: rows.pop(), "ended before"),
        (
            lambda rows: rows[0]["audio"].__setitem__("bytes", b"wrong"),
            "audio hash",
        ),
    ),
)
def test_materialize_dev_corpus_rejects_source_drift_without_replacing_output(
    tmp_path, mutation, message
):
    manifest, references, source_rows = _write_materialization_contract(tmp_path)
    destination = tmp_path / "real-speech-dev-v1" / "audio"
    destination.mkdir(parents=True)
    marker = destination / "existing.txt"
    marker.write_text("preserve", encoding="utf-8")
    mutation(source_rows)

    with pytest.raises(DevCorpusError, match=message):
        materialize_dev_corpus(
            manifest,
            references,
            destination,
            source_rows=source_rows,
            expected_count=2,
        )

    assert marker.read_text(encoding="utf-8") == "preserve"
