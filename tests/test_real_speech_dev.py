import hashlib
import builtins
import io
import json
from pathlib import Path
import wave

import numpy as np
import pytest
import soundfile as sf

from deafbench.leaderboard.dev_corpus import (
    DEV_DATASET_REVISION,
    DevCorpusError,
    _pinned_source_rows,
    load_dev_contract,
    materialize_dev_corpus,
)
from deafbench.leaderboard import dev_corpus
from deafbench.leaderboard.official import OPEN_ASR_EVALUATOR_REVISION


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_versioned_real_speech_dev_contract_is_valid():
    corpus = REPO_ROOT / "benchmarks" / "real-speech-dev-v1"

    contract = load_dev_contract(
        corpus / "manifest.json", corpus / "references.jsonl"
    )

    assert len(contract.samples) == 100
    assert contract.population_count == 2_703


def test_versioned_real_speech_dev_baseline_is_self_consistent():
    result = json.loads(
        (
            REPO_ROOT
            / "experiments"
            / "real-speech-dev-v1"
            / "faster-whisper-small-en.json"
        ).read_text(encoding="utf-8")
    )
    corpus = REPO_ROOT / "benchmarks" / "real-speech-dev-v1"

    assert result["lane"] == "real-speech-dev-v1"
    assert result["result_kind"] == "local_development_baseline"
    assert (
        result["evaluator"]["upstream_revision"]
        == OPEN_ASR_EVALUATOR_REVISION
    )
    assert result["corpus"]["manifest_sha256"] == hashlib.sha256(
        (corpus / "manifest.json").read_bytes()
    ).hexdigest()
    assert result["corpus"]["references_sha256"] == hashlib.sha256(
        (corpus / "references.jsonl").read_bytes()
    ).hexdigest()
    metrics = result["metrics"]
    error_count = (
        metrics["substitutions"]
        + metrics["insertions"]
        + metrics["deletions"]
    )
    assert metrics["wer_percent"] == pytest.approx(
        100 * error_count / metrics["reference_words"]
    )
    assert metrics["strict_critical_recall_percent"] is None
    assert metrics["canonical_critical_recall_percent"] is None
    assert metrics["local_rtfx"] is None


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
        "selection": {
            "strategy": "sha256_id_lowest",
            "count": count,
            "population_count": 2,
        },
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
    ("contents", "message"),
    (
        ("[]", "manifest must be an object"),
        ("{not-json", "manifest is unreadable"),
    ),
)
def test_load_dev_contract_rejects_unreadable_manifest(
    tmp_path, contents, message
):
    manifest, references, _ = _write_contract(tmp_path)
    manifest.write_text(contents, encoding="utf-8")

    with pytest.raises(DevCorpusError, match=message):
        load_dev_contract(manifest, references, expected_count=2)


def test_load_dev_contract_rejects_manifest_directory(tmp_path):
    _, references, _ = _write_contract(tmp_path)

    with pytest.raises(DevCorpusError, match="manifest is unreadable"):
        load_dev_contract(tmp_path, references, expected_count=2)


def test_load_dev_contract_rejects_manifest_schema_drift(tmp_path):
    manifest_path, references, manifest = _write_contract(tmp_path)
    manifest.pop("purpose")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DevCorpusError, match="fields do not match"):
        load_dev_contract(manifest_path, references, expected_count=2)


@pytest.mark.parametrize("field", ("source", "selection"))
def test_load_dev_contract_rejects_non_object_sections(tmp_path, field):
    manifest_path, references, manifest = _write_contract(tmp_path)
    manifest[field] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DevCorpusError, match=f"{field} must be an object"):
        load_dev_contract(manifest_path, references, expected_count=2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dataset_id", "other/dataset", "dataset ID"),
        ("config", "other", "config"),
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


def test_load_dev_contract_rejects_unreadable_references(tmp_path):
    manifest_path, references, manifest = _write_contract(tmp_path)
    references.unlink()
    references.mkdir()
    manifest["references_sha256"] = hashlib.sha256(b"").hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DevCorpusError, match="references are unreadable"):
        load_dev_contract(manifest_path, references, expected_count=2)


def test_load_dev_contract_rejects_invalid_reference_json(tmp_path):
    manifest_path, references, manifest = _write_contract(tmp_path)
    references.write_text("{not-json\n", encoding="utf-8")
    manifest["references_sha256"] = hashlib.sha256(
        references.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DevCorpusError, match="references are invalid"):
        load_dev_contract(manifest_path, references, expected_count=2)


def test_pinned_source_rows_rejects_unsupported_python(monkeypatch, tmp_path):
    manifest, references, _ = _write_contract(tmp_path)
    contract = load_dev_contract(manifest, references, expected_count=2)
    monkeypatch.setattr(dev_corpus.sys, "version_info", (3, 14))

    with pytest.raises(DevCorpusError, match="Python 3.11-3.13"):
        _pinned_source_rows(contract)


def test_pinned_source_rows_preserves_transitive_import_failure(
    monkeypatch, tmp_path
):
    manifest, references, _ = _write_contract(tmp_path)
    contract = load_dev_contract(manifest, references, expected_count=2)
    monkeypatch.setattr(dev_corpus.sys, "version_info", (3, 13))
    real_import = builtins.__import__

    def missing_transitive(name, *args, **kwargs):
        if name == "datasets":
            raise ModuleNotFoundError(
                "No module named 'unexpected_dependency'",
                name="unexpected_dependency",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_transitive)

    with pytest.raises(ModuleNotFoundError, match="unexpected_dependency"):
        _pinned_source_rows(contract)


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


def test_load_dev_contract_rejects_invalid_source_population(tmp_path):
    manifest_path, references, manifest = _write_contract(tmp_path)
    manifest["selection"]["population_count"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DevCorpusError, match="population count"):
        load_dev_contract(manifest_path, references, expected_count=2)


def test_load_dev_contract_rejects_invalid_source_audio_hash(tmp_path):
    manifest_path, references, manifest = _write_contract(tmp_path)
    rows = [json.loads(line) for line in references.read_text().splitlines()]
    rows[0]["source_audio_sha256"] = "not-a-sha256"
    references.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest["references_sha256"] = hashlib.sha256(
        references.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DevCorpusError, match="audio hash is invalid"):
        load_dev_contract(manifest_path, references, expected_count=2)


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
        (
            lambda rows: rows[0].__setitem__("id", 1),
            "sample ID is invalid",
        ),
        (lambda rows: rows.__setitem__(0, {**rows[0], "text": "wrong"}), "text"),
        (lambda rows: rows.pop(), "population count"),
        (
            lambda rows: rows.__setitem__(1, {**rows[1], "id": rows[0]["id"]}),
            "duplicate sample IDs",
        ),
        (
            lambda rows: rows[0]["audio"].__setitem__("bytes", b"wrong"),
            "audio hash",
        ),
        (
            lambda rows: rows[0].__setitem__("audio", None),
            "audio payload",
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


def test_materialize_dev_corpus_rejects_source_order_drift(tmp_path):
    manifest, references, source_rows = _write_materialization_contract(tmp_path)
    source_rows[0]["id"] = "zzz"

    with pytest.raises(DevCorpusError, match="sample order changed"):
        materialize_dev_corpus(
            manifest,
            references,
            tmp_path / "audio",
            source_rows=source_rows,
            expected_count=2,
        )


def test_materialize_dev_corpus_rejects_unreadable_audio(tmp_path):
    manifest, references, source_rows = _write_materialization_contract(tmp_path)
    invalid_audio = b"not-a-readable-audio-container"
    source_rows[0]["audio"]["bytes"] = invalid_audio
    rows = [json.loads(line) for line in references.read_text().splitlines()]
    rows[0]["source_audio_sha256"] = hashlib.sha256(invalid_audio).hexdigest()
    references.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    contract = json.loads(manifest.read_text(encoding="utf-8"))
    contract["references_sha256"] = hashlib.sha256(
        references.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(DevCorpusError, match="audio is unreadable"):
        materialize_dev_corpus(
            manifest,
            references,
            tmp_path / "audio",
            source_rows=source_rows,
            expected_count=2,
        )
