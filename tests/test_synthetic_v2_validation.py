import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from deafbench.benchmark.quality import AlignmentEvidence
from deafbench.benchmark.synthetic_v2_validation import (
    _jsonl_by_id,
    validate_replacement_candidates,
)


class _Aligner:
    adapter_revision = "aligner-test-revision"

    def __init__(self):
        self.paths = []

    def align(self, audio_path, prepared, *, score_threshold):
        self.paths.append(audio_path)
        return AlignmentEvidence(
            reference_sha256=prepared.reference_sha256,
            token_coverage=1.0,
            critical_entity_coverage={term: 1.0 for term in prepared.entity_word_ranges},
            coverage_score_threshold=score_threshold,
            adapter="test-forced-aligner",
            adapter_revision=self.adapter_revision,
        )


class _ASR:
    adapter_revision = "asr-test-revision"

    def __init__(self):
        self.paths = []

    def transcribe(self, audio_path):
        self.paths.append(audio_path)
        return "the meeting starts at eight thirty p m"


def _write_wav(path: Path) -> None:
    rate = 48_000
    silence = np.zeros(round(rate * 0.2))
    time = np.arange(round(rate * 0.6)) / rate
    speech = 0.2 * np.sin(2 * math.pi * 220 * time)
    pcm = np.round(np.concatenate((silence, speech, silence)) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


@pytest.mark.parametrize("record", [[], "sample-1", 1, None])
def test_validation_rejects_nonobject_json_records(tmp_path: Path, record) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid validation input record"):
        _jsonl_by_id(manifest)


def test_validation_records_all_gate_and_validator_evidence(tmp_path: Path):
    corpus = tmp_path / "corpus"
    audio = corpus / "audio-synthetic"
    audio.mkdir(parents=True)
    validation_speech = corpus / "validation-speech"
    validation_speech.mkdir()
    reference = {
        "id": "sample-1",
        "text": "The meeting starts at 8:30 PM.",
        "critical": ["8:30 PM"],
        "critical_types": {"8:30 PM": "TIME"},
        "sounds": [],
    }
    text = reference["text"]
    digest = hashlib.sha256(text.encode()).hexdigest()
    (corpus / "references.jsonl").write_text(json.dumps(reference) + "\n")
    (corpus / "generation-manifest.jsonl").write_text(
        json.dumps({"id": "sample-1", "reference_sha256": digest}) + "\n"
    )
    _write_wav(audio / "sample-1.wav")
    _write_wav(validation_speech / "sample-1.wav")
    policy = Path(__file__).parents[1] / "benchmarks/synthetic-v2/quality-policy.json"

    aligner = _Aligner()
    independent_asr = _ASR()
    report = validate_replacement_candidates(
        corpus,
        policy,
        aligner,
        independent_asr,
        sample_ids=("sample-1",),
    )

    sample = report["samples"][0]
    assert sample["status"] == "accepted"
    assert len(sample["gates"]) == 10
    assert sample["independent_asr"]["text"].endswith("eight thirty p m")
    assert report["validators"] == {
        "forced_aligner_revision": "aligner-test-revision",
        "independent_asr_revision": "asr-test-revision",
    }
    assert aligner.paths == [validation_speech / "sample-1.wav"]
    assert independent_asr.paths == [validation_speech / "sample-1.wav"]
