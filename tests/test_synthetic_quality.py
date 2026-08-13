import hashlib
import json
import math
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from deafbench.benchmark.quality import (
    AlignmentEvidence,
    QualityRules,
    evaluate_synthetic_sample,
)


_REFERENCE = "The meeting starts at 8:30 PM."
_CRITICAL_TYPES = {"8:30 PM": "TIME"}


def _reference_hash(text: str = _REFERENCE) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _alignment(
    *,
    coverage: float = 1.0,
    entity_coverage: float = 1.0,
) -> AlignmentEvidence:
    return AlignmentEvidence(
        reference_sha256=_reference_hash(),
        audio_sha256="",
        token_coverage=coverage,
        critical_entity_coverage={"8:30 PM": entity_coverage},
        coverage_score_threshold=0.25,
        adapter="test-forced-aligner",
        adapter_revision="test-revision",
    )


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 48_000,
    channels: int = 1,
    leading: float = 0.2,
    speech: float = 0.6,
    trailing: float = 0.2,
    amplitude: float = 0.2,
) -> None:
    silence_a = np.zeros(round(sample_rate * leading), dtype=np.float32)
    times = np.arange(round(sample_rate * speech)) / sample_rate
    voiced = amplitude * np.sin(2 * math.pi * 220 * times)
    silence_b = np.zeros(round(sample_rate * trailing), dtype=np.float32)
    samples = np.concatenate((silence_a, voiced, silence_b))
    pcm = np.round(np.clip(samples, -1, 1) * 32767).astype("<i2")
    if channels == 2:
        pcm = np.repeat(pcm[:, None], 2, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _evaluate(path: Path, **overrides):
    arguments = {
        "reference_text": _REFERENCE,
        "critical_types": _CRITICAL_TYPES,
        "synthesis_reference_sha256": _reference_hash(),
        "alignment": _alignment(),
        "independent_asr_text": "The meeting starts at 8 30 p.m.",
    }
    arguments.update(overrides)
    arguments["alignment"] = replace(
        arguments["alignment"], audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return evaluate_synthetic_sample(path, **arguments)


def test_all_predeclared_quality_gates_accept_valid_sample(tmp_path: Path):
    wav = tmp_path / "valid.wav"
    _write_wav(wav)

    decision = _evaluate(wav)

    assert decision.status == "accepted"
    assert decision.disagreements == ()
    assert {gate.name for gate in decision.gates} == {
        "readable_container",
        "required_format",
        "nonempty_waveform",
        "not_truncated",
        "bounded_silence",
        "bounded_clipping",
        "plausible_duration",
        "reference_hash",
        "forced_alignment_coverage",
        "typed_critical_entity_fidelity",
    }
    assert all(gate.passed for gate in decision.gates)


@pytest.mark.parametrize(
    ("gate", "build"),
    [
        ("readable_container", lambda path: path.write_bytes(b"not a wav")),
        (
            "required_format",
            lambda path: _write_wav(path, sample_rate=16_000, channels=2),
        ),
        ("nonempty_waveform", lambda path: _write_wav(path, speech=0.0)),
        (
            "not_truncated",
            lambda path: _write_wav(path, leading=0.0, trailing=0.2),
        ),
        (
            "bounded_silence",
            lambda path: _write_wav(path, leading=1.0, trailing=0.2),
        ),
        (
            "bounded_clipping",
            lambda path: _write_wav(path, amplitude=1.0),
        ),
        (
            "plausible_duration",
            lambda path: _write_wav(path, speech=4.0),
        ),
    ],
)
def test_each_waveform_gate_quarantines_its_failure(
    tmp_path: Path,
    gate: str,
    build,
):
    wav = tmp_path / f"{gate}.wav"
    build(wav)

    decision = _evaluate(wav)

    assert decision.status == "quarantined"
    assert not decision.gate(gate).passed


def test_duration_uses_typed_spoken_words_not_display_tokens(tmp_path: Path):
    wav = tmp_path / "spoken-duration.wav"
    _write_wav(wav, speech=2.5)

    decision = _evaluate(wav)

    assert decision.gate("plausible_duration").passed


def test_reference_hash_mismatch_is_quarantined(tmp_path: Path):
    wav = tmp_path / "reference.wav"
    _write_wav(wav)

    decision = _evaluate(wav, synthesis_reference_sha256="0" * 64)

    assert not decision.gate("reference_hash").passed
    assert decision.status == "quarantined"


def test_low_forced_alignment_coverage_is_quarantined(tmp_path: Path):
    wav = tmp_path / "alignment.wav"
    _write_wav(wav)

    decision = _evaluate(wav, alignment=_alignment(coverage=0.8))

    assert not decision.gate("forced_alignment_coverage").passed
    assert decision.status == "quarantined"


def test_low_typed_entity_alignment_is_quarantined(tmp_path: Path):
    wav = tmp_path / "entity.wav"
    _write_wav(wav)

    decision = _evaluate(wav, alignment=_alignment(entity_coverage=0.5))

    assert not decision.gate("typed_critical_entity_fidelity").passed
    assert decision.status == "quarantined"


def test_typed_entity_gate_is_not_applicable_when_reference_has_no_typed_labels(
    tmp_path: Path,
):
    wav = tmp_path / "untyped.wav"
    _write_wav(wav)
    alignment = AlignmentEvidence(
        reference_sha256=_reference_hash(),
        audio_sha256="",
        token_coverage=1.0,
        critical_entity_coverage={},
        coverage_score_threshold=0.25,
        adapter="test-forced-aligner",
        adapter_revision="test-revision",
    )

    decision = _evaluate(
        wav,
        critical_types={},
        alignment=alignment,
        independent_asr_text=None,
    )

    assert decision.gate("typed_critical_entity_fidelity").passed
    assert decision.status == "accepted"


def test_independent_asr_cannot_accept_failed_alignment(tmp_path: Path):
    wav = tmp_path / "asr-cannot-accept.wav"
    _write_wav(wav)

    decision = _evaluate(
        wav,
        alignment=_alignment(coverage=0.5),
        independent_asr_text=_REFERENCE,
    )

    assert decision.status == "quarantined"
    assert not decision.gate("forced_alignment_coverage").passed


def test_alignment_coverage_from_a_weaker_score_floor_is_quarantined(tmp_path: Path):
    wav = tmp_path / "weak-coverage-floor.wav"
    _write_wav(wav)
    alignment = AlignmentEvidence(
        reference_sha256=_reference_hash(),
        audio_sha256="",
        token_coverage=1.0,
        critical_entity_coverage={"8:30 PM": 1.0},
        coverage_score_threshold=0.10,
        adapter="test-forced-aligner",
        adapter_revision="test-revision",
    )

    decision = _evaluate(wav, alignment=alignment)

    assert not decision.gate("forced_alignment_coverage").passed
    assert decision.status == "quarantined"


def test_validator_disagreement_is_recorded_and_quarantined(tmp_path: Path):
    wav = tmp_path / "disagreement.wav"
    _write_wav(wav)

    decision = _evaluate(
        wav,
        independent_asr_text="The meeting starts at 8:13 PM.",
    )

    assert decision.status == "quarantined"
    assert decision.disagreements == (
        "independent ASR missed aligned critical entity: 8:30 PM",
    )


def test_quality_rules_reject_invalid_thresholds():
    with pytest.raises(ValueError, match="silence"):
        QualityRules(min_edge_silence_ms=800, max_edge_silence_ms=700)


def test_faster_whisper_cannot_supply_forced_alignment_evidence():
    with pytest.raises(ValueError, match="not an accepted forced-alignment"):
        AlignmentEvidence(
            reference_sha256=_reference_hash(),
            audio_sha256="",
            token_coverage=1.0,
            critical_entity_coverage={"8:30 PM": 1.0},
            coverage_score_threshold=0.25,
            adapter="faster-whisper",
            adapter_revision="any",
        )


def test_predeclared_policy_matches_default_rules():
    policy_path = (
        Path(__file__).parents[1] / "benchmarks" / "synthetic-v2" / "quality-policy.json"
    )

    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    assert policy["status"] == "predeclared-before-generation"
    assert QualityRules.from_policy(policy_path) == QualityRules()
