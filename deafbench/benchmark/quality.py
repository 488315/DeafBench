"""Fail-closed admission policy for synthetic benchmark audio."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from deafbench.critical_entities import ENTITY_TYPES, canonical_contains

from .quality_audio import inspect_audio
from .spoken_reference import prepare_spoken_reference


@dataclass(frozen=True)
class QualityRules:
    """Predeclared, model-independent synthetic audio thresholds."""

    sample_rate: int = 48_000
    channels: int = 1
    min_edge_silence_ms: int = 100
    max_edge_silence_ms: int = 700
    max_clipping_fraction: float = 0.001
    min_seconds_per_word: float = 0.08
    max_seconds_per_word: float = 0.45
    min_alignment_coverage: float = 0.95
    min_alignment_token_score: float = 0.25
    min_entity_coverage: float = 0.95
    analysis_frame_ms: int = 10

    def __post_init__(self) -> None:
        if not 0 <= self.min_edge_silence_ms <= self.max_edge_silence_ms:
            raise ValueError("minimum silence must not exceed maximum silence")
        if self.sample_rate <= 0 or self.channels <= 0 or self.analysis_frame_ms <= 0:
            raise ValueError("audio format and analysis frame must be positive")
        if not 0 <= self.max_clipping_fraction <= 1:
            raise ValueError("clipping fraction must be between zero and one")
        if not 0 < self.min_seconds_per_word <= self.max_seconds_per_word:
            raise ValueError("duration bounds must be positive and ordered")
        if not 0 <= self.min_alignment_coverage <= 1:
            raise ValueError("alignment coverage must be between zero and one")
        if not 0 <= self.min_alignment_token_score <= 1:
            raise ValueError("alignment token score must be between zero and one")
        if not 0 <= self.min_entity_coverage <= 1:
            raise ValueError("entity coverage must be between zero and one")

    @classmethod
    def from_policy(cls, path: Path) -> "QualityRules":
        """Load the versioned thresholds from a corpus policy record."""
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise ValueError("unsupported synthetic quality policy schema")
        return cls(**document["rules"])


@dataclass(frozen=True)
class AlignmentEvidence:
    """Evidence emitted by a reference-conditioned forced aligner."""

    reference_sha256: str
    token_coverage: float
    critical_entity_coverage: Mapping[str, float]
    coverage_score_threshold: float
    adapter: str
    adapter_revision: str

    def __post_init__(self) -> None:
        if "faster-whisper" in self.adapter.casefold():
            raise ValueError("faster-whisper is not an accepted forced-alignment adapter")
        if not 0 <= self.token_coverage <= 1:
            raise ValueError("token coverage must be between zero and one")
        if not 0 <= self.coverage_score_threshold <= 1:
            raise ValueError("coverage score threshold must be between zero and one")
        if any(not 0 <= value <= 1 for value in self.critical_entity_coverage.values()):
            raise ValueError("entity coverage must be between zero and one")
        object.__setattr__(
            self,
            "critical_entity_coverage",
            MappingProxyType(dict(self.critical_entity_coverage)),
        )


@dataclass(frozen=True)
class QualityGate:
    name: str
    passed: bool
    evidence: str


@dataclass(frozen=True)
class QualityDecision:
    status: str
    gates: tuple[QualityGate, ...]
    disagreements: tuple[str, ...]

    def gate(self, name: str) -> QualityGate:
        for gate in self.gates:
            if gate.name == name:
                return gate
        raise KeyError(name)


def _gate(name: str, passed: bool, evidence: str) -> QualityGate:
    return QualityGate(name=name, passed=bool(passed), evidence=evidence)


def _reference_sha256(reference_text: str) -> str:
    return hashlib.sha256(reference_text.encode("utf-8")).hexdigest()


def _duration_gate(
    duration: float,
    reference_text: str,
    critical_types: Mapping[str, str],
    rules: QualityRules,
) -> bool:
    prepared = prepare_spoken_reference(reference_text, critical_types)
    word_count = max(1, len(prepared.words))
    seconds_per_word = duration / word_count
    return rules.min_seconds_per_word <= seconds_per_word <= rules.max_seconds_per_word


def evaluate_synthetic_sample(
    audio_path: Path,
    *,
    reference_text: str,
    critical_types: Mapping[str, str],
    synthesis_reference_sha256: str,
    alignment: AlignmentEvidence,
    independent_asr_text: str | None = None,
    rules: QualityRules = QualityRules(),
) -> QualityDecision:
    """Evaluate all predeclared gates; ASR can only flag disagreement."""
    unknown_types = set(critical_types.values()) - ENTITY_TYPES
    if unknown_types:
        raise ValueError(f"unknown critical entity types: {sorted(unknown_types)}")

    audio = inspect_audio(audio_path, frame_ms=rules.analysis_frame_ms)
    expected_hash = _reference_sha256(reference_text)
    readable = audio.readable and audio.error is None
    correct_format = (
        readable
        and audio.sample_rate == rules.sample_rate
        and audio.channels == rules.channels
    )
    has_edges = (
        audio.leading_silence_ms is not None
        and audio.trailing_silence_ms is not None
    )
    not_truncated = (
        has_edges
        and audio.leading_silence_ms >= rules.min_edge_silence_ms
        and audio.trailing_silence_ms >= rules.min_edge_silence_ms
    )
    bounded_silence = (
        has_edges
        and audio.leading_silence_ms <= rules.max_edge_silence_ms
        and audio.trailing_silence_ms <= rules.max_edge_silence_ms
    )
    entity_coverage = {
        term: alignment.critical_entity_coverage.get(term, 0.0)
        for term in critical_types
    }

    gates = (
        _gate("readable_container", readable, audio.error or "PCM16 WAV readable"),
        _gate(
            "required_format",
            correct_format,
            f"{audio.sample_rate} Hz, {audio.channels} channel(s), {audio.container}/{audio.subtype}",
        ),
        _gate("nonempty_waveform", audio.has_signal, f"peak={audio.peak_amplitude:.6f}"),
        _gate(
            "not_truncated",
            not_truncated,
            f"leading={audio.leading_silence_ms}, trailing={audio.trailing_silence_ms} ms",
        ),
        _gate(
            "bounded_silence",
            bounded_silence,
            f"leading={audio.leading_silence_ms}, trailing={audio.trailing_silence_ms} ms",
        ),
        _gate(
            "bounded_clipping",
            audio.clipping_fraction <= rules.max_clipping_fraction,
            f"fraction={audio.clipping_fraction:.8f}",
        ),
        _gate(
            "plausible_duration",
            _duration_gate(
                audio.duration_seconds,
                reference_text,
                critical_types,
                rules,
            ),
            f"duration={audio.duration_seconds:.3f} seconds",
        ),
        _gate(
            "reference_hash",
            synthesis_reference_sha256 == expected_hash
            and alignment.reference_sha256 == expected_hash,
            f"expected={expected_hash}",
        ),
        _gate(
            "forced_alignment_coverage",
            alignment.token_coverage >= rules.min_alignment_coverage
            and alignment.coverage_score_threshold >= rules.min_alignment_token_score,
            (
                f"coverage={alignment.token_coverage:.6f}, "
                f"score_floor={alignment.coverage_score_threshold:.6f}"
            ),
        ),
        _gate(
            "typed_critical_entity_fidelity",
            all(value >= rules.min_entity_coverage for value in entity_coverage.values()),
            (
                f"coverage={entity_coverage}"
                if entity_coverage
                else "not applicable: reference has no typed entity labels"
            ),
        ),
    )

    disagreements: list[str] = []
    if independent_asr_text is not None:
        for term, entity_type in critical_types.items():
            aligned = entity_coverage[term] >= rules.min_entity_coverage
            transcribed = canonical_contains(term, independent_asr_text, entity_type)
            if aligned and not transcribed:
                disagreements.append(
                    f"independent ASR missed aligned critical entity: {term}"
                )
            elif transcribed and not aligned:
                disagreements.append(
                    f"independent ASR found unaligned critical entity: {term}"
                )

    accepted = all(gate.passed for gate in gates) and not disagreements
    return QualityDecision(
        status="accepted" if accepted else "quarantined",
        gates=gates,
        disagreements=tuple(disagreements),
    )
