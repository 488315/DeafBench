"""Orchestration and evidence records for synthetic-v2 replacement admission."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from .quality import AlignmentEvidence, QualityRules, evaluate_synthetic_sample
from .spoken_reference import SpokenReference, prepare_spoken_reference
from .synthetic_v2_corpus import REPLACEMENT_REASONS
from .workspace import atomic_write_json, load_reference_records


class ForcedAligner(Protocol):
    adapter_revision: str

    def align(
        self,
        audio_path: Path,
        prepared: SpokenReference,
        *,
        score_threshold: float,
    ) -> AlignmentEvidence: ...


class IndependentASR(Protocol):
    adapter_revision: str

    def transcribe(self, audio_path: Path) -> str: ...


def _jsonl_by_id(path: Path) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"invalid validation input record: {path}")
        sample_id = value.get("id")
        if not isinstance(sample_id, str):
            raise ValueError(f"invalid validation input record: {path}")
        if sample_id in records:
            raise ValueError(f"duplicate validation input ID: {sample_id}")
        records[sample_id] = value
    return records


def validate_replacement_candidates(
    corpus_dir: Path,
    policy_path: Path,
    aligner: ForcedAligner,
    independent_asr: IndependentASR,
    *,
    sample_ids: Sequence[str] = tuple(REPLACEMENT_REASONS),
) -> dict[str, Any]:
    """Run every admission gate for the regenerated sample allowlist."""
    corpus_dir = Path(corpus_dir)
    policy_path = Path(policy_path)
    rules = QualityRules.from_policy(policy_path)
    references = {
        cast(str, record["id"]): record
        for record in load_reference_records(corpus_dir / "references.jsonl")
    }
    generation = _jsonl_by_id(corpus_dir / "generation-manifest.jsonl")
    samples: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        if sample_id not in references or sample_id not in generation:
            raise ValueError(f"missing replacement validation input: {sample_id}")
        reference = references[sample_id]
        prepared = prepare_spoken_reference(
            cast(str, reference["text"]),
            cast(Mapping[str, str], reference["critical_types"]),
        )
        audio_path = corpus_dir / "audio-synthetic" / f"{sample_id}.wav"
        alignment = aligner.align(
            audio_path,
            prepared,
            score_threshold=rules.min_alignment_token_score,
        )
        asr_text = independent_asr.transcribe(audio_path)
        decision = evaluate_synthetic_sample(
            audio_path,
            reference_text=cast(str, reference["text"]),
            critical_types=cast(Mapping[str, str], reference["critical_types"]),
            synthesis_reference_sha256=cast(
                str, generation[sample_id]["reference_sha256"]
            ),
            alignment=alignment,
            independent_asr_text=asr_text,
            rules=rules,
        )
        samples.append(
            {
                "id": sample_id,
                "status": decision.status,
                "gates": [
                    {
                        "name": gate.name,
                        "passed": gate.passed,
                        "evidence": gate.evidence,
                    }
                    for gate in decision.gates
                ],
                "disagreements": list(decision.disagreements),
                "forced_alignment": {
                    "audio_sha256": alignment.audio_sha256,
                    "adapter": alignment.adapter,
                    "adapter_revision": alignment.adapter_revision,
                    "coverage_score_threshold": alignment.coverage_score_threshold,
                    "token_coverage": alignment.token_coverage,
                    "critical_entity_coverage": dict(
                        alignment.critical_entity_coverage
                    ),
                },
                "independent_asr": {
                    "audio_sha256": alignment.audio_sha256,
                    "adapter": "torchaudio-WAV2VEC2_ASR_BASE_960H",
                    "adapter_revision": independent_asr.adapter_revision,
                    "text": asr_text,
                    "role": "supporting evidence only",
                },
            }
        )
    return {
        "schema_version": 1,
        "scope": "regenerated-replacements-only",
        "inherited_samples": "frozen core-v1 bytes; not admitted under v2 gates",
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "validators": {
            "forced_aligner_revision": aligner.adapter_revision,
            "independent_asr_revision": independent_asr.adapter_revision,
        },
        "samples": samples,
    }


def write_quality_report(path: Path, report: Mapping[str, Any]) -> None:
    """Persist a deterministic validation report beside generated evidence."""
    atomic_write_json(path, report)
