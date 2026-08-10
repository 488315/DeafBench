"""Transactional construction of the separately versioned synthetic-v2 corpus."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, cast

import numpy as np

from .scenes import DEFAULT_SCENE_PROFILE, mix_scene, plan_scene, resample_mono
from .spoken_reference import SpokenReference, prepare_spoken_reference
from .workspace import atomic_write_jsonl, atomic_write_wav, load_reference_records


REPLACEMENT_REASONS: Mapping[str, str] = {
    "core-001": "questionable synthesis of typed TIME 2:15 PM",
    "core-009": "questionable synthesis of typed TIME 4:45 PM",
    "core-011": "questionable synthesis of typed USERNAME dev_user twenty three",
    "core-016": "questionable synthesis truncates typed DIGIT_SEQUENCE",
}


@dataclass(frozen=True)
class GeneratedSpeech:
    samples: np.ndarray
    sample_rate: int
    metadata: Mapping[str, Any]


class ReplacementGenerator(Protocol):
    def generate(
        self,
        sample_id: str,
        prepared: SpokenReference,
    ) -> GeneratedSpeech: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene_metadata(plan: Any) -> dict[str, Any]:
    return {
        "profile": plan.scene_profile,
        "seed": plan.seed,
        "sample_rate": plan.sample_rate,
        "speech": {
            "start_ms": plan.speech_start_ms,
            "end_ms": plan.speech_end_ms,
        },
        "background": {
            "profile": plan.background_profile,
            "start_ms": plan.background_start_ms,
            "end_ms": plan.background_end_ms,
            "snr_db": plan.background_snr_db,
        },
        "events": [
            {
                "label": event.label,
                "start_ms": event.start_ms,
                "end_ms": event.end_ms,
            }
            for event in plan.events
        ],
    }


def build_synthetic_v2_candidates(
    core_v1_dir: Path,
    destination: Path,
    generator: ReplacementGenerator,
    *,
    seed: int = 42,
) -> Path:
    """Copy 21 parents and synthesize exactly four candidates into a new corpus."""
    core_v1_dir = Path(core_v1_dir)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite synthetic-v2 corpus: {destination}")

    references = core_v1_dir / "references.jsonl"
    core_audio = core_v1_dir / "audio-synthetic"
    records = load_reference_records(references)
    ids = {cast(str, record["id"]) for record in records}
    missing_replacements = set(REPLACEMENT_REASONS) - ids
    if missing_replacements:
        raise ValueError(f"core-v1 is missing replacement IDs: {sorted(missing_replacements)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".synthetic-v2-", dir=destination.parent))
    try:
        shutil.copyfile(references, staging / "references.jsonl")
        output_audio = staging / "audio-synthetic"
        output_audio.mkdir()
        manifest: list[Mapping[str, Any]] = []
        for record in records:
            sample_id = cast(str, record["id"])
            parent_audio = core_audio / f"{sample_id}.wav"
            if not parent_audio.is_file():
                raise FileNotFoundError(f"missing core-v1 parent audio: {parent_audio}")
            output = output_audio / parent_audio.name
            parent_hash = _sha256(parent_audio)
            prepared = prepare_spoken_reference(
                cast(str, record["text"]),
                cast(Mapping[str, str], record["critical_types"]),
            )
            reason = REPLACEMENT_REASONS.get(sample_id)
            scene: Mapping[str, Any] | None = None
            if reason is None:
                shutil.copyfile(parent_audio, output)
                generation: Mapping[str, Any] = {
                    "engine": "copied-core-v1-parent",
                    "version": "frozen-core-v1",
                }
            else:
                generated = generator.generate(sample_id, prepared)
                speech = resample_mono(generated.samples, generated.sample_rate)
                plan = plan_scene(
                    sample_id,
                    len(speech),
                    cast(list[str], record["sounds"]),
                    seed=seed,
                    scene_profile=DEFAULT_SCENE_PROFILE,
                )
                mixed = mix_scene(speech, plan)
                atomic_write_wav(output, mixed.astype("<i2", copy=False).tobytes())
                generation = dict(generated.metadata)
                scene = _scene_metadata(plan)

            manifest.append(
                {
                    "id": sample_id,
                    "parent_sample": sample_id,
                    "parent_audio_sha256": parent_hash,
                    "audio_sha256": _sha256(output),
                    "reference_sha256": prepared.reference_sha256,
                    "replacement_reason": reason,
                    "generation": generation,
                    "scene": scene,
                }
            )
        manifest_path = staging / "generation-manifest.jsonl"
        atomic_write_jsonl(manifest_path, manifest)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "generation-manifest.jsonl"
