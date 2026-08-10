"""WhisperSpeech generation and transactional synthetic audio sets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import wave
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import numpy as np

from deafbench.benchmark.scenes import (
    DEFAULT_SCENE_PROFILE,
    ScenePlan,
    mix_scene,
    plan_scene,
    resample_mono,
)
from deafbench.benchmark.workspace import (
    STANDARD_SAMPLE_RATE,
    atomic_write_jsonl,
    atomic_write_wav,
    inspect_audio_set,
    load_reference_records,
)


@dataclass(frozen=True)
class SpeechAudio:
    """Speech samples and the actual rate reported by the TTS backend."""

    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class TTSInfo:
    """Persisted identity of the speech generator used for one set."""

    engine: str
    version: str


SpeechGenerator = Callable[[str], SpeechAudio]

_MANIFEST_FIELDS = {
    "id",
    "wav",
    "fingerprint",
    "scene_profile",
    "seed",
    "sample_rate",
    "tts",
    "speech",
    "background",
    "events",
}


def create_whisperspeech_generator() -> tuple[SpeechGenerator, TTSInfo]:
    """Create one reusable WhisperSpeech pipeline and its provenance."""
    try:
        from whisperspeech.pipeline import Pipeline
    except ModuleNotFoundError as exc:
        if exc.name != "whisperspeech":
            raise
        raise RuntimeError(
            "WhisperSpeech is not installed. Run: "
            'python -m pip install "deafbench[benchmark]"'
        ) from exc

    pipeline = Pipeline()
    try:
        version = metadata.version("WhisperSpeech")
    except metadata.PackageNotFoundError:
        version = "unknown"

    def generate(text: str) -> SpeechAudio:
        audio = pipeline.generate(text, lang="en")
        if hasattr(audio, "detach"):
            audio = audio.detach()
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        elif samples.ndim == 2:
            samples = samples.T
        else:
            raise RuntimeError("WhisperSpeech returned an invalid audio shape")
        return SpeechAudio(samples, 24_000)

    return generate, TTSInfo("whisperspeech", version)


def generation_fingerprint(
    references: Path,
    scene_profile: str,
    seed: int,
    tts_info: TTSInfo,
) -> str:
    """Hash all inputs that determine a generated synthetic set."""
    value = {
        "references_sha256": hashlib.sha256(
            Path(references).read_bytes()
        ).hexdigest(),
        "scene_profile": scene_profile,
        "seed": seed,
        "tts_engine": tts_info.engine,
        "tts_version": tts_info.version,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid manifest JSON on line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError("Manifest records must be objects")
            records.append(value)
    if not records:
        raise ValueError("Synthetic manifest is empty")
    return tuple(records)


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_interval(value: object, limit: int) -> bool:
    if not isinstance(value, dict) or set(value) != {"start_ms", "end_ms"}:
        return False
    start = value.get("start_ms")
    end = value.get("end_ms")
    return (
        _integer(start)
        and _integer(end)
        and 0 <= cast(int, start) <= cast(int, end) <= limit
    )


def _valid_timing(record: Mapping[str, Any]) -> bool:
    background = record.get("background")
    if not isinstance(background, dict) or set(background) != {
        "profile",
        "start_ms",
        "end_ms",
        "snr_db",
    }:
        return False
    profile = background.get("profile")
    start = background.get("start_ms")
    end = background.get("end_ms")
    snr = background.get("snr_db")
    if not (
        isinstance(profile, str)
        and profile
        and _integer(start)
        and _integer(end)
        and start == 0
        and cast(int, end) > 0
        and isinstance(snr, (int, float))
        and not isinstance(snr, bool)
        and math.isfinite(float(snr))
    ):
        return False
    limit = cast(int, end)
    if not _valid_interval(record.get("speech"), limit):
        return False
    events = record.get("events")
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict) or set(event) != {
            "label",
            "start_ms",
            "end_ms",
        }:
            return False
        if not isinstance(event.get("label"), str) or not event["label"]:
            return False
        if not _valid_interval(
            {"start_ms": event.get("start_ms"), "end_ms": event.get("end_ms")},
            limit,
        ):
            return False
    return True


def _scene_metadata_matches(
    reference: Mapping[str, Any],
    record: Mapping[str, Any],
    scene_profile: str,
    seed: int,
) -> bool:
    speech = cast(Mapping[str, Any], record["speech"])
    duration_ms = cast(int, speech["end_ms"]) - cast(
        int,
        speech["start_ms"],
    )
    speech_frames = duration_ms * STANDARD_SAMPLE_RATE // 1_000
    expected = plan_scene(
        cast(str, reference["id"]),
        speech_frames,
        cast(list[str], reference["sounds"]),
        seed=seed,
        scene_profile=scene_profile,
    )
    expected_record = _manifest_record(
        expected.sample_id,
        expected,
        "",
        TTSInfo("", ""),
    )
    return all(
        record[field] == expected_record[field]
        for field in ("speech", "background", "events")
    )


def _wav_frame_count(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        frame_count = handle.getnframes()
        frame_width = handle.getnchannels() * handle.getsampwidth()
        payload = handle.readframes(frame_count)
    if len(payload) != frame_count * frame_width:
        raise ValueError("Synthetic WAV payload is truncated")
    return frame_count


def _validate_synthetic_set(
    audio_dir: Path,
    references: Path,
    scene_profile: str,
    seed: int,
) -> None:
    if not _integer(seed):
        raise ValueError("Synthetic seed must be an integer")
    reference_records = load_reference_records(references)
    status = inspect_audio_set(references, audio_dir)
    if not status.complete:
        raise ValueError("Synthetic WAV set is incomplete")

    manifest_records = _read_manifest(audio_dir / "manifest.jsonl")
    if len(manifest_records) != len(reference_records):
        raise ValueError("Synthetic manifest does not match references")

    first_tts = manifest_records[0].get("tts")
    if not isinstance(first_tts, dict) or set(first_tts) != {
        "engine",
        "version",
    }:
        raise ValueError("Invalid TTS provenance")
    engine = first_tts.get("engine")
    version = first_tts.get("version")
    if not (
        isinstance(engine, str)
        and engine
        and isinstance(version, str)
        and version
    ):
        raise ValueError("Invalid TTS provenance")
    expected_fingerprint = generation_fingerprint(
        references,
        scene_profile,
        seed,
        TTSInfo(engine, version),
    )

    for reference, record in zip(
        reference_records,
        manifest_records,
        strict=True,
    ):
        sample_id = cast(str, reference["id"])
        if set(record) != _MANIFEST_FIELDS:
            raise ValueError("Invalid synthetic manifest fields")
        if record.get("id") != sample_id or record.get("wav") != f"{sample_id}.wav":
            raise ValueError("Synthetic manifest WAV does not match reference")
        if (
            record.get("fingerprint") != expected_fingerprint
            or record.get("scene_profile") != scene_profile
            or not _integer(record.get("seed"))
            or record.get("seed") != seed
            or record.get("sample_rate") != STANDARD_SAMPLE_RATE
            or record.get("tts") != first_tts
        ):
            raise ValueError("Synthetic generation settings are inconsistent")
        if not _valid_timing(record):
            raise ValueError("Invalid synthetic timing metadata")
        if not _scene_metadata_matches(
            reference,
            record,
            scene_profile,
            seed,
        ):
            raise ValueError("Synthetic scene metadata does not match inputs")

        background = cast(Mapping[str, Any], record["background"])
        end_ms = cast(int, background["end_ms"])
        if end_ms * STANDARD_SAMPLE_RATE % 1_000:
            raise ValueError("Synthetic duration is not frame-aligned")
        expected_frames = end_ms * STANDARD_SAMPLE_RATE // 1_000
        if _wav_frame_count(audio_dir / f"{sample_id}.wav") != expected_frames:
            raise ValueError("Synthetic WAV length does not match manifest")


def synthetic_set_is_current(
    audio_dir: Path,
    references: Path,
    scene_profile: str,
    seed: int,
) -> bool:
    """Return whether a complete set matches the requested persisted inputs."""
    try:
        _validate_synthetic_set(
            Path(audio_dir),
            Path(references),
            scene_profile,
            seed,
        )
    except (OSError, ValueError, TypeError, KeyError, wave.Error):
        return False
    return True


def _manifest_record(
    sample_id: str,
    plan: ScenePlan,
    fingerprint: str,
    tts_info: TTSInfo,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "wav": f"{sample_id}.wav",
        "fingerprint": fingerprint,
        "scene_profile": plan.scene_profile,
        "seed": plan.seed,
        "sample_rate": plan.sample_rate,
        "tts": {"engine": tts_info.engine, "version": tts_info.version},
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


def _promote_directory(staging: Path, audio_dir: Path) -> None:
    backup = audio_dir.with_name(f".{audio_dir.name}-backup")
    if backup.exists():
        if audio_dir.exists():
            shutil.rmtree(backup)
        else:
            os.replace(backup, audio_dir)
    if audio_dir.exists():
        os.replace(audio_dir, backup)
    try:
        os.replace(staging, audio_dir)
    except Exception:
        if backup.exists() and not audio_dir.exists():
            os.replace(backup, audio_dir)
        raise
    else:
        if backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError:
                pass


def generate_synthetic_set(
    references: Path,
    audio_dir: Path,
    speech_generator: SpeechGenerator,
    tts_info: TTSInfo,
    scene_profile: str = DEFAULT_SCENE_PROFILE,
    seed: int = 42,
) -> Path:
    """Generate, validate, and atomically replace one complete audio set."""
    if not _integer(seed):
        raise ValueError("Synthetic seed must be an integer")
    references = Path(references)
    audio_dir = Path(audio_dir)
    records = load_reference_records(references)
    fingerprint = generation_fingerprint(
        references,
        scene_profile,
        seed,
        tts_info,
    )
    audio_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".audio-synthetic-",
            dir=audio_dir.parent,
        )
    )
    promoted = False
    try:
        manifest_records: list[Mapping[str, Any]] = []
        for reference in records:
            sample_id = cast(str, reference["id"])
            speech = speech_generator(cast(str, reference["text"]))
            resampled = resample_mono(speech.samples, speech.sample_rate)
            plan = plan_scene(
                sample_id,
                len(resampled),
                cast(list[str], reference["sounds"]),
                seed=seed,
                scene_profile=scene_profile,
            )
            mixed = mix_scene(resampled, plan)
            pcm = mixed.astype("<i2", copy=False).tobytes(order="C")
            atomic_write_wav(staging / f"{sample_id}.wav", pcm)
            manifest_records.append(
                _manifest_record(sample_id, plan, fingerprint, tts_info)
            )

        atomic_write_jsonl(staging / "manifest.jsonl", manifest_records)
        _validate_synthetic_set(staging, references, scene_profile, seed)
        _promote_directory(staging, audio_dir)
        promoted = True
    finally:
        if not promoted and staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                pass
    return audio_dir / "manifest.jsonl"
