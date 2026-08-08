"""Testable core helpers for the DeafBench dataset recorder."""

from __future__ import annotations

import json
import os
import tempfile
import wave
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_DEVICE_NEEDLE = "Voicemeeter Out B3"
SOUND_EVENT_GAP_MS = 250
SUPPORTED_SOUND_EVENTS = (
    "[alarm]",
    "[door closes]",
    "[phone rings]",
    "[knock]",
    "[error notification]",
    "[siren]",
)


def _is_safe_sample_id(sample_id: object) -> bool:
    if not isinstance(sample_id, str) or not sample_id.strip():
        return False
    if sample_id != sample_id.strip():
        return False
    if sample_id in {".", ".."}:
        return False
    return not any(separator in sample_id for separator in ("/", "\\", ":"))


def load_prompts(path: Path) -> list[dict[str, Any]]:
    """Load and validate recorder prompts from a JSONL file."""
    prompts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Invalid record on line {line_number}: expected an object")

            sample_id = record.get("id")
            if not _is_safe_sample_id(sample_id):
                raise ValueError(f"Invalid id on line {line_number}: expected a safe file name")

            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Invalid text on line {line_number}: expected a string")

            sounds = record.get("sounds", [])
            if not isinstance(sounds, list) or not all(isinstance(label, str) for label in sounds):
                raise ValueError(f"Invalid sounds on line {line_number}: expected a list of strings")
            unsupported = [label for label in sounds if label not in SUPPORTED_SOUND_EVENTS]
            if unsupported:
                raise ValueError(
                    f"Unsupported sound event on line {line_number}: {unsupported[0]}"
                )

            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")

            seen_ids.add(sample_id)
            prompts.append(record)

    if not prompts:
        raise ValueError("No prompts found in references file")

    return prompts


def output_path(audio_dir: Path, sample_id: str) -> Path:
    """Return the expected WAV path for a sample ID."""
    if not _is_safe_sample_id(sample_id):
        raise ValueError("Invalid sample ID: expected a safe file name")
    return Path(audio_dir) / f"{sample_id}.wav"


def is_recorded(audio_dir: Path, sample_id: str) -> bool:
    """Return whether a sample already has a WAV file."""
    return output_path(audio_dir, sample_id).is_file()


def next_unrecorded_index(
    prompts: Sequence[Mapping[str, Any]],
    audio_dir: Path,
    current_index: int,
) -> int | None:
    """Find the next unrecorded sample after the current index without wrapping."""
    for index in range(current_index + 1, len(prompts)):
        sample_id = str(prompts[index]["id"])
        if not is_recorded(audio_dir, sample_id):
            return index
    return None


def find_preferred_input_device(
    devices: Iterable[Mapping[str, Any]],
    needle: str = DEFAULT_DEVICE_NEEDLE,
) -> int | None:
    """Return the first input-capable device index whose name contains the needle."""
    normalized_needle = needle.casefold()
    for index, device in enumerate(devices):
        name = str(device.get("name", ""))
        max_input_channels = int(device.get("max_input_channels", 0) or 0)
        if max_input_channels > 0 and normalized_needle in name.casefold():
            return index
    return None


def downmix_to_mono(samples: np.ndarray) -> np.ndarray:
    """Convert integer PCM samples to a two-dimensional int16 mono array."""
    data = np.asarray(samples)

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("Audio samples must have shape (frames,) or (frames, channels)")

    if data.shape[1] == 1:
        mono = data[:, :1]
    else:
        mono = np.rint(data.astype(np.float64).mean(axis=1, keepdims=True))

    clipped = np.clip(mono, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return clipped.astype(np.int16, copy=False)


def _silence(duration_seconds: float) -> np.ndarray:
    frames = max(0, int(round(DEFAULT_SAMPLE_RATE * duration_seconds)))
    return np.zeros((frames, 1), dtype=np.int16)


def _pcm(signal: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype(np.int16).reshape(-1, 1)


def _tone(frequency: float, duration_seconds: float, amplitude: float = 0.35) -> np.ndarray:
    frames = max(1, int(round(DEFAULT_SAMPLE_RATE * duration_seconds)))
    time_axis = np.arange(frames, dtype=np.float64) / DEFAULT_SAMPLE_RATE
    signal = amplitude * np.sin(2.0 * np.pi * frequency * time_axis)

    fade_frames = min(frames // 2, max(1, DEFAULT_SAMPLE_RATE // 100))
    fade = np.linspace(0.0, 1.0, fade_frames, endpoint=True)
    signal[:fade_frames] *= fade
    signal[-fade_frames:] *= fade[::-1]
    return _pcm(signal)


def _join(parts: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(parts), axis=0) if parts else np.empty((0, 1), dtype=np.int16)


def _alarm() -> np.ndarray:
    parts: list[np.ndarray] = []
    for frequency in (880.0, 660.0, 880.0, 660.0):
        parts.append(_tone(frequency, 0.22, 0.34))
        parts.append(_silence(0.08))
    return _join(parts[:-1])


def _door_closes() -> np.ndarray:
    frames = int(round(DEFAULT_SAMPLE_RATE * 0.45))
    time_axis = np.arange(frames, dtype=np.float64) / DEFAULT_SAMPLE_RATE
    envelope = np.exp(-11.0 * time_axis)
    signal = envelope * (
        0.52 * np.sin(2.0 * np.pi * 78.0 * time_axis)
        + 0.20 * np.sin(2.0 * np.pi * 145.0 * time_axis)
        + 0.08 * np.sin(2.0 * np.pi * 900.0 * time_axis)
    )
    return _pcm(signal)


def _phone_rings() -> np.ndarray:
    frames = int(round(DEFAULT_SAMPLE_RATE * 0.38))
    time_axis = np.arange(frames, dtype=np.float64) / DEFAULT_SAMPLE_RATE
    ring = _pcm(
        0.22 * np.sin(2.0 * np.pi * 440.0 * time_axis)
        + 0.22 * np.sin(2.0 * np.pi * 480.0 * time_axis)
    )
    return _join([ring, _silence(0.18), ring])


def _knock() -> np.ndarray:
    frames = int(round(DEFAULT_SAMPLE_RATE * 0.09))
    time_axis = np.arange(frames, dtype=np.float64) / DEFAULT_SAMPLE_RATE
    envelope = np.exp(-42.0 * time_axis)
    knock = _pcm(
        envelope
        * (
            0.58 * np.sin(2.0 * np.pi * 115.0 * time_axis)
            + 0.18 * np.sin(2.0 * np.pi * 720.0 * time_axis)
        )
    )
    return _join([knock, _silence(0.11), knock, _silence(0.11), knock])


def _error_notification() -> np.ndarray:
    return _join([_tone(760.0, 0.16, 0.30), _silence(0.07), _tone(520.0, 0.28, 0.34)])


def _siren() -> np.ndarray:
    frames = int(round(DEFAULT_SAMPLE_RATE * 1.2))
    time_axis = np.arange(frames, dtype=np.float64) / DEFAULT_SAMPLE_RATE
    frequency = 800.0 + 240.0 * np.sin(2.0 * np.pi * 1.35 * time_axis)
    phase = 2.0 * np.pi * np.cumsum(frequency) / DEFAULT_SAMPLE_RATE
    return _pcm(0.30 * np.sin(phase))


_SOUND_EVENT_FACTORIES = {
    "[alarm]": _alarm,
    "[door closes]": _door_closes,
    "[phone rings]": _phone_rings,
    "[knock]": _knock,
    "[error notification]": _error_notification,
    "[siren]": _siren,
}


def synthesize_sound_event(label: str) -> np.ndarray:
    """Generate the deterministic 48 kHz mono PCM cue for one sound label."""
    try:
        factory = _SOUND_EVENT_FACTORIES[label]
    except KeyError as exc:
        raise ValueError(f"Unsupported sound event: {label}") from exc
    return factory()


def append_sound_events(samples: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Append synthetic acoustic events after speech in the reference label order."""
    mono = downmix_to_mono(samples)
    if not labels:
        return mono

    gap = _silence(SOUND_EVENT_GAP_MS / 1000.0)
    parts: list[np.ndarray] = [mono]
    for label in labels:
        parts.append(gap)
        parts.append(synthesize_sound_event(label))
    return _join(parts)


def atomic_write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """Atomically write standardized 16-bit mono PCM WAV audio."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mono = downmix_to_mono(samples)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}-",
            suffix=".wav.tmp",
            dir=destination.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(mono.tobytes(order="C"))

        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
