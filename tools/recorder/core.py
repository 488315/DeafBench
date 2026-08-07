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
