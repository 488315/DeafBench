"""Distil-Whisper benchmark adapter using the Faster-Whisper runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from deafbench.benchmark.models import ModelRunInfo, _validated_wavs
from deafbench.benchmark.models.faster_whisper import _load_backend
from deafbench.benchmark.workspace import atomic_write_jsonl


DEFAULT_MODEL = "distil-large-v3"


def run_distil_whisper(
    audio_dir: Path,
    references: Path,
    output: Path,
    model_id: str = DEFAULT_MODEL,
    backend: Any | None = None,
) -> ModelRunInfo:
    """Transcribe a complete set with the distilled Whisper checkpoint."""
    wav_paths = _validated_wavs(audio_dir, references)
    runtime = _load_backend() if backend is None else backend
    model = runtime.WhisperModel(
        model_id,
        device="cpu",
        compute_type="int8",
    )
    records: list[Mapping[str, str]] = []

    for wav_path in wav_paths:
        segments, _info = model.transcribe(
            str(wav_path),
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
        )
        text_parts: list[str] = []
        for segment in segments:
            text = getattr(segment, "text", None)
            if not isinstance(text, str):
                raise ValueError(
                    f"Invalid Distil-Whisper segment for {wav_path.name}: "
                    "expected text to be a string"
                )
            text_parts.append(text)
        records.append({"id": wav_path.stem, "text": "".join(text_parts)})

    atomic_write_jsonl(output, records)
    return ModelRunInfo("distil-whisper", model_id)
