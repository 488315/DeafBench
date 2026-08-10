"""Installed Whisper adapter for Model A transcript predictions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from deafbench.benchmark.models import ModelRunInfo, _validated_wavs
from deafbench.benchmark.workspace import atomic_write_jsonl


def _load_backend() -> Any:
    try:
        import whisper
    except ModuleNotFoundError as exc:
        if exc.name != "whisper":
            raise
        raise RuntimeError(
            "Whisper is not installed. Run: "
            "python -m pip install -U openai-whisper"
        ) from exc
    return whisper


def run_whisper(
    audio_dir: Path,
    references: Path,
    output: Path,
    model_id: str = "turbo",
    backend: Any | None = None,
) -> ModelRunInfo:
    """Transcribe one complete audio set into Model A JSONL records."""
    wav_paths = _validated_wavs(audio_dir, references)
    runtime = _load_backend() if backend is None else backend
    model = runtime.load_model(model_id)
    records: list[Mapping[str, Any]] = []

    for wav_path in wav_paths:
        result = model.transcribe(
            str(wav_path),
            language="en",
            task="transcribe",
            verbose=False,
        )
        if not isinstance(result, Mapping):
            raise ValueError(
                f"Invalid Whisper result for {wav_path.name}: "
                "expected a mapping"
            )
        text = result.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"Invalid Whisper transcript for {wav_path.name}: "
                "expected a string"
            )
        records.append({"id": wav_path.stem, "text": text})

    atomic_write_jsonl(output, records)
    return ModelRunInfo("whisper", model_id)
