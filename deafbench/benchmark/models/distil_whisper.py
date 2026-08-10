"""Distil-Whisper benchmark adapter using the Faster-Whisper runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.models.faster_whisper import _run_local_whisper


DEFAULT_MODEL = "distil-large-v3"


def run_distil_whisper(
    audio_dir: Path,
    references: Path,
    output: Path,
    model_id: str = DEFAULT_MODEL,
    backend: Any | None = None,
) -> ModelRunInfo:
    """Transcribe a complete set with the distilled Whisper checkpoint."""
    return _run_local_whisper(
        audio_dir,
        references,
        output,
        "distil-whisper",
        model_id,
        {
            "beam_size": 5,
            "language": "en",
            "condition_on_previous_text": False,
        },
        backend,
    )
