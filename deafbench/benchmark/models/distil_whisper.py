"""Distil-Whisper benchmark adapter using the Faster-Whisper runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.models.faster_whisper import _run_local_whisper


DEFAULT_MODEL = "Systran/faster-distil-whisper-large-v3"
DEFAULT_MODEL_REVISION = "c3058b475261292e64a0412df1d2681c06260fab"


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
        DEFAULT_MODEL_REVISION,
    )
