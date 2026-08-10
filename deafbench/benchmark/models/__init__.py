"""Packaged model adapters for DeafBench benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from deafbench.benchmark.workspace import inspect_audio_set


MODEL_NAMES = (
    "whisper",
    "whisper-at",
    "faster-whisper",
    "distil-whisper",
    "qwen3-asr-0.6b",
)


@dataclass(frozen=True)
class ModelRunInfo:
    """Identity of the model runtime used for one prediction set."""

    name: str
    model_id: str
    revision: str | None = None
    decoding: Mapping[str, object] | None = None


def _validated_wavs(audio_dir: Path, references: Path) -> tuple[Path, ...]:
    status = inspect_audio_set(references, audio_dir)
    if not status.complete:
        raise ValueError(
            "Expected a complete audio set: "
            f"missing={list(status.missing)}; "
            f"extra={list(status.extra)}; "
            f"invalid={list(status.invalid)}"
        )
    return tuple(sorted(Path(audio_dir).glob("*.wav"), key=lambda path: path.name))


__all__ = ["MODEL_NAMES", "ModelRunInfo"]
