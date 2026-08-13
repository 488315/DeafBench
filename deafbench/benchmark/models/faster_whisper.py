"""CPU-friendly Faster-Whisper benchmark adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from deafbench.benchmark.models import ModelRunInfo, _validated_wavs
from deafbench.benchmark.workspace import atomic_write_jsonl


DEFAULT_MODEL = "small.en"
DEFAULT_MODEL_REVISION = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"


def _load_backend() -> Any:
    try:
        import faster_whisper
    except ModuleNotFoundError as exc:
        if exc.name != "faster_whisper":
            raise
        raise RuntimeError(
            "Faster-Whisper is not installed. Run: "
            "python -m pip install -U faster-whisper"
        ) from exc
    return faster_whisper


def _run_local_whisper(
    audio_dir: Path,
    references: Path,
    output: Path,
    model_name: str,
    model_id: str,
    transcribe_options: Mapping[str, object],
    backend: Any | None = None,
    model_revision: str | None = None,
) -> ModelRunInfo:
    wav_paths = _validated_wavs(audio_dir, references)
    runtime = _load_backend() if backend is None else backend
    model_options: dict[str, object] = {
        "device": "cpu",
        "compute_type": "int8",
    }
    if model_revision is not None:
        model_options["revision"] = model_revision
    model = runtime.WhisperModel(model_id, **model_options)
    records: list[Mapping[str, str]] = []

    for wav_path in wav_paths:
        segments, _info = model.transcribe(
            str(wav_path),
            **transcribe_options,
        )
        text_parts: list[str] = []
        for segment in segments:
            text = getattr(segment, "text", None)
            if not isinstance(text, str):
                raise ValueError(
                    f"Invalid {model_name} segment for {wav_path.name}: "
                    "expected text to be a string"
                )
            text_parts.append(text)
        records.append({"id": wav_path.stem, "text": "".join(text_parts)})

    atomic_write_jsonl(output, records)
    decoding = None
    if model_revision is not None:
        decoding = {
            key: value for key, value in model_options.items() if key != "revision"
        }
        decoding.update(transcribe_options)
    return ModelRunInfo(
        model_name,
        model_id,
        revision=model_revision,
        decoding=decoding,
    )


def run_faster_whisper(
    audio_dir: Path,
    references: Path,
    output: Path,
    model_id: str = DEFAULT_MODEL,
    backend: Any | None = None,
) -> ModelRunInfo:
    """Transcribe a complete audio set with CPU INT8 Faster-Whisper."""
    return _run_local_whisper(
        audio_dir,
        references,
        output,
        "faster-whisper",
        model_id,
        {"beam_size": 5, "language": "en"},
        backend,
        DEFAULT_MODEL_REVISION,
    )
