"""Installed Whisper-AT adapter for structured Model B predictions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from deafbench.benchmark.models import ModelRunInfo, _validated_wavs
from deafbench.benchmark.workspace import atomic_write_jsonl


DEFAULT_MODEL = "medium.en"
DEFAULT_AT_TIME_RES = 10.0
DEFAULT_TOP_K = 5
DEFAULT_P_THRESHOLD = -1.0
AUDIOSET_CLASS_COUNT = 527

AUDIOSET_TO_DEAFBENCH = {
    "Alarm": "[alarm]",
    "Alarm clock": "[alarm]",
    "Smoke detector, smoke alarm": "[alarm]",
    "Fire alarm": "[alarm]",
    "Car alarm": "[alarm]",
    "Slam": "[door closes]",
    "Telephone bell ringing": "[phone rings]",
    "Ringtone": "[phone rings]",
    "Knock": "[knock]",
    "Beep, bleep": "[error notification]",
    "Ping": "[error notification]",
    "Ding": "[error notification]",
    "Siren": "[siren]",
    "Civil defense siren": "[siren]",
    "Police car (siren)": "[siren]",
    "Ambulance (siren)": "[siren]",
    "Fire engine, fire truck (siren)": "[siren]",
}


def _iter_segments(parsed: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(parsed, Mapping):
        yield parsed
        return
    if not isinstance(parsed, Iterable) or isinstance(parsed, (str, bytes)):
        raise ValueError("Malformed Whisper-AT audio tags: expected segments")
    for segment in parsed:
        if not isinstance(segment, Mapping):
            raise ValueError(
                "Malformed Whisper-AT audio tags: expected segment mappings"
            )
        yield segment


def extract_audio_tags(parsed: Any) -> tuple[list[str], list[str]]:
    """Return unique raw AudioSet tags and mapped DeafBench sound labels."""
    raw_tags: list[str] = []
    sounds: list[str] = []
    seen_raw: set[str] = set()
    seen_sounds: set[str] = set()

    for segment in _iter_segments(parsed):
        tags = segment.get("audio tags", [])
        if not isinstance(tags, Iterable) or isinstance(tags, (str, bytes)):
            raise ValueError(
                "Malformed Whisper-AT audio tags: expected a tag list"
            )
        for tag_entry in tags:
            if not isinstance(tag_entry, (list, tuple)) or not tag_entry:
                raise ValueError(
                    "Malformed Whisper-AT audio tags: expected tag entries"
                )
            label = tag_entry[0]
            if not isinstance(label, str) or not label:
                raise ValueError(
                    "Malformed Whisper-AT audio tags: expected tag labels"
                )
            if label not in seen_raw:
                raw_tags.append(label)
                seen_raw.add(label)
            mapped = AUDIOSET_TO_DEAFBENCH.get(label)
            if mapped is not None and mapped not in seen_sounds:
                sounds.append(mapped)
                seen_sounds.add(mapped)
    return raw_tags, sounds


def _validate_time_resolution(value: float) -> None:
    units = value / 0.4
    if (
        not math.isfinite(value)
        or value <= 0
        or not math.isclose(
            units,
            round(units),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("at_time_res must be a positive multiple of 0.4")


def _load_backend() -> Any:
    try:
        import whisper_at
    except ModuleNotFoundError as exc:
        if exc.name != "whisper_at":
            raise
        raise RuntimeError(
            "Whisper-AT is not installed. See the upstream "
            "Whisper-AT installation instructions."
        ) from exc
    return whisper_at


def run_whisper_at(
    audio_dir: Path,
    references: Path,
    output: Path,
    model_id: str = DEFAULT_MODEL,
    at_time_res: float = DEFAULT_AT_TIME_RES,
    top_k: int = DEFAULT_TOP_K,
    p_threshold: float = DEFAULT_P_THRESHOLD,
    backend: Any | None = None,
) -> ModelRunInfo:
    """Transcribe and tag one complete audio set into Model B JSONL."""
    _validate_time_resolution(at_time_res)
    wav_paths = _validated_wavs(audio_dir, references)
    runtime = _load_backend() if backend is None else backend
    model = runtime.load_model(model_id)
    include_class_list = list(range(AUDIOSET_CLASS_COUNT))
    records: list[Mapping[str, Any]] = []

    for wav_path in wav_paths:
        result = model.transcribe(
            str(wav_path),
            language="en",
            task="transcribe",
            verbose=False,
            at_time_res=at_time_res,
        )
        if not isinstance(result, Mapping):
            raise ValueError(
                f"Invalid Whisper-AT result for {wav_path.name}: "
                "expected a mapping"
            )
        parsed = runtime.parse_at_label(
            result,
            language="follow_asr",
            top_k=top_k,
            p_threshold=p_threshold,
            include_class_list=include_class_list,
        )
        raw_tags, sounds = extract_audio_tags(parsed)
        text = result.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"Invalid Whisper-AT transcript for {wav_path.name}: "
                "expected a string"
            )
        records.append(
            {
                "id": wav_path.stem,
                "text": text,
                "sounds": sounds,
                "audio_tags": raw_tags,
            }
        )

    atomic_write_jsonl(output, records)
    return ModelRunInfo("whisper-at", model_id)
