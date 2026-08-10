"""Validated paths and atomic artifacts for DeafBench benchmark runs."""

from __future__ import annotations

import json
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast


AudioSource = Literal["auto", "human", "synthetic"]
ResolvedAudioSource = Literal["human", "synthetic"]
STANDARD_SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class AudioSetStatus:
    """Completeness and validation result for one audio directory."""

    complete: bool
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    invalid: tuple[str, ...]


@dataclass(frozen=True)
class RunPaths:
    """All source-aware paths owned by one benchmark run."""

    dataset_dir: Path
    references: Path
    human_audio: Path
    synthetic_audio: Path
    run_dir: Path
    predictions: Path
    report: Path
    metadata: Path


def validate_dataset_name(dataset: str) -> str:
    """Return a safe dataset directory name or reject it."""
    if not dataset or dataset in {".", ".."} or any(
        separator in dataset for separator in ("/", "\\", ":")
    ):
        raise ValueError("Invalid dataset name")
    return dataset


def _is_safe_reference_id(sample_id: object) -> bool:
    if not isinstance(sample_id, str) or not sample_id:
        return False
    if sample_id != sample_id.strip() or sample_id in {".", ".."}:
        return False
    if "\x00" in sample_id or any(char in sample_id for char in ("/", "\\", ":")):
        return False
    candidate = Path(sample_id)
    return not candidate.is_absolute() and not candidate.drive


def _string_list(record: Mapping[str, Any], key: str) -> list[str]:
    value = record.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Invalid reference record: {key} must be a list of strings")
    return value


def load_reference_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load ordered, schema-validated benchmark reference records."""
    records: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Invalid reference record on line {line_number}: expected an object"
                )

            sample_id = value.get("id")
            if not _is_safe_reference_id(sample_id):
                raise ValueError(f"Invalid reference ID on line {line_number}")
            text = value.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"Invalid reference record on line {line_number}: text must be a string"
                )
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate reference ID: {sample_id}")

            normalized = dict(value)
            normalized["critical"] = _string_list(value, "critical")
            normalized["sounds"] = _string_list(value, "sounds")
            seen_ids.add(sample_id)
            records.append(normalized)

    if not records:
        raise ValueError("No reference records found")
    return tuple(records)


def load_reference_ids(path: Path) -> tuple[str, ...]:
    """Return validated IDs through the benchmark's single reference parser."""
    return tuple(cast(str, record["id"]) for record in load_reference_records(path))


def validate_wav_format(path: Path) -> None:
    """Require uncompressed 48 kHz, 16-bit, mono PCM WAV audio."""
    try:
        with wave.open(str(path), "rb") as handle:
            valid = (
                handle.getnchannels() == 1
                and handle.getsampwidth() == 2
                and handle.getframerate() == STANDARD_SAMPLE_RATE
                and handle.getcomptype() == "NONE"
            )
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"Invalid WAV format: {path}") from exc
    if not valid:
        raise ValueError(f"Invalid WAV format: {path}")


def inspect_audio_set(references: Path, audio_dir: Path) -> AudioSetStatus:
    """Compare the available WAV files with the validated reference IDs."""
    expected = set(load_reference_ids(references))
    audio_path = Path(audio_dir)
    wav_files = {path.stem: path for path in audio_path.glob("*.wav")}
    available = set(wav_files)
    invalid: list[str] = []

    for sample_id, wav_path in wav_files.items():
        try:
            validate_wav_format(wav_path)
        except ValueError:
            invalid.append(sample_id)

    missing = tuple(sorted(expected - available))
    extra = tuple(sorted(available - expected))
    invalid_tuple = tuple(sorted(invalid))
    return AudioSetStatus(
        complete=not missing and not extra and not invalid_tuple,
        missing=missing,
        extra=extra,
        invalid=invalid_tuple,
    )


def resolve_audio_source(
    requested: AudioSource,
    human_status: AudioSetStatus,
) -> ResolvedAudioSource:
    """Resolve automatic selection without accepting partial human audio."""
    if requested == "synthetic":
        return "synthetic"
    if requested == "human":
        if not human_status.complete:
            raise ValueError("Human audio set is incomplete")
        return "human"
    if requested != "auto":
        raise ValueError(f"Unsupported audio source: {requested}")
    return "human" if human_status.complete else "synthetic"


def resolve_run_paths(
    repo_root: Path,
    dataset: str,
    model: str,
    source: ResolvedAudioSource,
) -> RunPaths:
    """Return deterministic workspace paths for one model and audio source."""
    dataset_dir = Path(repo_root) / "benchmarks" / validate_dataset_name(dataset)
    run_dir = dataset_dir / "runs" / model / source
    return RunPaths(
        dataset_dir=dataset_dir,
        references=dataset_dir / "references.jsonl",
        human_audio=dataset_dir / "audio",
        synthetic_audio=dataset_dir / "audio-synthetic",
        run_dir=run_dir,
        predictions=run_dir / "predictions.jsonl",
        report=run_dir / "report.md",
        metadata=run_dir / "run.json",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(text)
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically promote UTF-8 text from a sibling temporary file."""
    _atomic_write_text(path, text)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write a deterministic JSON object."""
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically write ordered JSON objects as newline-delimited JSON."""
    text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    _atomic_write_text(path, text)


def atomic_write_wav(
    path: Path,
    pcm_frames: bytes,
    sample_rate: int = STANDARD_SAMPLE_RATE,
) -> None:
    """Atomically write exact uncompressed 16-bit mono PCM frames."""
    if sample_rate != STANDARD_SAMPLE_RATE:
        raise ValueError("sample_rate must be 48000")
    if len(pcm_frames) % 2:
        raise ValueError("pcm_frames must contain complete 16-bit samples")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        with wave.open(str(temp_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm_frames)
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
