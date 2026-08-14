"""Installed Whisper-AT adapter for structured Model B predictions."""

from __future__ import annotations

import hashlib
import math
import os
import wave
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
from urllib.request import urlopen

from deafbench.benchmark.models import ModelRunInfo, _validated_wavs
from deafbench.benchmark.workspace import atomic_write_jsonl
from deafbench.model_registry import get_model_license


DEFAULT_MODEL = "medium.en"
UPSTREAM_MODEL_ID = "YuanGongND/whisper-at"
UPSTREAM_REVISION = "17d94d6acd53866390ce70f95afa13507dcb8aef"
DEFAULT_AT_TIME_RES = 10.0
DEFAULT_TOP_K = 5
DEFAULT_P_THRESHOLD = -1.0
AUDIOSET_CLASS_COUNT = 527
PINNED_CHECKPOINTS = {
    "medium.en.pt": (
        "https://openaipublic.azureedge.net/main/whisper/models/"
        "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/"
        "medium.en.pt",
        "d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f",
    ),
    "medium.en_ori.pth": (
        "https://www.dropbox.com/s/bbvylvmgns8ja4p/medium.en_ori.pth?dl=1",
        "2bb6ed52169cffd19623106dadb71918da27f8664ea1788c6379956b91ad2cea",
    ),
}

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


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_pinned_checkpoints(
    cache_dir: Path,
    *,
    opener: Callable[[str], Any] = urlopen,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_digest) in PINNED_CHECKPOINTS.items():
        destination = cache_dir / filename
        if destination.exists():
            if (
                not destination.is_file()
                or _checkpoint_digest(destination) != expected_digest
            ):
                raise RuntimeError(f"Whisper-AT checkpoint hash mismatch: {filename}")
            continue

        partial = cache_dir / f"{filename}.part"
        try:
            with opener(url) as source, partial.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
            if _checkpoint_digest(partial) != expected_digest:
                raise RuntimeError(
                    f"Whisper-AT checkpoint hash mismatch: {filename}"
                )
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise


def run_whisper_at(
    audio_dir: Path,
    references: Path,
    output: Path,
    model_id: str = DEFAULT_MODEL,
    at_time_res: float = DEFAULT_AT_TIME_RES,
    top_k: int = DEFAULT_TOP_K,
    p_threshold: float = DEFAULT_P_THRESHOLD,
    backend: Any | None = None,
    clock: Callable[[], float] = perf_counter,
) -> ModelRunInfo:
    """Transcribe and tag one complete audio set into Model B JSONL."""
    if model_id != DEFAULT_MODEL:
        raise ValueError(f"Unsupported unpinned Whisper-AT model: {model_id}")
    _validate_time_resolution(at_time_res)
    wav_paths = _validated_wavs(audio_dir, references)
    if backend is None:
        import torch

        get_model_license(UPSTREAM_MODEL_ID)
        cache_root = Path(
            os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        ) / "whisper"
        _prepare_pinned_checkpoints(cache_root)
        runtime = _load_backend()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = runtime.load_model(
            model_id,
            device=device,
            download_root=str(cache_root),
        )
    else:
        runtime = backend
        torch = None
        device = "injected"
        model = runtime.load_model(model_id)
    include_class_list = list(range(AUDIOSET_CLASS_COUNT))
    records: list[Mapping[str, Any]] = []
    latencies_ms: list[float] = []
    total_audio_seconds = 0.0

    for wav_path in wav_paths:
        if torch is not None and device == "cuda":
            torch.cuda.synchronize()
        started = clock()
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
        text = result.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"Invalid Whisper-AT transcript for {wav_path.name}: "
                "expected a string"
            )
        parsed = runtime.parse_at_label(
            result,
            language="follow_asr",
            top_k=top_k,
            p_threshold=p_threshold,
            include_class_list=include_class_list,
        )
        raw_tags, sounds = extract_audio_tags(parsed)
        if torch is not None and device == "cuda":
            torch.cuda.synchronize()
        latency_ms = round((clock() - started) * 1_000.0, 6)
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise ValueError(
                f"Invalid Whisper-AT latency for {wav_path.name}: "
                "expected a positive finite number"
            )
        with wave.open(str(wav_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"Invalid Whisper-AT duration for {wav_path.name}: "
                "expected a positive finite number"
            )
        latencies_ms.append(latency_ms)
        total_audio_seconds += duration
        records.append(
            {
                "id": wav_path.stem,
                "latency_ms": latency_ms,
                "text": text,
                "sounds": sounds,
                "audio_tags": raw_tags,
            }
        )

    atomic_write_jsonl(output, records)
    return ModelRunInfo(
        "whisper-at",
        UPSTREAM_MODEL_ID,
        revision=UPSTREAM_REVISION,
        decoding={
            "at_time_res": at_time_res,
            "backend_model": model_id,
            "device": device,
            "language": "en",
            "p_threshold": p_threshold,
            "task": "transcribe",
            "top_k": top_k,
        },
        performance={
            "local_rtfx": total_audio_seconds
            / (sum(latencies_ms) / 1_000.0),
            "median_latency_ms": median(latencies_ms),
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated()
                if torch is not None and device == "cuda"
                else 0
            ),
        },
    )
