"""Shared model-independent WAV chunking for bounded ASR runtimes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


@contextmanager
def contiguous_audio_chunks(
    wav_path: Path,
    soundfile: Any,
    *,
    max_audio_seconds: float,
    runtime_name: str,
) -> Iterator[tuple[float, tuple[Path, ...]]]:
    """Yield a WAV or contiguous temporary chunks within a runtime limit."""
    info = soundfile.info(str(wav_path))
    if info.channels != 1:
        raise ValueError(f"{runtime_name} requires mono audio: {wav_path}")
    if info.duration <= 0:
        raise ValueError(f"{runtime_name} requires nonempty audio: {wav_path}")
    duration = float(info.duration)
    if duration <= max_audio_seconds:
        yield duration, (wav_path,)
        return

    frames_per_chunk = int(max_audio_seconds * info.samplerate)
    with TemporaryDirectory(prefix="deafbench-audio-") as directory:
        chunk_paths: list[Path] = []
        for index, start in enumerate(range(0, info.frames, frames_per_chunk)):
            frame_count = min(frames_per_chunk, info.frames - start)
            audio, sample_rate = soundfile.read(
                str(wav_path),
                start=start,
                frames=frame_count,
                always_2d=False,
            )
            if sample_rate != info.samplerate:
                raise ValueError(f"{runtime_name} sample rate changed: {wav_path}")
            chunk_path = Path(directory) / f"chunk-{index:04d}.wav"
            soundfile.write(
                str(chunk_path),
                audio,
                sample_rate,
                subtype=info.subtype,
            )
            chunk_paths.append(chunk_path)
        yield duration, tuple(chunk_paths)
