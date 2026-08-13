"""Deterministic container and waveform diagnostics for corpus admission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioDiagnostics:
    """Container facts and waveform measurements without policy decisions."""

    readable: bool
    error: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    container: str | None = None
    subtype: str | None = None
    frame_count: int = 0
    duration_seconds: float = 0.0
    peak_amplitude: float = 0.0
    clipping_fraction: float = 0.0
    leading_silence_ms: float | None = None
    trailing_silence_ms: float | None = None
    has_signal: bool = False


def _edge_silence(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: int,
) -> tuple[float | None, float | None]:
    frame_size = max(1, round(sample_rate * frame_ms / 1000))
    usable = len(samples) - (len(samples) % frame_size)
    if usable == 0:
        return None, None
    frames = samples[:usable].reshape(-1, frame_size)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak_rms = float(np.max(rms, initial=0.0))
    if peak_rms <= 1e-4:
        return None, None
    activity_threshold = max(1e-4, peak_rms * 0.10)
    active = np.flatnonzero(rms >= activity_threshold)
    if active.size == 0:
        return None, None
    leading = float(active[0] * frame_ms)
    trailing = float((len(frames) - active[-1] - 1) * frame_ms)
    return leading, trailing


def inspect_audio(path: Path, *, frame_ms: int = 10) -> AudioDiagnostics:
    """Decode an audio container and return facts needed by admission policy."""
    try:
        info = sf.info(path)
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        return AudioDiagnostics(readable=False, error=str(error))

    channels = info.channels
    frame_count = info.frames
    samples = samples.mean(axis=1) if samples.size else np.empty(0, dtype=np.float32)
    peak = float(np.max(np.abs(samples), initial=0.0))
    clipping = float(np.mean(np.abs(samples) >= (32767 / 32768))) if samples.size else 0.0
    leading, trailing = _edge_silence(samples, sample_rate, frame_ms=frame_ms)
    return AudioDiagnostics(
        readable=True,
        sample_rate=sample_rate,
        channels=channels,
        container=info.format,
        subtype=info.subtype,
        frame_count=frame_count,
        duration_seconds=(frame_count / sample_rate if sample_rate else 0.0),
        peak_amplitude=peak,
        clipping_fraction=clipping,
        leading_silence_ms=leading,
        trailing_silence_ms=trailing,
        has_signal=peak > 1e-4,
    )
