"""Local ASR load-trial metrics without a hosted-service dependency."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any


def _number(
    observation: Mapping[str, object],
    field: str,
    *,
    positive: bool = False,
) -> float:
    value = observation.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
        or (not positive and float(value) < 0.0)
    ):
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"Load observation {field} must be a finite {qualifier}number")
    return float(value)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_load_trial(
    observations: Sequence[Mapping[str, object]],
    *,
    concurrency: int,
    wall_seconds: float,
) -> dict[str, Any]:
    """Summarize one declared local trial at a fixed concurrency level."""
    if not observations:
        raise ValueError("Load trial requires at least one observation")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("Load trial concurrency must be a positive integer")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or wall_seconds <= 0.0
    ):
        raise ValueError("Load trial wall time must be finite and positive")

    audio = [_number(item, "audio_seconds", positive=True) for item in observations]
    latency = [_number(item, "latency_ms") for item in observations]
    ttfb = [_number(item, "ttfb_ms") for item in observations]
    vram = [
        _number(item, "peak_vram_bytes")
        for item in observations
        if item.get("peak_vram_bytes") is not None
    ]
    cpu = [
        _number(item, "peak_cpu_percent")
        for item in observations
        if item.get("peak_cpu_percent") is not None
    ]
    if any(value > 100.0 for value in cpu):
        raise ValueError("Load observation peak_cpu_percent must not exceed 100")

    total_audio = sum(audio)
    wall = float(wall_seconds)
    return {
        "requests": len(observations),
        "concurrency": concurrency,
        "audio_seconds": total_audio,
        "wall_seconds": wall,
        "throughput_requests_per_second": len(observations) / wall,
        "aggregate_rtf": wall / total_audio,
        "aggregate_rtfx": total_audio / wall,
        "median_latency_ms": median(latency),
        "p95_latency_ms": _percentile(latency, 0.95),
        "median_ttfb_ms": median(ttfb),
        "p95_ttfb_ms": _percentile(ttfb, 0.95),
        "ttfb_over_500ms": sum(value > 500.0 for value in ttfb),
        "peak_vram_bytes": int(max(vram)) if vram else None,
        "peak_cpu_percent": max(cpu) if cpu else None,
    }
