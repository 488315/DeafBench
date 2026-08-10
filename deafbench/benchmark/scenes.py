"""Deterministic scene planning and mixing for synthetic benchmarks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from deafbench.recorder.core import synthesize_sound_event


DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_SCENE_PROFILE = "default-v1"
BACKGROUND_PROFILE = "office-v1"
BACKGROUND_SNR_DB = 15.0
SPEECH_LEAD_MS = 500
SCENE_TAIL_MS = 500


@dataclass(frozen=True)
class TimedEvent:
    """One environmental sound event positioned within a scene."""

    label: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class ScenePlan:
    """Reproducible timing and ambience metadata for one audio scene."""

    sample_id: str
    scene_profile: str
    seed: int
    sample_rate: int
    speech_start_ms: int
    speech_end_ms: int
    scene_end_ms: int
    background_profile: str
    background_start_ms: int
    background_end_ms: int
    background_snr_db: float
    events: tuple[TimedEvent, ...]


def _derive_seed(
    seed: int,
    namespace: str,
    scene_profile: str,
    sample_id: str,
) -> int:
    payload = f"{seed}:{namespace}:{scene_profile}:{sample_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _as_float_mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError(
            "Audio samples must have shape (frames,) or (frames, channels)"
        )
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    return data


def resample_mono(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Average channels and linearly resample to float64 mono audio."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")

    mono = _as_float_mono(samples)
    if len(mono) == 0:
        return np.empty((0, 1), dtype=np.float64)

    target_frames = max(1, round(len(mono) * target_rate / source_rate))
    source_positions = np.linspace(0.0, 1.0, len(mono))
    target_positions = np.linspace(0.0, 1.0, target_frames)
    resampled = np.interp(target_positions, source_positions, mono[:, 0])
    return resampled.reshape(-1, 1)


def _event_centers(count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.float64)
    if count == 1:
        return np.array([0.5], dtype=np.float64)
    return np.linspace(0.25, 0.75, count)


def _plan_events(
    sample_id: str,
    sound_labels: Sequence[str],
    seed: int,
    scene_profile: str,
    speech_start_ms: int,
    speech_end_ms: int,
) -> tuple[TimedEvent, ...]:
    centers = _event_centers(len(sound_labels))
    if len(centers) == 0:
        return ()

    duration_ms = speech_end_ms - speech_start_ms
    event_seed = _derive_seed(seed, "events", scene_profile, sample_id)
    jitter = np.random.default_rng(event_seed).uniform(
        -0.08,
        0.08,
        len(sound_labels),
    )
    events: list[TimedEvent] = []
    for label, center, offset in zip(sound_labels, centers, jitter):
        cue = synthesize_sound_event(label)
        cue_duration_ms = round(1_000 * len(cue) / DEFAULT_SAMPLE_RATE)
        target_center_ms = (
            speech_start_ms + (center + offset) * duration_ms
        )
        unclamped_start = round(target_center_ms - cue_duration_ms / 2)
        latest_start = max(
            speech_start_ms,
            speech_end_ms - cue_duration_ms,
        )
        start_ms = min(
            max(unclamped_start, speech_start_ms),
            latest_start,
        )
        events.append(
            TimedEvent(
                label=label,
                start_ms=start_ms,
                end_ms=start_ms + cue_duration_ms,
            )
        )
    return tuple(
        sorted(events, key=lambda event: (event.start_ms, event.label))
    )


def plan_scene(
    sample_id: str,
    speech_frames: int,
    sound_labels: Sequence[str],
    seed: int = 42,
    scene_profile: str = DEFAULT_SCENE_PROFILE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> ScenePlan:
    """Build the exact deterministic plan for a supported scene profile."""
    if scene_profile != DEFAULT_SCENE_PROFILE:
        raise ValueError(f"Unsupported scene profile: {scene_profile}")
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError("default-v1 requires a 48000 Hz sample rate")
    if speech_frames < 0:
        raise ValueError("speech_frames must not be negative")

    speech_start_ms = SPEECH_LEAD_MS
    speech_duration_ms = round(
        1_000 * speech_frames / DEFAULT_SAMPLE_RATE
    )
    speech_end_ms = speech_start_ms + speech_duration_ms
    events = _plan_events(
        sample_id,
        sound_labels,
        seed,
        scene_profile,
        speech_start_ms,
        speech_end_ms,
    )
    content_end_ms = max(
        speech_end_ms,
        max((event.end_ms for event in events), default=speech_end_ms),
    )
    scene_end_ms = content_end_ms + SCENE_TAIL_MS
    return ScenePlan(
        sample_id=sample_id,
        scene_profile=scene_profile,
        seed=seed,
        sample_rate=sample_rate,
        speech_start_ms=speech_start_ms,
        speech_end_ms=speech_end_ms,
        scene_end_ms=scene_end_ms,
        background_profile=BACKGROUND_PROFILE,
        background_start_ms=0,
        background_end_ms=scene_end_ms,
        background_snr_db=BACKGROUND_SNR_DB,
        events=events,
    )


def _office_background(
    speech: np.ndarray,
    plan: ScenePlan,
    scene_frames: int,
) -> np.ndarray:
    background_seed = _derive_seed(
        plan.seed,
        "background",
        plan.scene_profile,
        plan.sample_id,
    )
    raw_noise = np.random.default_rng(background_seed).normal(
        0.0,
        1.0,
        scene_frames,
    )
    kernel = np.ones(64, dtype=np.float64) / 64.0
    smoothed = np.convolve(raw_noise, kernel, mode="same")
    noise_rms = float(np.sqrt(np.mean(np.square(smoothed))))
    speech_rms = float(np.sqrt(np.mean(np.square(speech))))
    target_rms = (
        speech_rms / (10 ** (plan.background_snr_db / 20.0))
        if speech_rms > 0.0
        else 0.01
    )
    return smoothed * (target_rms / noise_rms)


def _frame_at(milliseconds: int, sample_rate: int) -> int:
    return round(milliseconds * sample_rate / 1_000)


def mix_scene(speech_pcm: np.ndarray, plan: ScenePlan) -> np.ndarray:
    """Mix speech, deterministic ambience, and planned event cues."""
    if plan.scene_profile != DEFAULT_SCENE_PROFILE:
        raise ValueError(f"Unsupported scene profile: {plan.scene_profile}")
    if plan.sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError("default-v1 requires a 48000 Hz sample rate")

    speech = _as_float_mono(speech_pcm)[:, 0]
    scene_frames = _frame_at(plan.scene_end_ms, plan.sample_rate)
    scene = _office_background(speech, plan, scene_frames)

    speech_start = _frame_at(plan.speech_start_ms, plan.sample_rate)
    speech_end = speech_start + len(speech)
    if speech_end > scene_frames:
        raise ValueError("Speech does not fit within the scene plan")
    scene[speech_start:speech_end] += speech

    for event in plan.events:
        cue = synthesize_sound_event(event.label)[:, 0].astype(np.float64)
        cue /= 32_768.0
        event_start = _frame_at(event.start_ms, plan.sample_rate)
        event_end = event_start + len(cue)
        if event_end > scene_frames:
            raise ValueError("Event does not fit within the scene plan")
        scene[event_start:event_end] += cue

    peak = float(np.max(np.abs(scene))) if len(scene) else 0.0
    if peak > 0.98:
        scene *= 0.98 / peak
    converted = np.clip(
        np.rint(scene * 32_767.0),
        -32_768,
        32_767,
    ).astype(np.int16)
    return converted.reshape(-1, 1)
