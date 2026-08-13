"""Deterministic noise-only intervals for ASR hallucination evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


INTERSTITIAL_NOISE_PROFILES = (
    "street-noise",
    "office-chatter",
    "keyboard-clicks",
    "breathing",
    "rustling",
)


@dataclass(frozen=True)
class InterstitialInterval:
    """The exact noise-only frame range and its acoustic configuration."""

    start_frame: int
    end_frame: int
    profile: str
    snr_db: float


@dataclass(frozen=True)
class InterstitialScene:
    """Two speech runs separated by one known noise-only interval."""

    samples: np.ndarray
    interval: InterstitialInterval
    sample_rate: int


@dataclass(frozen=True)
class InterstitialResponse:
    """Whether a model emitted speech or annotations for a noise-only clip."""

    ignored: bool
    lexical_hallucination: bool
    hallucinated_word_count: int
    non_speech_labels: tuple[str, ...]


def _mono_float(samples: np.ndarray) -> np.ndarray:
    value = np.asarray(samples)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    if value.ndim != 2 or value.shape[1] < 1:
        raise ValueError("Audio must have shape (frames,) or (frames, channels)")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError("Audio samples must be numeric")
    converted = value.astype(np.float64)
    if np.issubdtype(value.dtype, np.integer):
        converted /= max(abs(np.iinfo(value.dtype).min), np.iinfo(value.dtype).max)
    if converted.shape[1] > 1:
        converted = converted.mean(axis=1, keepdims=True)
    if not np.all(np.isfinite(converted)):
        raise ValueError("Audio samples must be finite")
    return converted


def _smoothed_noise(rng: np.random.Generator, frames: int, width: int) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, frames + width - 1)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(raw, kernel, mode="valid")


def _profile_noise(
    profile: str,
    frames: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    time_axis = np.arange(frames, dtype=np.float64) / sample_rate
    if profile == "street-noise":
        return _smoothed_noise(rng, frames, 96) + 0.2 * rng.normal(size=frames)
    if profile == "office-chatter":
        carrier = _smoothed_noise(rng, frames, 24)
        envelope = 0.55 + 0.45 * np.square(np.sin(2.0 * np.pi * 3.1 * time_axis))
        return carrier * envelope
    if profile == "keyboard-clicks":
        signal = np.zeros(frames, dtype=np.float64)
        click_count = max(1, round(frames / sample_rate * 8))
        click_frames = max(1, round(sample_rate * 0.012))
        for start in rng.integers(0, max(1, frames - click_frames + 1), click_count):
            stop = min(frames, start + click_frames)
            envelope = np.exp(-np.linspace(0.0, 7.0, stop - start))
            signal[start:stop] += rng.choice((-1.0, 1.0)) * envelope
        return signal
    if profile == "breathing":
        carrier = _smoothed_noise(rng, frames, 48)
        envelope = 0.1 + 0.9 * np.square(np.sin(2.0 * np.pi * 0.32 * time_axis))
        return carrier * envelope
    if profile == "rustling":
        raw = rng.normal(size=frames)
        high_pass = np.diff(raw, prepend=raw[0])
        envelope = 0.35 + 0.65 * np.square(np.sin(2.0 * np.pi * 4.7 * time_axis))
        return high_pass * envelope
    raise ValueError(f"Unsupported interstitial noise profile: {profile}")


def build_interstitial_scene(
    speech_before: np.ndarray,
    speech_after: np.ndarray,
    *,
    profile: str,
    snr_db: float,
    duration_seconds: float = 0.5,
    seed: int = 42,
    sample_rate: int = 48_000,
) -> InterstitialScene:
    """Place calibrated noise between speech runs and retain its exact interval."""
    if profile not in INTERSTITIAL_NOISE_PROFILES:
        raise ValueError(f"Unsupported interstitial noise profile: {profile}")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be finite and positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    before = _mono_float(speech_before)
    after = _mono_float(speech_after)
    anchor_rms = [
        float(np.sqrt(np.mean(np.square(anchor)))) if len(anchor) else 0.0
        for anchor in (before[:, 0], after[:, 0])
    ]
    if any(rms <= 0.0 for rms in anchor_rms):
        raise ValueError("SNR calibration requires a non-silent speech anchor")
    speech_rms = float(np.sqrt(np.mean(np.square(np.concatenate((before[:, 0], after[:, 0]))))))

    noise_frames = round(sample_rate * duration_seconds)
    if noise_frames < 1:
        raise ValueError("duration_seconds must produce at least one frame")
    rng = np.random.default_rng(seed)
    noise = _profile_noise(profile, noise_frames, sample_rate, rng)
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    if noise_rms <= 0.0:
        raise ValueError("Noise profile produced an empty acoustic signal")
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    noise *= target_noise_rms / noise_rms

    start_frame = len(before)
    end_frame = start_frame + noise_frames
    samples = np.concatenate((before[:, 0], noise, after[:, 0])).reshape(-1, 1)
    return InterstitialScene(
        samples=samples,
        interval=InterstitialInterval(
            start_frame=start_frame,
            end_frame=end_frame,
            profile=profile,
            snr_db=snr_db,
        ),
        sample_rate=sample_rate,
    )


_BRACKETED_EVENT = re.compile(r"\[([^\]\r\n]+)\]")
_LEXICAL_TOKEN = re.compile(r"[^\W_]+(?:['_-][^\W_]+)*", re.UNICODE)
_KNOWN_EVENT_LABELS = frozenset(
    profile.replace("-", " ") for profile in INTERSTITIAL_NOISE_PROFILES
)


def _remove_known_event_annotation(match: re.Match[str]) -> str:
    label = " ".join(match.group(1).lower().replace("-", " ").split())
    return " " if label in _KNOWN_EVENT_LABELS else match.group(0)


def _normalize_sound_label(value: str) -> str:
    label = value.strip()
    if label.startswith("[") and label.endswith("]"):
        label = label[1:-1]
    return " ".join(label.lower().replace("-", " ").split())


def evaluate_interstitial_prediction(
    prediction: Mapping[str, object],
) -> InterstitialResponse:
    """Separate hallucinated speech from explicit non-speech annotations."""
    text = prediction.get("text", "")
    if not isinstance(text, str):
        raise ValueError("Interstitial prediction text must be a string")
    raw_sounds = prediction.get("sounds", ())
    if isinstance(raw_sounds, (str, bytes)) or not isinstance(raw_sounds, Sequence):
        raise ValueError("Interstitial prediction sounds must be a sequence of strings")
    sounds = tuple(raw_sounds)
    if not all(isinstance(sound, str) and sound.strip() for sound in sounds):
        raise ValueError("Interstitial prediction sounds must contain non-empty strings")

    unannotated = _BRACKETED_EVENT.sub(_remove_known_event_annotation, text)
    hallucinated_words = _LEXICAL_TOKEN.findall(unannotated)
    unsupported_label = any(
        _normalize_sound_label(match.group(1)) not in _KNOWN_EVENT_LABELS
        for match in _BRACKETED_EVENT.finditer(text)
    )
    for sound in sounds:
        if _normalize_sound_label(sound) not in _KNOWN_EVENT_LABELS:
            unsupported_label = True
            hallucinated_words.extend(_LEXICAL_TOKEN.findall(sound))
    emitted_output = bool(text.strip() or sounds)
    return InterstitialResponse(
        ignored=not emitted_output,
        lexical_hallucination=unsupported_label or bool(hallucinated_words),
        hallucinated_word_count=len(hallucinated_words),
        non_speech_labels=tuple(str(sound) for sound in sounds),
    )


def summarize_interstitial_robustness(
    cases: Sequence[tuple[InterstitialInterval, Mapping[str, object]]],
) -> dict[str, Any]:
    """Aggregate ignore and hallucination rates overall and at each SNR."""
    if not cases:
        raise ValueError("Interstitial robustness requires at least one case")
    for interval, _prediction in cases:
        if interval.profile not in INTERSTITIAL_NOISE_PROFILES:
            raise ValueError("Interstitial case has an unsupported noise profile")
        if not math.isfinite(interval.snr_db):
            raise ValueError("Interstitial case SNR must be finite")

    grouped: dict[float, list[InterstitialResponse]] = {}
    responses: list[InterstitialResponse] = []
    for interval, prediction in cases:
        response = evaluate_interstitial_prediction(prediction)
        responses.append(response)
        grouped.setdefault(interval.snr_db, []).append(response)

    def metrics(values: Sequence[InterstitialResponse]) -> dict[str, int | float]:
        total = len(values)
        ignored = sum(response.ignored for response in values)
        hallucinations = sum(response.lexical_hallucination for response in values)
        return {
            "samples": total,
            "ignored_intervals": ignored,
            "lexical_hallucinations": hallucinations,
            "ignore_rate_percent": 100.0 * ignored / total if total else 100.0,
            "lexical_hallucination_rate_percent": (
                100.0 * hallucinations / total if total else 0.0
            ),
        }

    summary: dict[str, Any] = metrics(responses)
    summary["by_snr_db"] = [
        {"snr_db": snr_db, **metrics(grouped[snr_db])}
        for snr_db in sorted(grouped, reverse=True)
    ]
    return summary
