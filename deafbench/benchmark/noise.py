"""Deterministic synthetic noise profiles shared by stress scenes."""

from __future__ import annotations

import numpy as np


INTERSTITIAL_NOISE_PROFILES = (
    "street-noise",
    "office-chatter",
    "keyboard-clicks",
    "breathing",
    "rustling",
)
NOISE_PROFILES = (*INTERSTITIAL_NOISE_PROFILES, "wind")


def _smoothed_noise(
    rng: np.random.Generator,
    frames: int,
    width: int,
) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, frames + width - 1)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(raw, kernel, mode="valid")


def synthesize_noise(
    profile: str,
    *,
    frames: int,
    sample_rate: int,
    seed: int,
) -> np.ndarray:
    """Return one raw, uncalibrated synthetic noise signal."""
    if profile not in NOISE_PROFILES:
        raise ValueError(f"Unsupported noise profile: {profile}")
    if frames <= 0:
        raise ValueError("frames must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    rng = np.random.default_rng(seed)
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
        for start in rng.integers(
            0,
            max(1, frames - click_frames + 1),
            click_count,
        ):
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

    low_frequency = _smoothed_noise(rng, frames, max(8, sample_rate // 120))
    gust = 0.25 + 0.75 * np.square(np.sin(2.0 * np.pi * 0.7 * time_axis))
    return low_frequency * gust
