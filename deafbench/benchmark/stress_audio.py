"""Model-independent acoustic transforms for controlled stress comparisons."""

from __future__ import annotations

import math

import numpy as np

from deafbench.benchmark.noise import synthesize_noise


def _mono(samples: np.ndarray) -> np.ndarray:
    value = np.asarray(samples)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    if value.ndim != 2 or value.shape[1] < 1 or not len(value):
        raise ValueError("Audio must contain one or more frames")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError("Audio samples must be numeric")
    result = value.astype(np.float64)
    if np.issubdtype(value.dtype, np.integer):
        result /= max(abs(np.iinfo(value.dtype).min), np.iinfo(value.dtype).max)
    if result.shape[1] > 1:
        result = result.mean(axis=1, keepdims=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("Audio samples must be finite")
    return result


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    target_frames = max(1, round(len(samples) * target_rate / source_rate))
    source_positions = np.arange(len(samples), dtype=np.float64)
    target_positions = np.linspace(0.0, len(samples) - 1, target_frames)
    return np.interp(target_positions, source_positions, samples)


def add_noise_at_snr(
    samples: np.ndarray,
    profile: str,
    snr_db: float,
    sample_rate: int,
    seed: int,
) -> np.ndarray:
    """Mix a deterministic noise profile at the requested RMS SNR."""
    speech = _mono(samples)[:, 0]
    if sample_rate <= 0 or not math.isfinite(snr_db):
        raise ValueError("sample rate and SNR must be finite and positive")
    speech_rms = float(np.sqrt(np.mean(np.square(speech))))
    if speech_rms <= 0.0:
        raise ValueError("SNR calibration requires non-silent speech")
    noise = synthesize_noise(
        profile,
        frames=len(speech),
        sample_rate=sample_rate,
        seed=seed,
    )
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    target_rms = speech_rms / (10 ** (snr_db / 20.0))
    mixed = speech + noise * target_rms / noise_rms
    return mixed.reshape(-1, 1)


def _mulaw_roundtrip(samples: np.ndarray) -> np.ndarray:
    mu = 255.0
    clipped = np.clip(samples, -1.0, 1.0)
    encoded = np.sign(clipped) * np.log1p(mu * np.abs(clipped)) / np.log1p(mu)
    quantized = np.rint((encoded + 1.0) * 127.5).astype(np.uint8)
    restored = quantized.astype(np.float64) / 127.5 - 1.0
    return np.sign(restored) * np.expm1(np.abs(restored) * np.log1p(mu)) / mu


def simulate_telephony(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Round-trip audio through an 8 kHz mu-law narrowband simulation."""
    if sample_rate < 8_000:
        raise ValueError("Telephony input sample rate must be at least 8000 Hz")
    speech = _mono(samples)[:, 0]
    narrowband = _resample(speech, sample_rate, 8_000)
    decoded = _mulaw_roundtrip(narrowband)
    restored = _resample(decoded, 8_000, sample_rate)
    if len(restored) != len(speech):
        restored = np.interp(
            np.linspace(0.0, len(restored) - 1, len(speech)),
            np.arange(len(restored), dtype=np.float64),
            restored,
        )
    return restored.reshape(-1, 1)


def apply_reverberation(
    samples: np.ndarray,
    sample_rate: int,
    rt60_seconds: float,
) -> np.ndarray:
    """Apply a deterministic exponentially decaying room impulse response."""
    if sample_rate <= 0 or not math.isfinite(rt60_seconds) or rt60_seconds <= 0:
        raise ValueError("sample rate and RT60 must be finite and positive")
    speech = _mono(samples)[:, 0]
    tap_count = max(2, round(sample_rate * min(rt60_seconds, 2.0)))
    times = np.arange(tap_count, dtype=np.float64) / sample_rate
    impulse = np.exp(-times * math.log(1_000.0) / rt60_seconds)
    impulse[0] = 1.0
    impulse[1:] *= 0.08
    result = np.convolve(speech, impulse, mode="full")[: len(speech)]
    return result.reshape(-1, 1)


def insert_silence(
    samples: np.ndarray,
    sample_rate: int,
    duration_seconds: float,
    at_fraction: float = 0.5,
) -> np.ndarray:
    """Insert a known intra-phrase silence without closing the utterance."""
    if not 0.0 <= at_fraction <= 1.0:
        raise ValueError("Pause position fraction must be between zero and one")
    if sample_rate <= 0 or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("sample rate and pause duration must be finite and positive")
    speech = _mono(samples)
    split = round(len(speech) * at_fraction)
    silence = np.zeros((round(sample_rate * duration_seconds), 1))
    return np.concatenate((speech[:split], silence, speech[split:]))


def vary_rate(samples: np.ndarray, factor: float) -> np.ndarray:
    """Create a deterministic duration stress proxy at the requested rate."""
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("Rate factor must be finite and positive")
    speech = _mono(samples)[:, 0]
    frames = max(1, round(len(speech) / factor))
    result = np.interp(
        np.linspace(0.0, len(speech) - 1, frames),
        np.arange(len(speech), dtype=np.float64),
        speech,
    )
    return result.reshape(-1, 1)
