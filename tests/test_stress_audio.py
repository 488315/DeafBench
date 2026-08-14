import numpy as np
import pytest

from deafbench.benchmark.stress_audio import (
    add_noise_at_snr,
    apply_reverberation,
    insert_silence,
    simulate_telephony,
    vary_rate,
)


def _tone(frames: int = 48_000, sample_rate: int = 48_000) -> np.ndarray:
    time = np.arange(frames) / sample_rate
    return (0.15 * np.sin(2 * np.pi * 440 * time)).reshape(-1, 1)


@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 10.0, 20.0])
def test_additive_noise_reaches_declared_snr(snr_db: float) -> None:
    speech = _tone()
    stressed = add_noise_at_snr(
        speech,
        profile="street-noise",
        snr_db=snr_db,
        sample_rate=48_000,
        seed=42,
    )
    noise = stressed[:, 0] - speech[:, 0]
    measured = 20 * np.log10(
        np.sqrt(np.mean(np.square(speech[:, 0])))
        / np.sqrt(np.mean(np.square(noise)))
    )

    assert measured == pytest.approx(snr_db, abs=0.01)


def test_telephony_is_deterministic_and_preserves_duration() -> None:
    speech = _tone()

    first = simulate_telephony(speech, sample_rate=48_000)
    second = simulate_telephony(speech, sample_rate=48_000)

    assert first.shape == speech.shape
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, speech)


def test_reverberation_preserves_length_and_adds_a_tail() -> None:
    impulse = np.zeros((48_000, 1))
    impulse[1_000, 0] = 0.5

    result = apply_reverberation(impulse, sample_rate=48_000, rt60_seconds=0.8)

    assert result.shape == impulse.shape
    assert result[1_000, 0] == pytest.approx(0.5)
    assert np.any(result[1_001:, 0])


def test_long_pause_is_inserted_at_the_requested_frame() -> None:
    speech = _tone(frames=4_800)

    result = insert_silence(
        speech,
        sample_rate=48_000,
        duration_seconds=0.25,
        at_fraction=0.5,
    )

    assert result.shape == (16_800, 1)
    assert np.all(result[2_400:14_400, 0] == 0.0)
    np.testing.assert_array_equal(result[:2_400], speech[:2_400])


@pytest.mark.parametrize(("factor", "expected_frames"), [(0.5, 9_600), (2.0, 2_400)])
def test_rate_variation_changes_duration(factor: float, expected_frames: int) -> None:
    result = vary_rate(_tone(frames=4_800), factor=factor)

    assert result.shape == (expected_frames, 1)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: add_noise_at_snr(np.zeros((10, 1)), "wind", 0, 48_000, 1), "non-silent"),
        (lambda: simulate_telephony(_tone(10), sample_rate=7_999), "at least 8000"),
        (lambda: apply_reverberation(_tone(10), 48_000, 0.0), "positive"),
        (lambda: insert_silence(_tone(10), 48_000, 1.0, 1.1), "fraction"),
        (lambda: vary_rate(_tone(10), 0.0), "positive"),
    ],
)
def test_stress_audio_rejects_invalid_configuration(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_stress_audio_accepts_integer_stereo_and_one_dimensional_audio() -> None:
    stereo = np.column_stack(
        (np.full(48, 1_000, dtype=np.int16), np.full(48, -500, dtype=np.int16))
    )
    telephony = simulate_telephony(stereo, sample_rate=8_000)
    rate = vary_rate(np.full(48, 0.1), factor=1.0)

    assert telephony.shape == (48, 1)
    assert rate.shape == (48, 1)


@pytest.mark.parametrize(
    "audio",
    [
        np.empty((0, 1)),
        np.zeros((1, 1, 1)),
        np.array([["not audio"]]),
        np.array([[float("nan")]]),
    ],
)
def test_stress_audio_rejects_malformed_audio(audio: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Audio"):
        vary_rate(audio, factor=1.0)
