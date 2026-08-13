import numpy as np
import pytest

from deafbench.benchmark.interstitial import (
    INTERSTITIAL_NOISE_PROFILES,
    build_interstitial_scene,
)


@pytest.mark.parametrize("profile", INTERSTITIAL_NOISE_PROFILES)
@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 10.0, 20.0])
def test_interstitial_scene_places_snr_calibrated_noise_between_speech(
    profile: str,
    snr_db: float,
) -> None:
    sample_rate = 48_000
    speech_before = np.full((sample_rate // 2, 1), 0.1)
    speech_after = np.full((sample_rate // 2, 1), -0.1)

    scene = build_interstitial_scene(
        speech_before,
        speech_after,
        profile=profile,
        snr_db=snr_db,
        duration_seconds=0.25,
        seed=42,
        sample_rate=sample_rate,
    )

    interval = scene.interval
    noise = scene.samples[interval.start_frame : interval.end_frame, 0]
    speech_rms = 0.1
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))

    assert scene.samples.shape == (sample_rate + sample_rate // 4, 1)
    assert interval.start_frame == len(speech_before)
    assert interval.end_frame == len(speech_before) + sample_rate // 4
    assert interval.profile == profile
    assert interval.snr_db == snr_db
    assert noise_rms == pytest.approx(
        speech_rms / (10 ** (snr_db / 20.0)),
        rel=0.03,
    )


def test_interstitial_scene_is_reproducible_and_seed_sensitive() -> None:
    speech = np.full((4_800, 1), 0.1)

    first = build_interstitial_scene(
        speech,
        speech,
        profile="keyboard-clicks",
        snr_db=10.0,
        seed=7,
    )
    same = build_interstitial_scene(
        speech,
        speech,
        profile="keyboard-clicks",
        snr_db=10.0,
        seed=7,
    )
    different = build_interstitial_scene(
        speech,
        speech,
        profile="keyboard-clicks",
        snr_db=10.0,
        seed=8,
    )

    np.testing.assert_array_equal(first.samples, same.samples)
    assert not np.array_equal(first.samples, different.samples)


@pytest.mark.parametrize(
    ("profile", "snr_db", "duration_seconds", "message"),
    [
        ("unknown", 10.0, 0.5, "Unsupported interstitial noise profile"),
        ("street-noise", float("nan"), 0.5, "finite"),
        ("street-noise", 10.0, 0.0, "positive"),
    ],
)
def test_interstitial_scene_rejects_invalid_configuration(
    profile: str,
    snr_db: float,
    duration_seconds: float,
    message: str,
) -> None:
    speech = np.full((4_800, 1), 0.1)

    with pytest.raises(ValueError, match=message):
        build_interstitial_scene(
            speech,
            speech,
            profile=profile,
            snr_db=snr_db,
            duration_seconds=duration_seconds,
        )


def test_interstitial_scene_rejects_silent_speech_anchor() -> None:
    silence = np.zeros((4_800, 1))

    with pytest.raises(ValueError, match="non-silent speech anchor"):
        build_interstitial_scene(
            silence,
            silence,
            profile="breathing",
            snr_db=10.0,
        )
