import numpy as np
import pytest

from deafbench.benchmark.scenes import (
    mix_scene,
    plan_scene,
    resample_mono,
)


def test_scene_plan_is_reproducible_and_seed_sensitive() -> None:
    labels = ["[phone rings]", "[knock]"]

    first = plan_scene("ns-008", 48_000 * 4, labels, seed=42)
    same = plan_scene("ns-008", 48_000 * 4, labels, seed=42)
    different = plan_scene("ns-008", 48_000 * 4, labels, seed=43)

    assert first == same
    assert first.events != different.events
    assert first.speech_start_ms == 500
    assert first.speech_end_ms == 4_500
    assert first.background_profile == "office-v1"
    assert first.background_start_ms == 0
    assert first.background_end_ms == first.scene_end_ms
    assert first.background_snr_db == 15.0
    assert {event.label for event in first.events} == set(labels)
    assert first.events == tuple(
        sorted(first.events, key=lambda event: (event.start_ms, event.label))
    )


def test_scene_without_events_ends_after_speech_tail() -> None:
    plan = plan_scene("core-001", 48_000, [], seed=42)

    assert plan.events == ()
    assert plan.speech_start_ms == 500
    assert plan.speech_end_ms == 1_500
    assert plan.scene_end_ms == 2_000


def test_resample_mono_doubles_24khz_frame_count() -> None:
    source = np.linspace(
        -0.25,
        0.25,
        24_000,
        dtype=np.float64,
    ).reshape(-1, 1)

    result = resample_mono(source, 24_000)

    assert result.dtype == np.float64
    assert result.shape == (48_000, 1)
    assert result[0, 0] == pytest.approx(-0.25)
    assert result[-1, 0] == pytest.approx(0.25)


def test_resample_mono_averages_channels_before_interpolation() -> None:
    stereo = np.array(
        [[-1.0, 1.0], [0.0, 2.0]],
        dtype=np.float64,
    )

    result = resample_mono(stereo, 24_000, target_rate=48_000)

    np.testing.assert_allclose(
        result[:, 0],
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    )


def test_mix_scene_returns_int16_mono_with_planned_length() -> None:
    speech = np.full((48_000, 1), 0.1, dtype=np.float64)
    plan = plan_scene("ns-001", len(speech), ["[alarm]"], seed=42)

    mixed = mix_scene(speech, plan)

    assert mixed.dtype == np.int16
    assert mixed.shape == (plan.scene_end_ms * 48, 1)
    assert np.max(np.abs(mixed.astype(np.int32))) <= 32_112


def test_mix_scene_is_reproducible_for_silent_speech() -> None:
    speech = np.zeros((4_800, 1), dtype=np.float64)
    plan = plan_scene("silent", len(speech), [], seed=42)

    first = mix_scene(speech, plan)
    second = mix_scene(speech, plan)

    np.testing.assert_array_equal(first, second)
    assert np.any(first)


def test_unsupported_scene_profile_is_rejected_before_mixing() -> None:
    with pytest.raises(ValueError, match="Unsupported scene profile"):
        plan_scene(
            "ns-001",
            48_000,
            ["[alarm]"],
            scene_profile="unknown-v1",
        )
