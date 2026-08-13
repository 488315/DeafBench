import numpy as np
import pytest

from deafbench.benchmark.interstitial import (
    INTERSTITIAL_NOISE_PROFILES,
    InterstitialInterval,
    build_interstitial_scene,
    evaluate_interstitial_prediction,
    summarize_interstitial_robustness,
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


@pytest.mark.parametrize(
    ("speech_before", "speech_after"),
    [
        (np.zeros((4_800, 1)), np.full((4_800, 1), 0.1)),
        (np.full((4_800, 1), 0.1), np.zeros((4_800, 1))),
        (np.empty((0, 1)), np.full((4_800, 1), 0.1)),
        (np.full((4_800, 1), 0.1), np.empty((0, 1))),
    ],
)
def test_interstitial_scene_requires_two_non_silent_speech_anchors(
    speech_before: np.ndarray,
    speech_after: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="non-silent speech anchor"):
        build_interstitial_scene(
            speech_before,
            speech_after,
            profile="breathing",
            snr_db=10.0,
        )


@pytest.mark.parametrize(
    ("prediction", "ignored", "hallucinated", "word_count"),
    [
        ({"text": ""}, True, False, 0),
        ({"text": "   "}, True, False, 0),
        ({"text": "[keyboard clicks]"}, False, False, 0),
        ({"text": "", "sounds": ["[rustling]"]}, False, False, 0),
        ({"text": "keyboard clicks"}, False, True, 2),
        ({"text": "Thanks for watching."}, False, True, 3),
        ({"text": "[thanks for watching]"}, False, True, 3),
        ({"text": "[street noise] hello"}, False, True, 1),
    ],
)
def test_interstitial_prediction_distinguishes_annotations_from_speech(
    prediction: dict[str, object],
    ignored: bool,
    hallucinated: bool,
    word_count: int,
) -> None:
    result = evaluate_interstitial_prediction(prediction)

    assert result.ignored is ignored
    assert result.lexical_hallucination is hallucinated
    assert result.hallucinated_word_count == word_count


def test_interstitial_robustness_reports_hallucinations_by_snr() -> None:
    cases = [
        (
            InterstitialInterval(100, 200, "street-noise", 20.0),
            {"text": ""},
        ),
        (
            InterstitialInterval(100, 200, "office-chatter", 10.0),
            {"text": "[office chatter]"},
        ),
        (
            InterstitialInterval(100, 200, "breathing", 0.0),
            {"text": "Thank you."},
        ),
        (
            InterstitialInterval(100, 200, "rustling", -5.0),
            {"text": "I can hear you."},
        ),
    ]

    summary = summarize_interstitial_robustness(cases)

    assert summary == {
        "samples": 4,
        "ignored_intervals": 1,
        "lexical_hallucinations": 2,
        "ignore_rate_percent": 25.0,
        "lexical_hallucination_rate_percent": 50.0,
        "by_snr_db": [
            {
                "snr_db": 20.0,
                "samples": 1,
                "ignored_intervals": 1,
                "lexical_hallucinations": 0,
                "ignore_rate_percent": 100.0,
                "lexical_hallucination_rate_percent": 0.0,
            },
            {
                "snr_db": 10.0,
                "samples": 1,
                "ignored_intervals": 0,
                "lexical_hallucinations": 0,
                "ignore_rate_percent": 0.0,
                "lexical_hallucination_rate_percent": 0.0,
            },
            {
                "snr_db": 0.0,
                "samples": 1,
                "ignored_intervals": 0,
                "lexical_hallucinations": 1,
                "ignore_rate_percent": 0.0,
                "lexical_hallucination_rate_percent": 100.0,
            },
            {
                "snr_db": -5.0,
                "samples": 1,
                "ignored_intervals": 0,
                "lexical_hallucinations": 1,
                "ignore_rate_percent": 0.0,
                "lexical_hallucination_rate_percent": 100.0,
            },
        ],
    }


@pytest.mark.parametrize(
    "prediction",
    [
        {"text": None},
        {"text": "", "sounds": "[rustling]"},
        {"text": "", "sounds": [""]},
    ],
)
def test_interstitial_prediction_rejects_malformed_output(
    prediction: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Interstitial prediction"):
        evaluate_interstitial_prediction(prediction)
