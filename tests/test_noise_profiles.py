import numpy as np
import pytest

from deafbench.benchmark.noise import NOISE_PROFILES, synthesize_noise


@pytest.mark.parametrize("profile", NOISE_PROFILES)
def test_noise_profiles_are_deterministic_non_silent_signals(profile: str) -> None:
    first = synthesize_noise(profile, frames=4_800, sample_rate=48_000, seed=17)
    same = synthesize_noise(profile, frames=4_800, sample_rate=48_000, seed=17)
    different = synthesize_noise(profile, frames=4_800, sample_rate=48_000, seed=18)

    assert first.shape == (4_800,)
    assert np.all(np.isfinite(first))
    assert np.sqrt(np.mean(np.square(first))) > 0.0
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize(
    ("profile", "frames", "sample_rate", "message"),
    [
        ("unknown", 4_800, 48_000, "Unsupported noise profile"),
        ("wind", 0, 48_000, "frames must be positive"),
        ("wind", 4_800, 0, "sample_rate must be positive"),
    ],
)
def test_noise_profiles_reject_invalid_configuration(
    profile: str,
    frames: int,
    sample_rate: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        synthesize_noise(
            profile,
            frames=frames,
            sample_rate=sample_rate,
            seed=17,
        )
