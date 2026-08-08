import pytest

from deafbench.recorder.core import DEFAULT_SAMPLE_RATE, _tone


pytestmark = pytest.mark.unit


def test_tone_handles_single_frame_duration():
    tone = _tone(440.0, 1.0 / DEFAULT_SAMPLE_RATE)

    assert tone.shape == (1, 1)
