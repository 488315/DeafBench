import numpy as np
import pytest

from tools.recorder.recorder import AudioRecorder


pytestmark = pytest.mark.unit


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class _FakeBackend:
    def __init__(self):
        self.stream = None

    def check_input_settings(self, **kwargs):
        pass

    def InputStream(self, **kwargs):
        self.stream = _FakeStream(**kwargs)
        return self.stream


class _InputOverflowStatus:
    input_overflow = True

    def __bool__(self):
        return True

    def __str__(self):
        return "input overflow"


def test_audio_recorder_rejects_capture_with_callback_status():
    backend = _FakeBackend()
    recorder = AudioRecorder(backend=backend)
    recorder.start(device_index=1, channels=2)

    backend.stream.kwargs["callback"](
        np.array([[100, 200]], dtype=np.int16),
        1,
        None,
        _InputOverflowStatus(),
    )

    with pytest.raises(RuntimeError, match="input overflow"):
        recorder.stop()
