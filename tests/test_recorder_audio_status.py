import numpy as np
import pytest

from tools.recorder.recorder import AudioRecorder


pytestmark = pytest.mark.unit


class _FakeStream:
    """Minimal input-stream fake for recorder lifecycle tests."""

    def __init__(self, **kwargs):
        """Store stream construction arguments for assertions."""
        self.kwargs = kwargs

    def start(self):
        """Simulate starting the stream."""
        pass

    def stop(self):
        """Simulate stopping the stream."""
        pass

    def close(self):
        """Simulate closing the stream."""
        pass


class _FakeBackend:
    """Minimal sounddevice backend fake used by AudioRecorder tests."""

    def __init__(self):
        """Initialize without an active fake stream."""
        self.stream = None

    def check_input_settings(self, **kwargs):
        """Accept recorder input settings without touching hardware."""
        pass

    def InputStream(self, **kwargs):
        """Create and retain a fake input stream."""
        self.stream = _FakeStream(**kwargs)
        return self.stream


class _InputOverflowStatus:
    """Represent a truthy PortAudio input-overflow status."""

    input_overflow = True

    def __bool__(self):
        """Report that the callback status contains an error flag."""
        return True

    def __str__(self):
        """Return the user-facing status description."""
        return "input overflow"


def test_audio_recorder_rejects_capture_with_callback_status():
    """Reject a take when PortAudio reports callback status flags."""
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
