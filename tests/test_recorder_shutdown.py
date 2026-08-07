import pytest

from tools.recorder.recorder import RecorderApp


pytestmark = pytest.mark.unit


class _FakeRecorder:
    """Recorder state fake for shutdown scheduling tests."""

    is_recording = False
    duration = 0.0
    peak_level = 0.0


class _FakeVariable:
    """Small Tk variable fake with get/set behavior."""

    def __init__(self, value):
        """Initialize the fake variable value."""
        self.value = value

    def get(self):
        """Return the current fake value."""
        return self.value

    def set(self, value):
        """Replace the current fake value."""
        self.value = value


class _FakeLevel(dict):
    """Dictionary-backed progress-bar value fake."""

    pass


class _FakeRoot:
    """Tk root fake that tracks scheduling and destruction."""

    def __init__(self):
        """Initialize scheduling and destruction counters."""
        self.after_calls = 0
        self.destroyed = False

    def after(self, delay, callback):
        """Record a requested scheduled callback."""
        del delay, callback
        self.after_calls += 1
        return "tick-1"

    def destroy(self):
        """Record that the fake Tk root was destroyed."""
        self.destroyed = True


class _FakeMessageBox:
    """Message-box fake that rejects unexpected warnings."""

    @staticmethod
    def showwarning(title, message):
        """Fail the test if shutdown unexpectedly shows a warning."""
        raise AssertionError(f"unexpected warning: {title}: {message}")


def test_tick_does_not_reschedule_after_app_closes():
    """Do not schedule another UI tick after the app closes."""
    app = RecorderApp.__new__(RecorderApp)
    app.root = _FakeRoot()
    app.recorder = _FakeRecorder()
    app.level = _FakeLevel()
    app.duration_var = _FakeVariable("0.0 s")
    app.messagebox = _FakeMessageBox()

    app._on_close()
    app._tick()

    assert app.root.destroyed is True
    assert app.root.after_calls == 0
