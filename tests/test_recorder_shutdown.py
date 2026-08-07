import pytest

from tools.recorder.recorder import RecorderApp


pytestmark = pytest.mark.unit


class _FakeRecorder:
    is_recording = False
    duration = 0.0
    peak_level = 0.0


class _FakeVariable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _FakeLevel(dict):
    pass


class _FakeRoot:
    def __init__(self):
        self.after_calls = 0
        self.destroyed = False

    def after(self, delay, callback):
        del delay, callback
        self.after_calls += 1
        return "tick-1"

    def destroy(self):
        self.destroyed = True


class _FakeMessageBox:
    @staticmethod
    def showwarning(title, message):
        raise AssertionError(f"unexpected warning: {title}: {message}")


def test_tick_does_not_reschedule_after_app_closes():
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
