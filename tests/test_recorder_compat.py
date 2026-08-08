from pathlib import Path

from tools.recorder import recorder as recorder_module


def test_legacy_recorder_injects_repository_root_when_unspecified(monkeypatch):
    calls = []
    monkeypatch.setattr(recorder_module._app, "main", lambda args: calls.append(args) or 0)

    assert recorder_module.main(["--dataset", "non-speech-v1"]) == 0

    repo_root = Path(recorder_module.__file__).resolve().parents[2]
    assert calls == [[
        "--repo-root",
        str(repo_root),
        "--dataset",
        "non-speech-v1",
    ]]


def test_legacy_recorder_restores_packaged_sounddevice(monkeypatch):
    previous = object()
    replacement = object()
    observed = []
    monkeypatch.setattr(recorder_module._app, "_sounddevice", previous)
    monkeypatch.setattr(recorder_module, "_sounddevice", replacement)

    def fake_main(_args):
        observed.append(recorder_module._app._sounddevice)
        return 0

    monkeypatch.setattr(recorder_module._app, "main", fake_main)

    assert recorder_module.main([]) == 0
    assert observed == [replacement]
    assert recorder_module._app._sounddevice is previous
