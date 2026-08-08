"""Compatibility wrapper for the installed DeafBench recorder."""

from __future__ import annotations

from deafbench.recorder import app as _app


AudioRecorder = _app.AudioRecorder
RecorderApp = _app.RecorderApp
FORMAT_TEXT = _app.FORMAT_TEXT
resolve_dataset_paths = _app.resolve_dataset_paths
build_parser = _app.build_parser
_sounddevice = _app._sounddevice


def main(argv: list[str] | None = None) -> int:
    """Run the packaged recorder through the legacy tools module path."""
    _app._sounddevice = _sounddevice
    return _app.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
