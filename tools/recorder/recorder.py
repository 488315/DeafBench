"""Compatibility wrapper for the installed DeafBench recorder."""

from __future__ import annotations

import sys
from pathlib import Path

from deafbench.recorder import app as _app


AudioRecorder = _app.AudioRecorder
RecorderApp = _app.RecorderApp
FORMAT_TEXT = _app.FORMAT_TEXT
resolve_dataset_paths = _app.resolve_dataset_paths
build_parser = _app.build_parser
_sounddevice = _app._sounddevice


def _legacy_args(argv: list[str] | None) -> list[str]:
    """Inject the source-tree repo root when the legacy caller omits it."""
    args = list(sys.argv[1:] if argv is None else argv)
    has_repo_root = any(
        arg == "--repo-root" or arg.startswith("--repo-root=")
        for arg in args
    )
    if not has_repo_root:
        repo_root = Path(__file__).resolve().parents[2]
        args = ["--repo-root", str(repo_root), *args]
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the packaged recorder through the legacy tools module path."""
    previous = _app._sounddevice
    _app._sounddevice = _sounddevice
    try:
        return _app.main(_legacy_args(argv))
    finally:
        _app._sounddevice = previous


if __name__ == "__main__":
    raise SystemExit(main())
