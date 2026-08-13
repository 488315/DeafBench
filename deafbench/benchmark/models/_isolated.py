"""Restricted subprocess protocol for audited third-party model code."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


RESULT_MARKER = "DEAFBENCH_MODEL_RESULT="
_WORKER_PREFIX = "deafbench.benchmark.models._"
_PASSTHROUGH_ENVIRONMENT = (
    "APPDATA",
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HOME",
    "LD_LIBRARY_PATH",
    "LOCALAPPDATA",
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "WINDIR",
    "XDG_CACHE_HOME",
)


class IsolatedModelError(RuntimeError):
    """Raised when an audited model worker cannot be trusted or executed."""


def _worker_environment() -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in _PASSTHROUGH_ENVIRONMENT
        if name in os.environ
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def invoke_isolated_worker(
    module: str,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: int = 7_200,
) -> dict[str, Any]:
    """Invoke a packaged model worker with offline flags and a scrubbed environment."""
    if not module.startswith(_WORKER_PREFIX) or not all(
        component.isidentifier() for component in module.split(".")
    ):
        raise IsolatedModelError(f"unsafe isolated worker module: {module}")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise IsolatedModelError("isolated worker timeout must be positive")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", module],
            input=json.dumps(payload, sort_keys=True),
            cwd=Path(tempfile.gettempdir()).resolve(),
            env=_worker_environment(),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except OSError as exc:
        raise IsolatedModelError("could not launch isolated model worker") from exc
    except subprocess.TimeoutExpired as exc:
        raise IsolatedModelError("isolated model worker timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "unknown error"
        raise IsolatedModelError(f"isolated model worker failed: {detail}") from exc

    marker_lines = [
        line.removeprefix(RESULT_MARKER)
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if len(marker_lines) != 1:
        raise IsolatedModelError(
            "isolated model worker did not return exactly one result marker"
        )
    try:
        result = json.loads(marker_lines[0])
    except json.JSONDecodeError as exc:
        raise IsolatedModelError(
            "isolated model worker returned malformed JSON"
        ) from exc
    if not isinstance(result, dict):
        raise IsolatedModelError("isolated model worker returned invalid data")
    return result
