from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


_RESOURCE_PACKAGE = "deafbench.whisper_at_compat"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, Any]:
    resource = files(_RESOURCE_PACKAGE).joinpath("manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _verify_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")


def prepare_source(source_root: Path) -> Path:
    """Verify and patch an exact checkout of the pinned Whisper-AT source."""
    manifest = _manifest()
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != manifest["revision"]:
        raise ValueError(
            f"Whisper-AT revision mismatch: expected {manifest['revision']}, "
            f"got {revision}"
        )

    source_files = manifest["source_files"]
    for relative_path, hashes in source_files.items():
        expected = hashes.get("before_sha256", hashes.get("sha256"))
        _verify_hash(source_root / relative_path, expected, Path(relative_path).name)

    patch = files(_RESOURCE_PACKAGE).joinpath(manifest["patch_file"])
    with patch.open("rb") as patch_stream:
        subprocess.run(
            ["git", "-C", str(source_root), "apply", "--check", "-"],
            check=True,
            stdin=patch_stream,
        )
    with patch.open("rb") as patch_stream:
        subprocess.run(
            ["git", "-C", str(source_root), "apply", "-"],
            check=True,
            stdin=patch_stream,
        )

    for relative_path, hashes in source_files.items():
        expected = hashes.get("after_sha256", hashes.get("sha256"))
        _verify_hash(source_root / relative_path, expected, Path(relative_path).name)
    return source_root / manifest["source_subdirectory"]


def install() -> None:
    """Clone, verify, patch, and install the pinned Whisper-AT package."""
    manifest = _manifest()
    with tempfile.TemporaryDirectory(prefix="deafbench-whisper-at-") as temporary:
        source_root = Path(temporary) / "source"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                manifest["upstream_url"],
                str(source_root),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "checkout",
                "--quiet",
                manifest["revision"],
            ],
            check=True,
        )
        package = prepare_source(source_root)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", str(package)],
            check=True,
        )
