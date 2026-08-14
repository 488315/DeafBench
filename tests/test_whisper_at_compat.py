from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from deafbench.whisper_at_compat.source import prepare_source


UPSTREAM_REVISION = "17d94d6acd53866390ce70f95afa13507dcb8aef"
SETUP_SHA256 = "85625b02a8b04b156aa4602653f15bb9470e118fde012575b6eb27fd2f157841"
REQUIREMENTS_SHA256 = (
    "1ac4d63c8ed415cc378781d560bf2de7f6a5b2fbbfc8de6fc8afc33d38ae3d76"
)
requires_upstream = pytest.mark.skipif(
    os.environ.get("DEAFBENCH_RUN_WHISPER_AT_INTEGRATION") != "1",
    reason="requires the pinned public Whisper-AT source",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def pinned_source(tmp_path: Path) -> Path:
    source = tmp_path / "whisper-at"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "https://github.com/YuanGongND/whisper-at.git",
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "checkout", "--quiet", UPSTREAM_REVISION],
        check=True,
    )
    return source


@pytest.mark.integration
@requires_upstream
def test_prepare_source_preserves_runtime_requirements(pinned_source: Path) -> None:
    package = pinned_source / "package" / "whisper-at"
    requirements_before = package.joinpath("requirements.txt").read_bytes()

    prepared = prepare_source(pinned_source)

    assert prepared == package
    assert package.joinpath("requirements.txt").read_bytes() == requirements_before
    assert _sha256(package / "setup.py") != SETUP_SHA256
    assert "setuptools>=83.0.0" in package.joinpath("pyproject.toml").read_text()


@pytest.mark.integration
@requires_upstream
def test_prepare_source_rejects_modified_upstream(pinned_source: Path) -> None:
    setup = pinned_source / "package" / "whisper-at" / "setup.py"
    setup.write_text(setup.read_text(encoding="utf-8") + "\n# modified\n")

    with pytest.raises(ValueError, match="setup.py hash"):
        prepare_source(pinned_source)


def test_manifest_pins_existing_model_revision() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        root.joinpath("deafbench/whisper_at_compat/manifest.json").read_text()
    )
    registry = json.loads(root.joinpath("deafbench/model-registry.json").read_text())
    whisper_at = next(
        model for model in registry["models"] if model["model_id"] == manifest["model_id"]
    )

    assert manifest["revision"] == UPSTREAM_REVISION == whisper_at["revision"]
    assert manifest["source_files"]["package/whisper-at/setup.py"][
        "before_sha256"
    ] == SETUP_SHA256
    assert manifest["source_files"]["package/whisper-at/requirements.txt"][
        "sha256"
    ] == REQUIREMENTS_SHA256
