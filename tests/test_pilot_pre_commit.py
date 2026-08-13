import os
import subprocess
import sys
from pathlib import Path

import pytest

from deafbench.pilot import hook
from deafbench.pilot.source_control import StagedFinding


def _without_coverage_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COV_CORE") and key != "COVERAGE_PROCESS_START"
    }


def test_pre_commit_hook_blocks_staged_audio(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    source = Path(__file__).parents[1]
    (repo / "sample.wav").write_bytes(b"synthetic")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)

    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test Operator"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.hooksPath", str(source / ".githooks")],
        check=True,
    )
    completed = subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "must fail"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **_without_coverage_environment(),
            "PYTHONPATH": str(source),
        },
    )

    assert completed.returncode == 1
    assert "BLOCKED sample.wav" in completed.stderr


def test_installed_hook_reports_findings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        hook,
        "scan_staged",
        lambda _: (StagedFinding("sample.wav", "customer artifact path"),),
    )
    monkeypatch.setattr(sys, "argv", ["hook", "--repo-root", "."])

    assert hook.main() == 1
    assert "BLOCKED sample.wav" in capsys.readouterr().out


def test_ci_runs_tracked_customer_artifact_scan() -> None:
    workflow = (Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )

    assert "python tools/check_customer_artifacts.py" in workflow
    assert "--tracked" in workflow
    assert "github.event.pull_request.base.sha" in workflow
    assert "fetch-depth: 0" in workflow
