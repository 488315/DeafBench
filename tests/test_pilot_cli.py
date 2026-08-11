import json
from pathlib import Path

import pytest

from deafbench.pilot.cli import main
from deafbench.pilot.rehearsal import RehearsalResult


def test_rehearsal_cli_uses_measured_storage_controls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = {}

    def runner(**kwargs: object) -> RehearsalResult:
        captured.update(kwargs)
        return RehearsalResult("case-test", 3, True, "a" * 64, True)

    assert (
        main(
            [
                "rehearse",
                "--repo-root",
                str(tmp_path),
                "--case-base",
                str(tmp_path / "cases"),
                "--records-root",
                str(tmp_path / "records"),
                "--operator",
                "operator",
            ],
            runner=runner,
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["model_count"] == 3
    assert callable(captured["protection_probe"])
    assert callable(captured["acl_restrictor"])
