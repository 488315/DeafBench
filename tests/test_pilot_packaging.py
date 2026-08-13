import tomllib
from pathlib import Path


def test_zero_custody_extra_is_isolated() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["optional-dependencies"]["zero-custody-pilot"] == [
        "cryptography>=48.0,<49.0"
    ]
    assert "cryptography>=48.0,<49.0" not in project["dependencies"]
