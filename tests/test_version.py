import tomllib
from pathlib import Path

import deafbench


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == "0.2.1"
    assert deafbench.__version__ == "0.2.1"


def test_package_links_to_public_project_surfaces() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["urls"] == {
        "Homepage": "https://488315.github.io/products/deafbench/",
        "Documentation": "https://github.com/488315/DeafBench#readme",
        "Source": "https://github.com/488315/DeafBench",
        "Changelog": "https://github.com/488315/DeafBench/blob/main/CHANGELOG.md",
        "Issues": "https://github.com/488315/DeafBench/issues",
    }


def test_changelog_declares_current_release() -> None:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "## 2026-08-14 / v0.2.1" in changelog
    assert "GHSA-69w3-r845-3855" in changelog
    assert "GHSA-29pf-2h5f-8g72" in changelog
    assert "GHSA-fgcw-684q-jj6r" in changelog
    assert "GHSA-qfhq-4f3w-5fph" in changelog
    assert "GHSA-rrmf-rvhw-rf47" in changelog
