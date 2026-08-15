import tomllib
from pathlib import Path

import deafbench


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == "0.2.2"
    assert deafbench.__version__ == "0.2.2"


def test_package_links_to_public_project_surfaces() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["urls"] == {
        "Homepage": "https://488315.github.io/products/deafbench/",
        "Documentation": "https://github.com/488315/DeafBench#readme",
        "Source": "https://github.com/488315/DeafBench",
        "Changelog": "https://github.com/488315/DeafBench/blob/main/CHANGELOG.md",
        "Issues": "https://github.com/488315/DeafBench/issues",
    }


def _release_section(changelog: str, heading: str) -> str:
    assert heading in changelog
    release_start = changelog.index(heading)
    next_heading = changelog.find("\n## ", release_start + len(heading))
    return changelog[release_start : next_heading if next_heading != -1 else None]


def test_changelog_declares_current_release() -> None:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release = _release_section(changelog, "## 2026-08-15 / v0.2.2")

    for expected in (
        "deafbench[audit]",
        "deafbench audit <case-folder>",
        "deafbench review <case-folder>",
        "peak_vram_bytes",
        "Unicode-capable fonts",
        "latest successful run",
    ):
        assert expected in release


def test_v021_changelog_keeps_security_dispositions() -> None:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release = _release_section(changelog, "## 2026-08-14 / v0.2.1")

    for advisory in (
        "GHSA-69w3-r845-3855",
        "GHSA-29pf-2h5f-8g72",
        "GHSA-fgcw-684q-jj6r",
        "GHSA-qfhq-4f3w-5fph",
        "GHSA-rrmf-rvhw-rf47",
    ):
        assert advisory in release
    assert "The two development Torch alerts remain open" in release
