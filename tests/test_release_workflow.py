from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_manual_release_workflow_cannot_publish_to_pypi() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
    )
    # PyYAML's YAML 1.1 resolver parses the unquoted GitHub Actions `on` key
    # as True.
    triggers = workflow[True]
    publish_job = workflow["jobs"]["publish"]

    assert triggers == {"release": {"types": ["published"]}, "workflow_dispatch": None}
    assert publish_job["if"] == (
        "github.event_name == 'release' && github.event.action == 'published' "
        "&& vars.DEAFBENCH_PYPI_PUBLISH == 'true'"
    )
    assert publish_job["environment"]["name"] == "pypi"
