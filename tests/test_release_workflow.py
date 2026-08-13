from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_manual_release_workflow_cannot_publish_to_pypi() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    publish_job = workflow.split("\n  publish:\n", maxsplit=1)[1]

    assert "workflow_dispatch:" in workflow
    assert (
        "if: github.event_name == 'release' && "
        "github.event.action == 'published'" in publish_job
    )
