from pathlib import Path

import pytest

from deafbench.pilot.workspace import create_case_workspace, validate_case_root


def test_case_workspace_uses_opaque_id_outside_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    cases = tmp_path / "cases"
    repo.mkdir()

    workspace = create_case_workspace(cases, worktrees=(repo,))

    assert workspace.case_id.startswith("case-")
    assert len(workspace.case_id) == 37
    assert workspace.root == cases / workspace.case_id
    assert workspace.root.is_dir()
    assert {path.name for path in workspace.root.iterdir()} == {
        "input",
        "work",
        "output",
        "temporary",
    }


@pytest.mark.parametrize("relative", ("repo", "repo/cases"))
def test_case_root_rejects_git_worktrees(tmp_path: Path, relative: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(ValueError, match="Git worktree"):
        validate_case_root(tmp_path / relative, worktrees=(repo,))


def test_case_root_rejects_cloud_synced_and_shared_paths(tmp_path: Path) -> None:
    cloud = tmp_path / "OneDrive"
    shared = tmp_path / "shared"

    with pytest.raises(ValueError, match="cloud-synced"):
        validate_case_root(cloud / "cases", cloud_roots=(cloud,))
    with pytest.raises(ValueError, match="shared"):
        validate_case_root(shared / "cases", shared_roots=(shared,))


@pytest.mark.parametrize("relative", (".", "customer-cases"))
def test_case_root_rejects_unsafe_broad_paths(
    tmp_path: Path,
    relative: str,
) -> None:
    with pytest.raises(ValueError, match="broad"):
        validate_case_root(tmp_path / relative, unsafe_roots=(tmp_path,))
