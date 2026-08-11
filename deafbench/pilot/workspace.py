"""Isolated filesystem ownership for founding-pilot cases."""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CASE_DIRECTORIES = ("input", "work", "output", "temporary")


@dataclass(frozen=True)
class CaseWorkspace:
    """Opaque case identity and its isolated local directories."""

    case_id: str
    root: Path


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_same_or_within(path: Path, parent: Path) -> bool:
    path_resolved = _resolved(path)
    parent_resolved = _resolved(parent)
    try:
        return os.path.commonpath((path_resolved, parent_resolved)) == str(
            parent_resolved
        )
    except ValueError:
        return False


def _overlaps(left: Path, right: Path) -> bool:
    return _is_same_or_within(left, right) or _is_same_or_within(right, left)


def discover_git_worktrees(repo_root: Path) -> tuple[Path, ...]:
    """Return every worktree registered by the repository."""
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "list", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        _resolved(Path(line.removeprefix("worktree ")))
        for line in completed.stdout.splitlines()
        if line.startswith("worktree ")
    )


def _environment_cloud_roots() -> tuple[Path, ...]:
    names = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
    return tuple(
        _resolved(Path(value))
        for name in names
        if (value := os.environ.get(name))
    )


def validate_case_root(
    root: Path,
    *,
    worktrees: Iterable[Path] = (),
    cloud_roots: Iterable[Path] | None = None,
    shared_roots: Iterable[Path] = (),
    unsafe_roots: Iterable[Path] | None = None,
) -> Path:
    """Resolve an isolated case root or reject unsafe storage."""
    candidate = _resolved(root)
    unsafe = tuple(unsafe_roots) if unsafe_roots is not None else (Path.home(),)
    clouds = (
        tuple(cloud_roots)
        if cloud_roots is not None
        else _environment_cloud_roots()
    )

    if candidate == Path(candidate.anchor) or any(
        _is_same_or_within(path, candidate) for path in unsafe
    ):
        raise ValueError("Case root is an unsafe broad path")
    if any(_overlaps(candidate, path) for path in worktrees):
        raise ValueError("Case root overlaps a Git worktree")
    if candidate.anchor.startswith("\\\\") or any(
        _overlaps(candidate, path) for path in shared_roots
    ):
        raise ValueError("Case root is on shared storage")
    if any(_overlaps(candidate, path) for path in clouds):
        raise ValueError("Case root is in a cloud-synced folder")
    return candidate


def create_case_workspace(
    base_root: Path,
    *,
    worktrees: Iterable[Path] = (),
    cloud_roots: Iterable[Path] | None = None,
    shared_roots: Iterable[Path] = (),
    unsafe_roots: Iterable[Path] | None = None,
) -> CaseWorkspace:
    """Create an opaque case without accepting customer naming input."""
    base = validate_case_root(
        base_root,
        worktrees=worktrees,
        cloud_roots=cloud_roots,
        shared_roots=shared_roots,
        unsafe_roots=unsafe_roots,
    )
    base.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        case_id = f"case-{uuid.uuid4().hex}"
        root = base / case_id
        try:
            root.mkdir()
        except FileExistsError:
            continue
        for name in CASE_DIRECTORIES:
            (root / name).mkdir()
        return CaseWorkspace(case_id=case_id, root=root)
    raise RuntimeError("Unable to allocate a unique case ID")
