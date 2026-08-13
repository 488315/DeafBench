"""Verified logical deletion for isolated pilot case artifacts."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


_CASE_ID = re.compile(r"case-[0-9a-f]{32}\Z")
DELETION_TARGETS = (
    "input",
    "work",
    "temporary",
    "output/transcripts",
    "output/predictions",
    "output/runs",
    "output/reports",
    "output/sensitive-drafts",
)
_SCAN_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DeletionResult:
    method: str
    categories: tuple[str, ...]
    paths_checked: tuple[str, ...]
    verified: bool


def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((path.resolve(), root.resolve())) == str(root.resolve())
    except ValueError:
        return False


def _contains_case_marker(root: Path, marker: bytes) -> bool:
    if not root.exists():
        return False
    for path in root.rglob("*"):
        if marker.decode() in path.name:
            return True
        if path.is_file():
            try:
                overlap = b""
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(_SCAN_BLOCK_BYTES), b""):
                        combined = overlap + block
                        if marker in combined:
                            return True
                        overlap = combined[-(len(marker) - 1) :]
            except OSError:
                return True
    return False


def logical_delete(
    case_root: Path,
    *,
    case_id: str,
    repository_roots: Iterable[Path] = (),
    backup_roots: Iterable[Path] = (),
    remove_tree: Callable[[Path], None] = shutil.rmtree,
) -> DeletionResult:
    """Delete owned artifact categories and verify all configured locations."""

    root = Path(case_root).resolve(strict=True)
    if not _CASE_ID.fullmatch(case_id) or root.name != case_id:
        raise ValueError("deletion target is not the exact opaque case root")
    targets = tuple(root / relative for relative in DELETION_TARGETS)
    if any(not _within(target, root) for target in targets):
        raise ValueError("deletion target escapes the case root")
    for target in targets:
        if target.exists():
            remove_tree(target)
    remaining = [str(path) for path in targets if path.exists()]
    checked_roots = tuple(Path(path).resolve() for path in (*repository_roots, *backup_roots))
    marker = case_id.encode("ascii")
    contaminated = [str(path) for path in checked_roots if _contains_case_marker(path, marker)]
    if remaining or contaminated:
        raise RuntimeError("verified logical deletion failed")
    return DeletionResult(
        method="verified logical deletion",
        categories=DELETION_TARGETS,
        paths_checked=tuple(map(str, (*targets, *checked_roots))),
        verified=True,
    )
