from pathlib import Path

import pytest

from deafbench.pilot.deletion import DELETION_TARGETS, logical_delete


CASE_ID = "case-" + "a" * 32


def _case(tmp_path: Path) -> Path:
    root = tmp_path / CASE_ID
    for relative in DELETION_TARGETS:
        target = root / relative
        target.mkdir(parents=True, exist_ok=True)
        (target / "artifact.bin").write_bytes(b"synthetic customer artifact")
    return root


def test_logical_deletion_removes_every_sensitive_category(tmp_path: Path) -> None:
    root = _case(tmp_path)
    repository = tmp_path / "repository"
    backup = tmp_path / "backup"
    repository.mkdir()
    backup.mkdir()

    result = logical_delete(
        root, case_id=CASE_ID, repository_roots=[repository], backup_roots=[backup]
    )

    assert result.verified is True
    assert result.method == "verified logical deletion"
    assert all(not (root / relative).exists() for relative in DELETION_TARGETS)


def test_deletion_fails_if_repository_or_backup_contains_case_marker(
    tmp_path: Path,
) -> None:
    root = _case(tmp_path)
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "record.txt").write_text(CASE_ID, encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed"):
        logical_delete(root, case_id=CASE_ID, backup_roots=[backup])


def test_deletion_detects_marker_across_scan_blocks(tmp_path: Path) -> None:
    root = _case(tmp_path)
    backup = tmp_path / "backup"
    backup.mkdir()
    prefix = b"x" * (1024 * 1024 - 4)
    (backup / "record.bin").write_bytes(prefix + CASE_ID.encode("ascii"))

    with pytest.raises(RuntimeError, match="failed"):
        logical_delete(root, case_id=CASE_ID, backup_roots=[backup])


def test_deletion_failure_blocks_success_result(tmp_path: Path) -> None:
    root = _case(tmp_path)

    def fail(_: Path) -> None:
        raise OSError("simulated locked file")

    with pytest.raises(OSError, match="locked"):
        logical_delete(root, case_id=CASE_ID, remove_tree=fail)


def test_deletion_rejects_non_case_and_mismatched_roots(tmp_path: Path) -> None:
    root = tmp_path / "broad"
    root.mkdir()
    with pytest.raises(ValueError, match="exact"):
        logical_delete(root, case_id=CASE_ID)
