"""Measurable Windows storage protections for isolated pilot cases."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ProtectionState:
    verified: bool
    evidence: str


@dataclass(frozen=True)
class StorageProtection:
    volume_protection: str
    account_acl_restricted: bool


def probe_bitlocker(path: Path) -> ProtectionState:
    """Measure BitLocker status for the volume containing ``path``."""

    drive = Path(path).resolve().drive
    completed = subprocess.run(
        ["manage-bde", "-status", drive],
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join((completed.stdout, completed.stderr))
    active = completed.returncode == 0 and "Protection On" in output
    return ProtectionState(active, "BitLocker Protection On" if active else output.strip())


def restrict_acl_to_current_account(path: Path) -> bool:
    """Replace inherited access with the current Windows account and SYSTEM."""

    account = os.environ.get("USERNAME")
    if not account:
        return False
    completed = subprocess.run(
        [
            "icacls",
            str(Path(path).resolve()),
            "/inheritance:r",
            "/grant:r",
            f"{account}:(OI)(CI)F",
            "/grant:r",
            "SYSTEM:(OI)(CI)F",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and "Successfully processed 1 files" in completed.stdout


def protect_case_storage(
    case_root: Path,
    *,
    protection_probe: Callable[[Path], ProtectionState] = probe_bitlocker,
    acl_restrictor: Callable[[Path], bool] = restrict_acl_to_current_account,
) -> StorageProtection:
    """Apply account isolation only after at-rest protection is measured."""

    root = Path(case_root).resolve(strict=True)
    state = protection_probe(root)
    if not state.verified:
        raise RuntimeError("at-rest volume protection is not verified")
    if not acl_restrictor(root):
        raise RuntimeError("account-only ACL restriction was not verified")
    return StorageProtection(state.evidence, True)
