"""Fail-closed staged-artifact scanning for customer case containment."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_BLOCKED_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_PRIVATE_KEY_SUFFIXES = {".key", ".pem", ".pfx", ".p12"}
_BLOCKED_NAMES = ("transcript", "prediction", "customer-report", "case-artifact")
_CASE_ID = re.compile(rb"case-[0-9a-f]{32}")
_SECRET = re.compile(rb"(?i)(api[_-]?key|password|auth[_-]?token)\s*[:=]\s*\S+")


@dataclass(frozen=True)
class StagedFinding:
    path: str
    reason: str


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def scan_staged(repo: Path) -> tuple[StagedFinding, ...]:
    """Inspect the selected repository's index without reading working files."""

    root = Path(repo).resolve()
    names = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR").decode(
        "utf-8"
    )
    findings: list[StagedFinding] = []
    for name in filter(None, names.splitlines()):
        lowered = name.lower()
        if Path(lowered).suffix in _PRIVATE_KEY_SUFFIXES:
            findings.append(StagedFinding(name, "private key artifact"))
            continue
        if Path(lowered).suffix in _BLOCKED_SUFFIXES or any(
            marker in lowered for marker in _BLOCKED_NAMES
        ):
            findings.append(StagedFinding(name, "customer artifact path"))
            continue
        try:
            content = _git(root, "show", f":{name}")
        except subprocess.CalledProcessError:
            findings.append(StagedFinding(name, "index content could not be inspected"))
            continue
        if _CASE_ID.search(content):
            findings.append(StagedFinding(name, "opaque customer case identifier"))
        elif _SECRET.search(content):
            findings.append(StagedFinding(name, "possible secret"))
    return tuple(findings)
