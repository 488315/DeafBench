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
_PRIVATE_KEY = re.compile(
    rb"-----BEGIN (?:ENCRYPTED |RSA |EC |OPENSSH )?PRIVATE KEY-----"
)


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


def _inspect(root: Path, names: list[str], *, source: str) -> tuple[StagedFinding, ...]:
    findings: list[StagedFinding] = []
    for name in names:
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
            content = _git(root, "show", f"{source}:{name}")
        except subprocess.CalledProcessError:
            findings.append(StagedFinding(name, "content could not be inspected"))
            continue
        if _PRIVATE_KEY.search(content):
            findings.append(StagedFinding(name, "private key artifact"))
        elif _CASE_ID.search(content):
            findings.append(StagedFinding(name, "opaque customer case identifier"))
        elif _SECRET.search(content):
            findings.append(StagedFinding(name, "possible secret"))
    return tuple(findings)


def scan_staged(repo: Path) -> tuple[StagedFinding, ...]:
    """Inspect the selected repository's index without reading working files."""

    root = Path(repo).resolve()
    names = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR").decode(
        "utf-8"
    )
    return _inspect(root, list(filter(None, names.splitlines())), source="")


def scan_tracked(repo: Path, *, base: str | None = None) -> tuple[StagedFinding, ...]:
    """Inspect tracked paths introduced or changed since an optional CI base."""

    root = Path(repo).resolve()
    if base is None:
        names = _git(root, "ls-tree", "-r", "--name-only", "HEAD").decode("utf-8")
    else:
        names = _git(
            root, "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"
        ).decode("utf-8")
    return _inspect(root, list(filter(None, names.splitlines())), source="HEAD")
