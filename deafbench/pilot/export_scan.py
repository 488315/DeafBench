"""Fail-closed scanning for artifacts leaving a customer computer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_ALLOWED_ARTIFACTS = frozenset({"manifest.json", "report.md"})
_PROHIBITED_FIELDS = frozenset(
    {
        "audio",
        "audio_path",
        "case_id",
        "critical_value",
        "critical_information_value",
        "filename",
        "filepath",
        "path",
        "prediction",
        "predictions",
        "raw_text",
        "sample",
        "sample_id",
        "samples",
        "speaker",
        "speaker_identity",
        "term",
        "text",
        "transcript",
        "transcripts",
    }
)
_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|password|auth[_-]?token|secret)\s*[:=]\s*\S+"
)
_LOCAL_ARTIFACT = re.compile(
    r"(?i)(?:[a-z]:[\\/]|\\\\|(?:^|\s)/(?:home|users|mnt|tmp)/|"
    r"(?:^|[\\/])\.\.(?:[\\/])|[^\s]+\.(?:wav|mp3|flac|m4a|ogg)(?:\s|$))"
)
_IDENTIFIER = re.compile(r"(?i)\b(?:case-[0-9a-f]{32}|core-\d{3,})\b")
_REPORT_CONTENT = re.compile(
    r"(?i)\b(?:transcript|speaker identity|file(?:name|path)|sample id|"
    r"critical(?:-information)? value)\s*[:=]"
)


@dataclass(frozen=True)
class ExportFinding:
    artifact: str
    reason: str


def _scan_string(artifact: str, value: str) -> Iterable[ExportFinding]:
    if _SECRET.search(value):
        yield ExportFinding(artifact, "possible secret")
    if _LOCAL_ARTIFACT.search(value):
        yield ExportFinding(artifact, "local path or filename")
    if _IDENTIFIER.search(value):
        yield ExportFinding(artifact, "case or sample identifier")


def _scan_json(artifact: str, value: object) -> Iterable[ExportFinding]:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _PROHIBITED_FIELDS:
                yield ExportFinding(artifact, f"prohibited field: {key}")
            yield from _scan_json(artifact, item)
    elif isinstance(value, list):
        for item in value:
            yield from _scan_json(artifact, item)
    elif isinstance(value, str):
        yield from _scan_string(artifact, value)


def scan_export_directory(root: Path) -> tuple[ExportFinding, ...]:
    """Inspect every export artifact and reject unknown or sensitive content."""

    directory = Path(root)
    if not directory.is_dir():
        return (ExportFinding(str(directory), "export directory is unreadable"),)
    findings: list[ExportFinding] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name not in _ALLOWED_ARTIFACTS:
            findings.append(ExportFinding(path.name, "unexpected export artifact"))
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            findings.append(ExportFinding(path.name, "unreadable export artifact"))
            continue
        if path.suffix == ".json":
            try:
                value = json.loads(content)
            except json.JSONDecodeError:
                findings.append(ExportFinding(path.name, "unreadable export artifact"))
                continue
            findings.extend(_scan_json(path.name, value))
        else:
            if _REPORT_CONTENT.search(content):
                findings.append(ExportFinding(path.name, "prohibited report content"))
            findings.extend(_scan_string(path.name, content))
    return tuple(findings)


def assert_export_safe(root: Path) -> None:
    """Block export when any artifact cannot be proven aggregate-only."""

    findings = scan_export_directory(root)
    if findings:
        reasons = ", ".join(
            f"{finding.artifact}: {finding.reason}" for finding in findings
        )
        raise ValueError(f"export blocked: {reasons}")
