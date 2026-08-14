"""Fail-closed validation for temporary dependency-risk dispositions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from deafbench.remote_code_audit import RemoteCodeAudit, verify_audited_files


_MODEL_ID = "ibm-granite/granite-speech-4.1-2b-nar"
_MODEL_REVISION = "a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2"
_AUDIT_RESOURCE = "granite-speech-4.1-2b-nar.json"
_STACK = {"torch": "2.9.1", "torchaudio": "2.9.1", "torchcodec": "0.9.1"}
_REVIEW_BY = date(2026, 11, 13)
_ADVISORIES = {
    15: ("GHSA-qfhq-4f3w-5fph", "CVE-2025-3001", ("torch.lstm_cell",)),
    16: ("GHSA-rrmf-rvhw-rf47", "CVE-2025-3000", ("torch.jit.script",)),
}
_OPEN_ASR_REACHABILITY = {
    "torch.lstm_cell": "absent_from_pinned_evaluation_path",
    "torch.jit.script": "present_only_in_non_evaluation_tools",
}
_OPEN_ASR_AFFECTED_FILES = [
    "egs/librispeech/ASR/zipformer/export-onnx.py",
    "egs/librispeech/ASR/zipformer/export-onnx-streaming.py",
    "egs/librispeech/ASR/zipformer/export-streaming-as-non-streaming-onnx.py",
    "egs/librispeech/ASR/zipformer/export.py",
    "egs/librispeech/ASR/zipformer/scaling_converter.py",
]
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LOCK_VERSION = re.compile(
    r"(?m)^(?P<package>torch|torchaudio|k2) @ .*?/"
    r"(?P=package)-(?P<version>\d[^/]+?)-cp\d"
)


class DependencyDispositionError(ValueError):
    """Raised when a dependency disposition is missing, stale, or changed."""


@dataclass(frozen=True)
class DependencyDisposition:
    """A time-bounded assessment for one exact dependency advisory."""

    alert_number: int
    package: str
    status: str
    affected_apis: tuple[str, ...]
    review_by: date
    compatible_stack: dict[str, str]


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DependencyDispositionError(f"invalid disposition {field}")
    return value


def _iso_date(value: object, field: str) -> date:
    if not isinstance(value, str):
        raise DependencyDispositionError(f"invalid disposition {field}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DependencyDispositionError(f"invalid disposition {field}") from exc


def validate_dependency_disposition(
    payload: object, *, today: date | None = None
) -> DependencyDisposition:
    """Validate one exact, unexpired Torch development-risk assessment."""
    record = _mapping(payload, "record")
    if record.get("schema_version") != 1:
        raise DependencyDispositionError("unsupported disposition schema")
    alert_number = record.get("alert_number")
    if alert_number not in _ADVISORIES:
        raise DependencyDispositionError("unexpected disposition alert")
    if record.get("package") != "torch" or record.get("manifest") != "pyproject.toml":
        raise DependencyDispositionError("invalid disposition package or manifest")
    if record.get("dependency_scope") != "development":
        raise DependencyDispositionError("invalid disposition scope")
    if record.get("installed_version") != _STACK["torch"]:
        raise DependencyDispositionError("invalid disposition installed version")
    if record.get("status") != "tolerable_risk":
        raise DependencyDispositionError("invalid disposition status")

    advisory = _mapping(record.get("advisory"), "advisory")
    ghsa, cve, affected_apis = _ADVISORIES[alert_number]
    if advisory != {"ghsa": ghsa, "cve": cve, "affected_apis": list(affected_apis)}:
        raise DependencyDispositionError("invalid disposition advisory")

    reviewed_utc = _iso_date(record.get("reviewed_utc"), "reviewed_utc")
    review_by = _iso_date(record.get("review_by"), "review_by")
    if review_by != _REVIEW_BY:
        raise DependencyDispositionError("invalid disposition review deadline")
    if review_by <= reviewed_utc or review_by < (today or date.today()):
        raise DependencyDispositionError("dependency disposition expired")

    model = _mapping(record.get("model"), "model")
    if model.get("id") != _MODEL_ID:
        raise DependencyDispositionError("invalid disposition model")
    if model.get("revision") != _MODEL_REVISION:
        raise DependencyDispositionError("invalid disposition model revision")
    if model.get("audit_resource") != _AUDIT_RESOURCE:
        raise DependencyDispositionError("invalid disposition audit resource")
    stack = _mapping(record.get("compatible_stack"), "compatible_stack")
    if stack != _STACK:
        raise DependencyDispositionError("invalid disposition compatible stack")
    reachability = _mapping(record.get("reachability"), "reachability")
    if reachability != {"first_party": False, "audited_remote_code": False}:
        raise DependencyDispositionError("invalid disposition reachability")

    return DependencyDisposition(
        alert_number=alert_number,
        package="torch",
        status="tolerable_risk",
        affected_apis=affected_apis,
        review_by=review_by,
        compatible_stack=dict(stack),
    )


def _validate_reviewed_hashes(payload: Mapping[str, Any]) -> None:
    hashes = _mapping(payload.get("reviewed_source_sha256"), "reviewed_source_sha256")
    if not hashes or any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        for path, digest in hashes.items()
    ):
        raise DependencyDispositionError("invalid reviewed source hashes")
    audit_resource = files("deafbench").joinpath(
        "remote-code-audits", _AUDIT_RESOURCE
    )
    try:
        audit = json.loads(audit_resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyDispositionError("remote-code audit is unavailable") from exc
    audited_hashes = {item["path"]: item["sha256"] for item in audit.get("files", ())}
    if hashes != audited_hashes:
        raise DependencyDispositionError("reviewed source differs from remote-code audit")


def _external_lane(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    lanes = payload.get("external_lanes")
    if (
        not isinstance(lanes, Sequence)
        or isinstance(lanes, (str, bytes))
        or len(lanes) != 1
    ):
        raise DependencyDispositionError("invalid external dependency lane")
    return _mapping(lanes[0], "external dependency lane")


def _open_asr_stack(lock_path: Path) -> dict[str, str]:
    try:
        lock_text = lock_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DependencyDispositionError("Open-ASR lock is unavailable") from exc
    versions = {
        match.group("package"): match.group("version").replace("%2B", "+")
        for match in _LOCK_VERSION.finditer(lock_text)
    }
    if set(versions) != {"torch", "torchaudio", "k2"}:
        raise DependencyDispositionError("Open-ASR ABI stack is incomplete")
    return {
        "torch": versions["torch"].split("+", maxsplit=1)[0],
        "torchaudio": versions["torchaudio"].split("+", maxsplit=1)[0],
        "k2": versions["k2"],
    }


def verify_open_asr_dependency_disposition(lock_path: Path) -> None:
    """Bind the recorded Open-ASR risk to live runner and lock authorities."""
    from deafbench.leaderboard.zipformer_runner import (
        ICEFALL_REVISION,
        ZIPFORMER_RUNNER_REVISION,
    )

    payload = _load_registry()
    expected = {
        "name": "open-asr-zipformer",
        "runner_revision": ZIPFORMER_RUNNER_REVISION,
        "icefall_revision": ICEFALL_REVISION,
        "compatible_stack": _open_asr_stack(lock_path),
        "reachability": _OPEN_ASR_REACHABILITY,
        "non_evaluation_affected_files": _OPEN_ASR_AFFECTED_FILES,
    }
    if _external_lane(payload) != expected:
        raise DependencyDispositionError("Open-ASR disposition differs from runtime")


def _load_registry() -> Mapping[str, Any]:
    resource = files("deafbench").joinpath("dependency-risk-dispositions.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyDispositionError("dependency dispositions are unavailable") from exc
    root = _mapping(payload, "registry")
    if root.get("schema_version") != 1:
        raise DependencyDispositionError("unsupported disposition registry schema")
    return root


def load_dependency_dispositions(
    *, today: date | None = None
) -> tuple[DependencyDisposition, ...]:
    """Load all packaged dispositions and reject omissions or duplicate alerts."""
    root = _load_registry()
    _validate_reviewed_hashes(root)
    _external_lane(root)
    records = root.get("dispositions")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise DependencyDispositionError("invalid disposition collection")
    dispositions = tuple(
        validate_dependency_disposition(item, today=today) for item in records
    )
    alert_numbers = [item.alert_number for item in dispositions]
    if set(alert_numbers) != set(_ADVISORIES) or len(alert_numbers) != len(
        set(alert_numbers)
    ):
        raise DependencyDispositionError("incomplete or duplicate dispositions")
    return dispositions


def verify_dependency_disposition_snapshot(
    audit: RemoteCodeAudit,
    snapshot_root: Path,
) -> None:
    """Reject changed or affected remote source before dependency execution."""
    if audit.model_id != _MODEL_ID or audit.revision != _MODEL_REVISION:
        raise DependencyDispositionError("dependency disposition audit differs")
    verify_audited_files(audit, snapshot_root)
    affected_apis = {
        api for item in load_dependency_dispositions() for api in item.affected_apis
    }
    root = snapshot_root.resolve(strict=True)
    for audited_file in audit.audited_files:
        if not audited_file.path.endswith(".py"):
            continue
        source_path = root.joinpath(*PurePosixPath(audited_file.path).parts)
        try:
            source = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DependencyDispositionError(
                f"audited dependency source is unreadable: {audited_file.path}"
            ) from exc
        matched = sorted(api for api in affected_apis if api in source)
        if matched:
            raise DependencyDispositionError(
                f"affected dependency API is reachable: {audited_file.path}: "
                + ", ".join(matched)
            )
