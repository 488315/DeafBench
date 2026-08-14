"""Fail-closed validation for temporary dependency-risk dispositions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from typing import Any, Mapping, Sequence


_MODEL_ID = "ibm-granite/granite-speech-4.1-2b-nar"
_MODEL_REVISION = "a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2"
_AUDIT_RESOURCE = "granite-speech-4.1-2b-nar.json"
_STACK = {"torch": "2.9.1", "torchaudio": "2.9.1", "torchcodec": "0.9.1"}
_ADVISORIES = {
    15: ("GHSA-qfhq-4f3w-5fph", "CVE-2025-3001", ("torch.lstm_cell",)),
    16: ("GHSA-rrmf-rvhw-rf47", "CVE-2025-3000", ("torch.jit.script",)),
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


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


def load_dependency_dispositions(
    *, today: date | None = None
) -> tuple[DependencyDisposition, ...]:
    """Load all packaged dispositions and reject omissions or duplicate alerts."""
    resource = files("deafbench").joinpath("dependency-risk-dispositions.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyDispositionError("dependency dispositions are unavailable") from exc
    root = _mapping(payload, "registry")
    if root.get("schema_version") != 1:
        raise DependencyDispositionError("unsupported disposition registry schema")
    _validate_reviewed_hashes(root)
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
