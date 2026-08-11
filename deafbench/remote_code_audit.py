"""Fail-closed validation for pinned third-party model source code."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from deafbench.model_registry import get_model_license


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_POLICY = {
    "allow_network_during_inference": False,
    "isolate_from_main_process": True,
    "require_exact_file_hashes": True,
    "trust_remote_code": True,
}


class RemoteCodeAuditError(ValueError):
    """Raised when audited model source is absent, changed, or unsafe."""


@dataclass(frozen=True)
class AuditedFile:
    """Expected digest for one reviewed file in a pinned model snapshot."""

    path: str
    sha256: str


@dataclass(frozen=True)
class RemoteCodeAudit:
    """Validated execution policy and file allowlist for one model revision."""

    model_id: str
    revision: str
    audited_files: tuple[AuditedFile, ...]


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RemoteCodeAuditError(f"invalid remote-code audit {field}")
    return value


def validate_remote_code_audit(payload: object) -> RemoteCodeAudit:
    """Validate one audit and bind it to registered model licensing metadata."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise RemoteCodeAuditError("unsupported remote-code audit schema")
    model_id = _required_text(payload, "model_id")
    revision = _required_text(payload, "revision")
    policy = payload.get("execution_policy")
    if policy != _REQUIRED_POLICY:
        raise RemoteCodeAuditError(f"unsafe remote-code policy for model: {model_id}")

    raw_files = payload.get("files")
    if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
        raise RemoteCodeAuditError(f"invalid audited files for model: {model_id}")
    audited_files: list[AuditedFile] = []
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping):
            raise RemoteCodeAuditError(f"invalid audited file for model: {model_id}")
        path = _required_text(raw_file, "path")
        digest = _required_text(raw_file, "sha256")
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != path
            or _SHA256.fullmatch(digest) is None
        ):
            raise RemoteCodeAuditError(f"unsafe audited file for model: {model_id}")
        audited_files.append(AuditedFile(path, digest))
    paths = [record.path for record in audited_files]
    if not paths or len(paths) != len(set(paths)):
        raise RemoteCodeAuditError(f"invalid audited file set for model: {model_id}")

    model_license = get_model_license(model_id)
    if not model_license.remote_code_required:
        raise RemoteCodeAuditError(f"model does not require remote code: {model_id}")
    if model_license.revision != revision:
        raise RemoteCodeAuditError(f"audit revision differs from registry: {model_id}")
    return RemoteCodeAudit(model_id, revision, tuple(audited_files))


def load_remote_code_audit(model_id: str) -> RemoteCodeAudit:
    """Load the unique packaged audit for a registered remote-code model."""
    audit_root = files("deafbench").joinpath("remote-code-audits")
    matches: list[RemoteCodeAudit] = []
    try:
        resources = tuple(audit_root.iterdir())
    except OSError as exc:
        raise RemoteCodeAuditError("remote-code audit registry is unavailable") from exc
    for resource in resources:
        if not resource.name.endswith(".json"):
            continue
        try:
            payload = json.loads(resource.read_text(encoding="utf-8"))
            audit = validate_remote_code_audit(payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise RemoteCodeAuditError(
                f"invalid remote-code audit resource: {resource.name}"
            ) from exc
        if audit.model_id == model_id:
            matches.append(audit)
    if len(matches) != 1:
        raise RemoteCodeAuditError(
            f"expected one remote-code audit for model: {model_id}"
        )
    return matches[0]


def verify_audited_files(audit: RemoteCodeAudit, snapshot_root: Path) -> None:
    """Reject a snapshot when any reviewed source file is missing or changed."""
    root = snapshot_root.resolve(strict=True)
    for audited_file in audit.audited_files:
        candidate = root.joinpath(*PurePosixPath(audited_file.path).parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            payload = resolved.read_bytes()
        except (OSError, ValueError) as exc:
            raise RemoteCodeAuditError(
                f"missing audited file for model {audit.model_id}: {audited_file.path}"
            ) from exc
        observed = hashlib.sha256(payload).hexdigest()
        if observed != audited_file.sha256:
            raise RemoteCodeAuditError(
                f"audited file hash mismatch for model {audit.model_id}: "
                f"{audited_file.path}"
            )
