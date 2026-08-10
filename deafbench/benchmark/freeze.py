"""Integrity verification for immutable benchmark corpus snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")


class FrozenCorpusError(ValueError):
    """Raised when a frozen corpus manifest or artifact is invalid."""


@dataclass(frozen=True)
class FrozenCorpusVerification:
    """Summary of files checked against a frozen corpus manifest."""

    verified_required: int
    verified_optional: int
    missing_optional: tuple[str, ...]


def _artifact_map(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(path, str) and isinstance(digest, str)
        for path, digest in value.items()
    ):
        raise FrozenCorpusError(f"{label} must map paths to SHA-256 values")
    return dict(value)


def _audio_map(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, list):
        raise FrozenCorpusError("audio artifacts must be a list")
    result: dict[str, str] = {}
    for record in value:
        if not isinstance(record, dict):
            raise FrozenCorpusError("audio artifact must be an object")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise FrozenCorpusError("audio artifact needs path and sha256")
        if path in result:
            raise FrozenCorpusError(f"duplicate frozen artifact: {path}")
        result[path] = digest
    return result


def _safe_artifact_path(root: Path, relative: str) -> Path:
    logical = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or logical.is_absolute()
        or ".." in logical.parts
        or "." in logical.parts
    ):
        raise FrozenCorpusError(f"unsafe frozen artifact path: {relative}")
    return root.joinpath(*logical.parts)


def _verify_one(root: Path, relative: str, expected: str) -> None:
    if _SHA256.fullmatch(expected) is None:
        raise FrozenCorpusError(f"invalid SHA-256 for frozen artifact: {relative}")
    path = _safe_artifact_path(root, relative)
    if not path.is_file():
        raise FrozenCorpusError(f"missing frozen artifact: {relative}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise FrozenCorpusError(f"hash mismatch for frozen artifact: {relative}")


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenCorpusError(f"invalid frozen corpus manifest: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise FrozenCorpusError("unsupported frozen corpus manifest")
    return value


def verify_frozen_corpus(
    manifest_path: Path | str,
    repo_root: Path | str,
    *,
    require_optional: bool = False,
) -> FrozenCorpusVerification:
    """Verify frozen files, checking generated evidence whenever it exists."""
    manifest = _load_manifest(Path(manifest_path))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FrozenCorpusError("frozen corpus artifacts must be an object")

    required = _artifact_map(artifacts.get("required", {}), "required artifacts")
    optional = _artifact_map(artifacts.get("optional", {}), "optional artifacts")
    audio = _audio_map(artifacts.get("audio"))
    overlap = set(required) & (set(optional) | set(audio))
    overlap.update(set(optional) & set(audio))
    if overlap:
        raise FrozenCorpusError(
            f"duplicate frozen artifact: {sorted(overlap)[0]}"
        )

    root = Path(repo_root).resolve()
    for relative, expected in required.items():
        _verify_one(root, relative, expected)

    verified_optional = 0
    missing_optional: list[str] = []
    for relative, expected in {**optional, **audio}.items():
        path = _safe_artifact_path(root, relative)
        if not path.is_file():
            missing_optional.append(relative)
            continue
        _verify_one(root, relative, expected)
        verified_optional += 1

    if require_optional and missing_optional:
        raise FrozenCorpusError(
            f"missing frozen artifact: {sorted(missing_optional)[0]}"
        )
    return FrozenCorpusVerification(
        verified_required=len(required),
        verified_optional=verified_optional,
        missing_optional=tuple(sorted(missing_optional)),
    )
