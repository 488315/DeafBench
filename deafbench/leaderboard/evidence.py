"""Fail-closed checks for frozen Open ASR evaluation evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class EvidenceIntegrityError(RuntimeError):
    """Raised when frozen evaluation evidence cannot be trusted."""


@dataclass(frozen=True)
class EvidenceVerification:
    """Summary returned after every frozen artifact passes verification."""

    artifact_count: int
    result_rows: int
    composite_wer: float
    hardware_label: str


def _read_manifest(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceIntegrityError(f"cannot read evidence manifest: {source}") from exc
    if not isinstance(value, dict):
        raise EvidenceIntegrityError("evidence manifest must contain an object")
    return value


def _artifact_path(repo_root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise EvidenceIntegrityError("artifact path must be a nonempty string")
    candidate = (repo_root / relative).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise EvidenceIntegrityError(f"artifact escapes repository: {relative}") from exc
    return candidate


def verify_evidence_manifest(
    manifest_path: Path | str,
    *,
    repo_root: Path | str,
) -> EvidenceVerification:
    """Verify every byte and declared row count in a frozen evidence manifest."""
    manifest = _read_manifest(manifest_path)
    root = Path(repo_root).resolve()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EvidenceIntegrityError("evidence manifest has no artifacts")

    result_rows = 0
    seen: set[Path] = set()
    verified_payloads: dict[Path, bytes] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise EvidenceIntegrityError("artifact entry must be an object")
        relative = entry.get("path")
        path = _artifact_path(root, relative)
        if path in seen:
            raise EvidenceIntegrityError(f"duplicate artifact: {relative}")
        seen.add(path)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise EvidenceIntegrityError(f"missing evidence artifact: {relative}") from exc
        if entry.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise EvidenceIntegrityError(f"artifact hash mismatch: {relative}")
        if entry.get("bytes") != len(payload):
            raise EvidenceIntegrityError(f"artifact size mismatch: {relative}")
        verified_payloads[path] = payload
        if "rows" in entry:
            actual_rows = len(payload.splitlines())
            if entry["rows"] != actual_rows:
                raise EvidenceIntegrityError(f"artifact row mismatch: {relative}")
            result_rows += actual_rows

    metrics = manifest.get("metrics")
    hardware = manifest.get("hardware")
    if not isinstance(metrics, dict) or not isinstance(hardware, dict):
        raise EvidenceIntegrityError("evidence summary metadata is incomplete")
    composite_wer = metrics.get("public_seven_set_macro_wer")
    score_relative = metrics.get("score_artifact")
    model_id = metrics.get("model_id")
    hardware_label = hardware.get("label")
    if not isinstance(composite_wer, (int, float)):
        raise EvidenceIntegrityError("missing public seven-set macro WER")
    if isinstance(composite_wer, bool) or not math.isfinite(composite_wer):
        raise EvidenceIntegrityError("public seven-set macro WER must be finite")
    score_path = _artifact_path(root, score_relative)
    try:
        score = json.loads(verified_payloads[score_path])
        score_wer = score["composite_wer"][model_id]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise EvidenceIntegrityError("cannot read score artifact summary") from exc
    if (
        isinstance(score_wer, bool)
        or not isinstance(score_wer, (int, float))
        or not math.isfinite(score_wer)
    ):
        raise EvidenceIntegrityError("score artifact macro WER must be finite")
    if float(score_wer) != float(composite_wer):
        raise EvidenceIntegrityError("declared WER does not match score artifact")
    if not isinstance(hardware_label, str) or not hardware_label.startswith("local "):
        raise EvidenceIntegrityError("hardware result must be labeled local")
    if manifest.get("result_rows") != result_rows:
        raise EvidenceIntegrityError("aggregate result row count mismatch")

    return EvidenceVerification(
        artifact_count=len(artifacts),
        result_rows=result_rows,
        composite_wer=float(composite_wer),
        hardware_label=hardware_label,
    )
