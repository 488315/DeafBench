"""Hashed, content-free certificates for verified logical deletion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from deafbench.pilot.deletion import DeletionResult


def issue_deletion_certificate(
    path: Path,
    *,
    case_id: str,
    result: DeletionResult,
    operator: str,
    deleted_at: datetime,
    retained_records: Iterable[str],
) -> str:
    """Write a byte-stable certificate and return its SHA-256 digest."""

    if not result.verified:
        raise ValueError("certificate requires verified deletion")
    if deleted_at.tzinfo is None or not operator.strip():
        raise ValueError("certificate requires an operator and timezone")
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "artifact_categories": list(result.categories),
        "paths_checked": list(result.paths_checked),
        "deleted_at": deleted_at.astimezone(timezone.utc).isoformat(),
        "method": result.method,
        "operator": operator,
        "verification_result": "passed",
        "retained_non_sensitive_records": sorted(retained_records),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    document = {**payload, "certificate_sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return digest
