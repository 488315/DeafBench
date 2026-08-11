"""Append-only, content-free event ledger for pilot case operations."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


EVENTS = frozenset(
    {
        "case_creation",
        "validation",
        "model_execution",
        "report_generation",
        "access",
        "retention_change",
        "delivery",
        "deletion",
    }
)
_PROHIBITED_METADATA = frozenset(
    {"customer_name", "email", "transcript", "prediction", "content", "raw_text"}
)
GENESIS_HASH = "0" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify_ledger(path: Path) -> bool:
    previous = GENESIS_HASH
    for sequence, entry in enumerate(_read(path), start=1):
        digest = entry.pop("entry_hash", None)
        if entry.get("sequence") != sequence or entry.get("previous_hash") != previous:
            return False
        if digest != hashlib.sha256(_canonical(entry)).hexdigest():
            return False
        previous = str(digest)
    return True


def append_event(
    path: Path,
    *,
    case_id: str,
    event: str,
    metadata: Mapping[str, str] | None = None,
    occurred_at: datetime | None = None,
) -> str:
    """Append one validated event and synchronously persist its hash chain."""

    if event not in EVENTS:
        raise ValueError("unsupported pilot ledger event")
    details = dict(metadata or {})
    if _PROHIBITED_METADATA.intersection(details):
        raise ValueError("ledger metadata contains a prohibited content field")
    if path.exists() and not verify_ledger(path):
        raise RuntimeError("pilot ledger integrity verification failed")
    existing = _read(path)
    previous = str(existing[-1]["entry_hash"]) if existing else GENESIS_HASH
    timestamp = occurred_at or datetime.now(timezone.utc)
    entry: dict[str, object] = {
        "sequence": len(existing) + 1,
        "occurred_at": timestamp.astimezone(timezone.utc).isoformat(),
        "case_id": case_id,
        "event": event,
        "metadata": details,
        "previous_hash": previous,
    }
    digest = hashlib.sha256(_canonical(entry)).hexdigest()
    entry["entry_hash"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest
