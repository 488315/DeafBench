"""Append-only, content-free event ledger for pilot case operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping


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
_CASE_ID = re.compile(r"case-[0-9a-f]{32}\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}\Z")
_METADATA_FIELDS = {
    "case_creation": frozenset(),
    "validation": frozenset(),
    "model_execution": frozenset({"model_id"}),
    "report_generation": frozenset(),
    "access": frozenset(),
    "retention_change": frozenset(),
    "delivery": frozenset(),
    "deletion": frozenset(),
}
GENESIS_HASH = "0" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@contextmanager
def _locked_ledger(path: Path) -> Iterator[None]:
    """Serialize validation and appends across local processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8", newline="\n") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            def unlock() -> None:
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            def unlock() -> None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            unlock()


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
    if _CASE_ID.fullmatch(case_id) is None:
        raise ValueError("ledger case identifier must be opaque")
    details = dict(metadata or {})
    if not set(details).issubset(_METADATA_FIELDS[event]):
        raise ValueError("ledger event contains unsupported metadata fields")
    if "model_id" in details and _MODEL_ID.fullmatch(details["model_id"]) is None:
        raise ValueError("ledger metadata value is invalid")
    with _locked_ledger(path):
        if not verify_ledger(path):
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
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return digest
