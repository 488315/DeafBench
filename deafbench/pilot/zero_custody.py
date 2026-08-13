"""Fail-closed declaration for customer-run, zero-custody execution."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "execution_mode",
        "customer_authorized_computer",
        "customer_audio_uploaded",
        "customer_audio_transferred_to_deafbench",
        "remote_shell_enabled",
        "unattended_access_enabled",
        "credentials_shared",
        "aggregate_only_export",
    }
)


@dataclass(frozen=True)
class ExecutionAttestation:
    """Measured inputs to the zero-custody processing boundary."""

    execution_mode: str
    aggregate_only_export: bool
    sha256: str


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def validate_execution_attestation(
    attestation: ExecutionAttestation,
) -> ExecutionAttestation:
    """Validate a loaded attestation again at a processing trust boundary."""
    if (
        not isinstance(attestation, ExecutionAttestation)
        or attestation.execution_mode != "customer_run"
        or attestation.aggregate_only_export is not True
        or not isinstance(attestation.sha256, str)
        or _SHA256.fullmatch(attestation.sha256) is None
    ):
        raise ValueError("zero-custody execution attestation is invalid")
    return attestation


def load_execution_attestation(path: Path) -> ExecutionAttestation:
    """Load an exact customer-execution declaration or fail closed."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("zero-custody attestation is unreadable") from exc
    if not isinstance(value, dict) or set(value) != REQUIRED_FIELDS:
        raise ValueError("zero-custody attestation fields are incomplete or unsupported")
    if (
        not isinstance(value["schema_version"], int)
        or isinstance(value["schema_version"], bool)
        or value["schema_version"] != 1
    ):
        raise ValueError("zero-custody attestation schema is unsupported")
    boolean_fields = REQUIRED_FIELDS - {"schema_version", "execution_mode"}
    if any(not isinstance(value[field], bool) for field in boolean_fields):
        raise ValueError("zero-custody attestation requires explicit Boolean values")

    safe = (
        value["execution_mode"] == "customer_run"
        and value["customer_authorized_computer"] is True
        and value["customer_audio_uploaded"] is False
        and value["customer_audio_transferred_to_deafbench"] is False
        and value["remote_shell_enabled"] is False
        and value["unattended_access_enabled"] is False
        and value["credentials_shared"] is False
        and value["aggregate_only_export"] is True
    )
    if not safe:
        raise ValueError("zero-custody execution conditions are not satisfied")
    return validate_execution_attestation(
        ExecutionAttestation(
            execution_mode="customer_run",
            aggregate_only_export=True,
            sha256=hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
    )
