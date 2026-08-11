"""Fail-closed declaration for customer-run, zero-custody execution."""

from __future__ import annotations

import json
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


def load_execution_attestation(path: Path) -> ExecutionAttestation:
    """Load an exact customer-execution declaration or fail closed."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("zero-custody attestation is unreadable") from exc
    if not isinstance(value, dict) or set(value) != REQUIRED_FIELDS:
        raise ValueError("zero-custody attestation fields are incomplete or unsupported")
    if value["schema_version"] != 1:
        raise ValueError("zero-custody attestation schema is unsupported")
    boolean_fields = REQUIRED_FIELDS - {"schema_version", "execution_mode"}
    if any(type(value[field]) is not bool for field in boolean_fields):
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
    return ExecutionAttestation(
        execution_mode="customer_run",
        aggregate_only_export=True,
    )
