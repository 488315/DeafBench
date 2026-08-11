"""Machine-readable authorization contract for pilot cases."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


CASE_ID_PATTERN = re.compile(r"case-[0-9a-f]{32}\Z")
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "authorization_reference",
        "authorization_date",
        "ownership_confirmed",
        "scope",
        "permitted_models",
        "planned_delivery_date",
        "planned_deletion_date",
        "sensitivity_classification",
        "deletion_agreement",
    }
)


@dataclass(frozen=True)
class AuthorizationRecord:
    """Validated permission and retention envelope for one case."""

    case_id: str
    authorization_reference: str
    authorization_date: date
    scope: str
    permitted_models: tuple[str, ...]
    planned_delivery_date: date
    planned_deletion_date: date
    sensitivity_classification: str


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"authorization field {field} must be nonempty text")
    return value.strip()


def _required_date(value: object, field: str) -> date:
    text = _required_text(value, field)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"authorization field {field} must be an ISO date") from exc


def load_authorization(
    path: Path, *, expected_case_id: str
) -> AuthorizationRecord:
    """Load a complete authorization record or fail closed."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("authorization record is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError("authorization record must be an object")
    missing = REQUIRED_FIELDS - value.keys()
    if missing:
        raise ValueError(
            "authorization record missing required fields: "
            + ", ".join(sorted(missing))
        )
    if value["schema_version"] != 1:
        raise ValueError("unsupported authorization schema version")

    case_id = _required_text(value["case_id"], "case_id")
    if not CASE_ID_PATTERN.fullmatch(case_id) or case_id != expected_case_id:
        raise ValueError("authorization case ID does not match workspace")
    if value["ownership_confirmed"] is not True:
        raise ValueError("audio ownership must be confirmed")
    if value["deletion_agreement"] is not True:
        raise ValueError("deletion agreement must be accepted")

    models = value["permitted_models"]
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(item, str) and item.strip() for item in models)
        or len(set(models)) != len(models)
    ):
        raise ValueError("permitted_models must be a nonempty unique text list")

    delivery = _required_date(
        value["planned_delivery_date"], "planned_delivery_date"
    )
    deletion = _required_date(
        value["planned_deletion_date"], "planned_deletion_date"
    )
    if deletion <= delivery:
        raise ValueError("planned deletion must be after delivery")
    if (deletion - delivery).days > 14:
        raise ValueError("planned deletion exceeds the 14-day pilot default")

    return AuthorizationRecord(
        case_id=case_id,
        authorization_reference=_required_text(
            value["authorization_reference"], "authorization_reference"
        ),
        authorization_date=_required_date(
            value["authorization_date"], "authorization_date"
        ),
        scope=_required_text(value["scope"], "scope"),
        permitted_models=tuple(item.strip() for item in models),
        planned_delivery_date=delivery,
        planned_deletion_date=deletion,
        sensitivity_classification=_required_text(
            value["sensitivity_classification"], "sensitivity_classification"
        ),
    )
