import json
from pathlib import Path

import pytest

from deafbench.pilot.authorization import load_authorization
from deafbench.pilot.intake import PROHIBITED_CATEGORIES, evaluate_intake


def _record(case_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": case_id,
        "authorization_reference": "agreement-sha256:abc123",
        "authorization_date": "2026-08-10",
        "ownership_confirmed": True,
        "scope": "Accessibility ASR audit of up to 100 test samples",
        "permitted_models": ["Qwen/Qwen3-ASR-1.7B-hf"],
        "planned_delivery_date": "2026-08-12",
        "planned_deletion_date": "2026-08-26",
        "sensitivity_classification": "synthetic",
        "deletion_agreement": True,
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_authorization_accepts_complete_record(tmp_path: Path) -> None:
    case_id = "case-" + "a" * 32
    path = tmp_path / "authorization.json"
    _write(path, _record(case_id))

    record = load_authorization(path, expected_case_id=case_id)

    assert record.case_id == case_id
    assert record.permitted_models == ("Qwen/Qwen3-ASR-1.7B-hf",)


@pytest.mark.parametrize(
    "missing",
    (
        "authorization_reference",
        "authorization_date",
        "ownership_confirmed",
        "scope",
        "permitted_models",
        "planned_delivery_date",
        "planned_deletion_date",
        "sensitivity_classification",
        "deletion_agreement",
    ),
)
def test_load_authorization_fails_closed_for_missing_fields(
    tmp_path: Path, missing: str
) -> None:
    case_id = "case-" + "b" * 32
    value = _record(case_id)
    del value[missing]
    path = tmp_path / "authorization.json"
    _write(path, value)

    with pytest.raises(ValueError, match="missing required fields"):
        load_authorization(path, expected_case_id=case_id)


def test_load_authorization_rejects_case_or_date_mismatch(tmp_path: Path) -> None:
    case_id = "case-" + "c" * 32
    value = _record(case_id)
    value["planned_deletion_date"] = "2026-08-11"
    path = tmp_path / "authorization.json"
    _write(path, value)

    with pytest.raises(ValueError, match="after delivery"):
        load_authorization(path, expected_case_id=case_id)

    value = _record(case_id)
    _write(path, value)
    with pytest.raises(ValueError, match="case ID"):
        load_authorization(path, expected_case_id="case-" + "d" * 32)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("ownership_confirmed", False, "ownership"),
        ("deletion_agreement", False, "deletion agreement"),
        ("planned_deletion_date", "2026-08-27", "exceeds the 14-day"),
        ("permitted_models", [], "nonempty unique"),
        (
            "permitted_models",
            ["Qwen/Qwen3-ASR-1.7B-hf", "Qwen/Qwen3-ASR-1.7B-hf"],
            "nonempty unique",
        ),
    ),
)
def test_load_authorization_rejects_unsafe_permission_envelopes(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    case_id = "case-" + "e" * 32
    record = _record(case_id)
    record[field] = value
    path = tmp_path / "authorization.json"
    _write(path, record)

    with pytest.raises(ValueError, match=message):
        load_authorization(path, expected_case_id=case_id)


def test_intake_rejects_unsafe_sensitivity_or_exclusion_declarations() -> None:
    safe_exclusions = {category: False for category in PROHIBITED_CATEGORIES}

    with pytest.raises(ValueError, match="classification"):
        evaluate_intake(
            sensitivity_classification="confidential",
            prohibited_categories=safe_exclusions,
        )

    decision = evaluate_intake(
        sensitivity_classification="synthetic",
        prohibited_categories={**safe_exclusions, "medical_records": True},
    )

    assert decision.accepted is False
    assert decision.reason_codes == ("medical_records",)
