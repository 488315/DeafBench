import json
from pathlib import Path

import pytest

from deafbench.pilot.intake import evaluate_intake, write_rejection


PROHIBITED = {
    "medical_records": False,
    "consumer_health_data": False,
    "minors_audio": False,
    "payment_information": False,
    "authentication_secrets": False,
    "legal_recordings": False,
    "other_regulated_or_high_risk": False,
}


def test_intake_accepts_only_declared_pilot_safe_material() -> None:
    decision = evaluate_intake(
        sensitivity_classification="synthetic",
        prohibited_categories=PROHIBITED,
    )

    assert decision.accepted is True
    assert decision.reason_codes == ()


@pytest.mark.parametrize("category", tuple(PROHIBITED))
def test_intake_rejects_each_prohibited_category(category: str) -> None:
    declared = dict(PROHIBITED)
    declared[category] = True

    decision = evaluate_intake(
        sensitivity_classification="non_sensitive",
        prohibited_categories=declared,
    )

    assert decision.accepted is False
    assert decision.reason_codes == (category,)


def test_intake_fails_closed_for_missing_or_unknown_declarations() -> None:
    incomplete = dict(PROHIBITED)
    del incomplete["medical_records"]
    with pytest.raises(ValueError, match="exactly declare"):
        evaluate_intake(
            sensitivity_classification="non_sensitive",
            prohibited_categories=incomplete,
        )
    with pytest.raises(ValueError, match="classification"):
        evaluate_intake(
            sensitivity_classification="confidential",
            prohibited_categories=PROHIBITED,
        )


def test_rejection_record_contains_reason_without_customer_content(
    tmp_path: Path,
) -> None:
    decision = evaluate_intake(
        sensitivity_classification="non_sensitive",
        prohibited_categories={**PROHIBITED, "authentication_secrets": True},
    )
    output = tmp_path / "rejection.json"

    write_rejection(output, decision)

    value = json.loads(output.read_text(encoding="utf-8"))
    assert value == {
        "accepted": False,
        "reason_codes": ["authentication_secrets"],
        "reason": "Founding pilot excludes declared authentication secrets.",
    }
