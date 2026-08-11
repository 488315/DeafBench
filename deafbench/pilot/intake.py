"""Founding-pilot sensitivity screening without retaining supplied content."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ALLOWED_CLASSIFICATIONS = frozenset(
    {"non_sensitive", "synthetic", "public_domain", "explicitly_consented_test"}
)
PROHIBITED_CATEGORIES = (
    "medical_records",
    "consumer_health_data",
    "minors_audio",
    "payment_information",
    "authentication_secrets",
    "legal_recordings",
    "other_regulated_or_high_risk",
)
_REASON_NAMES = {
    "medical_records": "medical records",
    "consumer_health_data": "consumer health data",
    "minors_audio": "minors' audio",
    "payment_information": "payment information",
    "authentication_secrets": "authentication secrets",
    "legal_recordings": "legal recordings",
    "other_regulated_or_high_risk": "other regulated or high-risk material",
}


@dataclass(frozen=True)
class IntakeDecision:
    """A content-free decision produced from explicit intake declarations."""

    accepted: bool
    reason_codes: tuple[str, ...]


def evaluate_intake(
    *,
    sensitivity_classification: str,
    prohibited_categories: Mapping[str, bool],
) -> IntakeDecision:
    """Fail closed unless every exclusion has an explicit Boolean declaration."""

    if sensitivity_classification not in ALLOWED_CLASSIFICATIONS:
        raise ValueError("unsupported founding-pilot sensitivity classification")
    if set(prohibited_categories) != set(PROHIBITED_CATEGORIES) or any(
        type(value) is not bool for value in prohibited_categories.values()
    ):
        raise ValueError("intake must exactly declare every prohibited category")

    reasons = tuple(
        category
        for category in PROHIBITED_CATEGORIES
        if prohibited_categories[category]
    )
    return IntakeDecision(accepted=not reasons, reason_codes=reasons)


def write_rejection(path: Path, decision: IntakeDecision) -> None:
    """Persist a content-free rejection reason for an excluded intake."""

    if decision.accepted or not decision.reason_codes:
        raise ValueError("only rejected intake decisions may be recorded")
    phrases = [_REASON_NAMES[code] for code in decision.reason_codes]
    reason = "Founding pilot excludes declared " + ", ".join(phrases) + "."
    payload = {
        "accepted": False,
        "reason_codes": list(decision.reason_codes),
        "reason": reason,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
