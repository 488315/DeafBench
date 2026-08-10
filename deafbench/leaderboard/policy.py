"""Fail-closed data-use policy for official real-speech evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class EvaluationPolicyError(RuntimeError):
    """Raised when a policy would compromise evaluation integrity."""


@dataclass(frozen=True)
class EvaluationPolicy:
    """Safety-relevant fields from the real-speech evaluation policy."""

    official_test_role: str
    repeated_label_tuning_allowed: bool
    training_development_evaluation_disjoint: bool
    contamination_status: str
    final_claim_eligible: bool


def _read_policy(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationPolicyError(f"cannot read evaluation policy: {source}") from exc
    if not isinstance(value, dict):
        raise EvaluationPolicyError("evaluation policy must contain an object")
    return value


def _required_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise EvaluationPolicyError(f"missing object: {key}")
    return value


def verify_evaluation_policy(path: Path | str) -> EvaluationPolicy:
    """Reject policy states that would leak official tests into development."""
    document = _read_policy(path)
    partitions = _required_object(document, "data_partitions")
    rules = _required_object(document, "integrity_rules")
    contamination = _required_object(document, "contamination_audit")

    official_test_role = partitions.get("official_public_test")
    repeated_tuning = rules.get("repeated_test_label_tuning_allowed")
    disjoint = rules.get("training_development_evaluation_disjoint")
    status = contamination.get("status")
    final_claim_eligible = contamination.get("final_claim_eligible")

    if official_test_role != "declared_milestone_evaluation_only":
        raise EvaluationPolicyError("official public test role is not protected")
    if repeated_tuning is not False:
        raise EvaluationPolicyError("test-label tuning must remain prohibited")
    if disjoint is not True:
        raise EvaluationPolicyError("data partitions must remain disjoint")
    if status not in {"clear", "suspected", "indeterminate"}:
        raise EvaluationPolicyError("invalid contamination audit status")
    if status != "clear" and final_claim_eligible is not False:
        raise EvaluationPolicyError("unknown contamination cannot permit a final claim")

    return EvaluationPolicy(
        official_test_role=official_test_role,
        repeated_label_tuning_allowed=repeated_tuning,
        training_development_evaluation_disjoint=disjoint,
        contamination_status=status,
        final_claim_eligible=final_claim_eligible,
    )
