import json
from pathlib import Path

import pytest

from deafbench.leaderboard.policy import (
    EvaluationPolicyError,
    verify_evaluation_policy,
)


_POLICY = (
    Path(__file__).parents[1] / "experiments" / "open-asr" / "evaluation-policy.json"
)


def _policy_copy() -> dict[str, object]:
    return json.loads(_POLICY.read_text(encoding="utf-8"))


def test_evaluation_policy_keeps_official_tests_evaluation_only():
    policy = verify_evaluation_policy(_POLICY)

    assert policy.official_test_role == "declared_milestone_evaluation_only"
    assert policy.repeated_label_tuning_allowed is False
    assert policy.training_development_evaluation_disjoint is True
    assert policy.contamination_status == "indeterminate"
    assert policy.final_claim_eligible is False


def test_evaluation_policy_fails_open_on_unknown_contamination(tmp_path):
    policy = _policy_copy()
    policy["contamination_audit"]["final_claim_eligible"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(EvaluationPolicyError, match="unknown contamination"):
        verify_evaluation_policy(path)


def test_evaluation_policy_rejects_test_label_tuning(tmp_path):
    policy = _policy_copy()
    policy["integrity_rules"]["repeated_test_label_tuning_allowed"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(EvaluationPolicyError, match="test-label tuning"):
        verify_evaluation_policy(path)


def test_evaluation_policy_rejects_textual_claim_eligibility(tmp_path):
    policy = _policy_copy()
    policy["contamination_audit"]["final_claim_eligible"] = "false"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(EvaluationPolicyError, match="must be Boolean"):
        verify_evaluation_policy(path)
