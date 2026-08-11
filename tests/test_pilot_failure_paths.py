import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deafbench.pilot.certificate import issue_deletion_certificate
from deafbench.pilot.deletion import DeletionResult
from deafbench.pilot.intake import PROHIBITED_CATEGORIES, evaluate_intake, write_rejection
from deafbench.pilot.ledger import append_event
from deafbench.pilot.rehearsal import _load_result
from deafbench.pilot.retention import (
    extend_retention,
    request_earlier_deletion,
    schedule_after_delivery,
)
from deafbench.pilot.storage import probe_bitlocker, restrict_acl_to_current_account


def test_native_storage_probes_report_only_command_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = subprocess.CompletedProcess([], 0, "Protection Status: Protection On", "")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    monkeypatch.setenv("USERNAME", "test-operator")

    assert probe_bitlocker(tmp_path).verified is True
    completed.stdout = "Successfully processed 1 files"
    assert restrict_acl_to_current_account(tmp_path) is True


def test_native_acl_probe_fails_without_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("USERNAME", raising=False)
    assert restrict_acl_to_current_account(tmp_path) is False


def test_content_free_records_reject_invalid_success_claims(tmp_path: Path) -> None:
    accepted = evaluate_intake(
        sensitivity_classification="synthetic",
        prohibited_categories={key: False for key in PROHIBITED_CATEGORIES},
    )
    with pytest.raises(ValueError, match="rejected"):
        write_rejection(tmp_path / "rejection.json", accepted)
    with pytest.raises(ValueError, match="verified"):
        issue_deletion_certificate(
            tmp_path / "certificate.json",
            case_id="case-test",
            result=DeletionResult("logical", (), (), False),
            operator="operator",
            deleted_at=datetime.now(timezone.utc),
            retained_records=(),
        )


def test_retention_rejects_ambiguous_or_non_extensions() -> None:
    with pytest.raises(ValueError, match="timezone"):
        schedule_after_delivery(datetime(2026, 8, 10))
    schedule = schedule_after_delivery(datetime(2026, 8, 10, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="timezone"):
        request_earlier_deletion(schedule, datetime(2026, 8, 11))
    with pytest.raises(ValueError, match="later"):
        extend_retention(
            schedule,
            deletion_deadline=schedule.deletion_deadline - timedelta(days=1),
            written_approval_reference="approval",
        )


def test_ledger_and_model_results_fail_closed_on_unknown_contracts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        append_event(tmp_path / "ledger.jsonl", case_id="case-test", event="unknown")
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"license_classification": "research_only"}))
    with pytest.raises(ValueError, match="not a complete"):
        _load_result(path)
