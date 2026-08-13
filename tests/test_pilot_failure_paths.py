from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deafbench.pilot.certificate import issue_deletion_certificate
from deafbench.pilot.deletion import DeletionResult
from deafbench.pilot.intake import PROHIBITED_CATEGORIES, evaluate_intake, write_rejection
from deafbench.pilot.ledger import append_event
from deafbench.pilot.retention import (
    extend_retention,
    request_earlier_deletion,
    schedule_after_delivery,
)


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
    with pytest.raises(ValueError, match="later"):
        extend_retention(
            schedule,
            deletion_deadline=schedule.deletion_deadline,
            written_approval_reference="approval",
        )


def test_ledger_fails_closed_on_unknown_contracts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        append_event(
            tmp_path / "ledger.jsonl",
            case_id="case-" + "a" * 32,
            event="unknown",
        )
