import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from deafbench.pilot.ledger import append_event, verify_ledger


def test_ledger_is_append_only_and_hash_chained(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    moment = datetime(2026, 8, 10, tzinfo=timezone.utc)
    first = append_event(path, case_id="case-test", event="case_creation", occurred_at=moment)
    append_event(
        path,
        case_id="case-test",
        event="model_execution",
        metadata={"model_id": "example/model"},
        occurred_at=moment,
    )

    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert entries[1]["previous_hash"] == first
    assert verify_ledger(path) is True


def test_ledger_detects_modified_history(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_event(path, case_id="case-test", event="access")
    path.write_text(path.read_text().replace('"event":"access"', '"event":"delivery"'))

    assert verify_ledger(path) is False
    with pytest.raises(RuntimeError, match="integrity"):
        append_event(path, case_id="case-test", event="deletion")


@pytest.mark.parametrize(
    "field", ["customer_name", "email", "transcript", "prediction", "content", "raw_text"]
)
def test_ledger_rejects_customer_content_fields(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValueError, match="prohibited"):
        append_event(
            tmp_path / "ledger.jsonl",
            case_id="case-test",
            event="access",
            metadata={field: "must not be logged"},
        )
