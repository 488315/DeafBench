import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from deafbench.pilot.ledger import _locked_ledger, append_event, verify_ledger


def test_ledger_is_append_only_and_hash_chained(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    moment = datetime(2026, 8, 10, tzinfo=timezone.utc)
    case_id = "case-" + "a" * 32
    first = append_event(path, case_id=case_id, event="case_creation", occurred_at=moment)
    append_event(
        path,
        case_id=case_id,
        event="model_execution",
        metadata={"model_id": "example/model"},
        occurred_at=moment,
    )

    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert entries[1]["previous_hash"] == first
    assert verify_ledger(path) is True


def test_ledger_detects_modified_history(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    case_id = "case-" + "a" * 32
    append_event(path, case_id=case_id, event="access")
    path.write_text(path.read_text().replace('"event":"access"', '"event":"delivery"'))

    assert verify_ledger(path) is False
    with pytest.raises(RuntimeError, match="integrity"):
        append_event(path, case_id=case_id, event="deletion")


@pytest.mark.parametrize(
    "field", ["customer_name", "email", "transcript", "prediction", "content", "raw_text"]
)
def test_ledger_rejects_customer_content_fields(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValueError, match="unsupported metadata"):
        append_event(
            tmp_path / "ledger.jsonl",
            case_id="case-" + "a" * 32,
            event="access",
            metadata={field: "must not be logged"},
        )


def test_ledger_rejects_nonopaque_case_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="case identifier"):
        append_event(tmp_path / "ledger.jsonl", case_id="case-customer", event="access")


def test_ledger_rejects_unknown_metadata_for_event(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported metadata"):
        append_event(
            tmp_path / "ledger.jsonl",
            case_id="case-" + "a" * 32,
            event="access",
            metadata={"note": "customer content could hide here"},
        )


def test_ledger_rejects_invalid_allowed_metadata_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metadata value"):
        append_event(
            tmp_path / "ledger.jsonl",
            case_id="case-" + "a" * 32,
            event="model_execution",
            metadata={"model_id": "customer name"},
        )


def test_ledger_lock_blocks_another_writer_until_history_is_complete(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    command = (
        "from pathlib import Path; "
        "from deafbench.pilot.ledger import append_event; "
        "append_event(Path(__import__('sys').argv[1]), "
        "case_id='case-' + 'a' * 32, event='access')"
    )

    with _locked_ledger(path):
        worker = subprocess.Popen([sys.executable, "-c", command, str(path)])
        with pytest.raises(subprocess.TimeoutExpired):
            worker.wait(timeout=0.1)
    assert worker.wait(timeout=3) == 0

    assert verify_ledger(path) is True
