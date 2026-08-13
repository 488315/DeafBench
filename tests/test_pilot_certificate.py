import json
from datetime import datetime, timezone
from pathlib import Path

from deafbench.pilot.certificate import issue_deletion_certificate
from deafbench.pilot.deletion import DeletionResult


def test_certificate_records_verified_content_free_deletion(tmp_path: Path) -> None:
    result = DeletionResult(
        method="verified logical deletion",
        categories=("input", "work"),
        paths_checked=("X:/cases/case-id/input", "X:/backups"),
        verified=True,
    )
    path = tmp_path / "certificate.json"

    digest = issue_deletion_certificate(
        path,
        case_id="case-test",
        result=result,
        operator="operator-account",
        deleted_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
        retained_records=("contract", "aggregate metrics", "deletion evidence"),
    )

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {
        "schema_version": 1,
        "case_id": "case-test",
        "artifact_categories": ["input", "work"],
        "paths_checked": ["X:/cases/case-id/input", "X:/backups"],
        "deleted_at": "2026-08-10T00:00:00+00:00",
        "method": "verified logical deletion",
        "operator": "operator-account",
        "verification_result": "passed",
        "retained_non_sensitive_records": [
            "aggregate metrics",
            "contract",
            "deletion evidence",
        ],
        "certificate_sha256": digest,
    }


def test_certificate_is_byte_stable(tmp_path: Path) -> None:
    result = DeletionResult("verified logical deletion", ("input",), ("X:/input",), True)
    values = []
    for name in ("one.json", "two.json"):
        path = tmp_path / name
        issue_deletion_certificate(
            path,
            case_id="case-test",
            result=result,
            operator="operator-account",
            deleted_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
            retained_records=("contract",),
        )
        values.append(path.read_bytes())

    assert values[0] == values[1]
