from datetime import datetime, timezone
from pathlib import Path

from deafbench.pilot.rehearsal import run_synthetic_rehearsal
from deafbench.pilot.storage import ProtectionState


def test_synthetic_rehearsal_completes_three_model_lifecycle(tmp_path: Path) -> None:
    repo = Path(__file__).parents[1]
    result = run_synthetic_rehearsal(
        repo_root=repo,
        case_base=tmp_path / "cases",
        records_root=tmp_path / "records",
        operator="test-operator",
        protection_probe=lambda _: ProtectionState(True, "test-only encrypted volume"),
        acl_restrictor=lambda _: True,
        delivered_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
    )

    assert result.model_count == 3
    assert result.deletion_verified is True
    assert result.ledger_verified is True
    assert len(result.certificate_sha256) == 64
    case_root = tmp_path / "cases" / result.case_id
    assert not (case_root / "input").exists()
    assert not (case_root / "output" / "reports").exists()
