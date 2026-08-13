import json
from pathlib import Path

import pytest

from deafbench.pilot import incident
from deafbench.pilot.incident import IncidentStop, ProcessingBlocked


@pytest.mark.parametrize(
    "category", ["authorization", "integrity", "access", "retention", "deletion"]
)
def test_every_control_failure_stops_processing(tmp_path: Path, category: str) -> None:
    stop = IncidentStop(tmp_path / "incident.json")

    with pytest.raises(ProcessingBlocked, match=category):
        stop.run_gate(category, lambda: (_ for _ in ()).throw(RuntimeError("failure")))
    with pytest.raises(ProcessingBlocked, match="pending approval"):
        stop.run_gate("access", lambda: "must not run")

    state = json.loads(stop.state_path.read_text(encoding="utf-8"))
    assert state["blocked"] is True
    assert state["incidents"][0]["reason_type"] == "RuntimeError"
    assert "failure" not in stop.state_path.read_text(encoding="utf-8")


def test_processing_resumes_only_after_explicit_approval(tmp_path: Path) -> None:
    stop = IncidentStop(tmp_path / "incident.json")
    with pytest.raises(ProcessingBlocked):
        stop.run_gate("integrity", lambda: 1 / 0)
    with pytest.raises(ValueError, match="explicit approval"):
        stop.restore(approval_reference="", operator="operator")

    stop.restore(approval_reference="approval-42", operator="operator")

    assert stop.run_gate("access", lambda: "resumed") == "resumed"


def test_incident_state_preserves_prior_record_when_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = IncidentStop(tmp_path / "incident.json")
    with pytest.raises(ProcessingBlocked):
        stop.run_gate("integrity", lambda: 1 / 0)
    original = stop.state_path.read_bytes()

    def fail_replace(_source, _destination) -> None:
        raise OSError("simulated promotion failure")

    monkeypatch.setattr(incident.os, "replace", fail_replace)
    with pytest.raises(OSError, match="promotion failure"):
        stop.restore(approval_reference="approval-42", operator="operator")

    assert stop.state_path.read_bytes() == original
    assert not tuple(tmp_path.glob("*.tmp"))
