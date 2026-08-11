"""Synthetic-only end-to-end rehearsal of the manual pilot lifecycle."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from deafbench.pilot.authorization import load_authorization
from deafbench.pilot.certificate import issue_deletion_certificate
from deafbench.pilot.deletion import logical_delete
from deafbench.pilot.incident import IncidentStop
from deafbench.pilot.intake import PROHIBITED_CATEGORIES, evaluate_intake
from deafbench.pilot.ledger import append_event, verify_ledger
from deafbench.pilot.retention import schedule_after_delivery
from deafbench.pilot.storage import ProtectionState, protect_case_storage
from deafbench.pilot.workspace import create_case_workspace, discover_git_worktrees


MODEL_RESULTS = (
    "qwen3-asr-1.7b.json",
    "parakeet-tdt-0.6b-v2.json",
    "granite-speech-4.1-2b.json",
)


@dataclass(frozen=True)
class RehearsalResult:
    case_id: str
    model_count: int
    deletion_verified: bool
    certificate_sha256: str
    ledger_verified: bool


def _load_result(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    evaluations = value.get("evaluations", [])
    synthetic = [item for item in evaluations if item.get("lane") == "synthetic-v2"]
    if (
        value.get("license_classification") != "commercial_candidate"
        or len(synthetic) != 1
        or synthetic[0].get("scope") != "complete"
    ):
        raise ValueError(f"result is not a complete commercial synthetic-v2 run: {path.name}")
    return value


def _authorize_synthetic_case(
    workspace_root: Path, case_id: str, now: datetime
) -> None:
    authorization_path = workspace_root / "input" / "authorization.json"
    authorization = {
        "schema_version": 1,
        "case_id": case_id,
        "authorization_reference": "synthetic-rehearsal-only",
        "authorization_date": now.date().isoformat(),
        "ownership_confirmed": True,
        "scope": "DeafBench authorized frozen synthetic corpus",
        "permitted_models": list(MODEL_RESULTS),
        "planned_delivery_date": now.date().isoformat(),
        "planned_deletion_date": (now.date() + timedelta(days=14)).isoformat(),
        "sensitivity_classification": "synthetic",
        "deletion_agreement": True,
    }
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    load_authorization(authorization_path, expected_case_id=case_id)


def _validate_model_results(
    repo: Path, case_id: str, ledger: Path, incident: IncidentStop
) -> list[dict[str, object]]:
    results = []
    for name in MODEL_RESULTS:
        value = incident.run_gate(
            "integrity",
            lambda name=name: _load_result(repo / "experiments/model-results" / name),
        )
        results.append(value)
        append_event(
            ledger,
            case_id=case_id,
            event="model_execution",
            metadata={"model_id": str(value["model"]["model_id"])},
        )
    return results


def _write_demo_report(
    workspace_root: Path, results: list[dict[str, object]]
) -> None:
    report_dir = workspace_root / "output" / "reports"
    report_dir.mkdir(parents=True)
    report = {
        "demonstration": True,
        "corpus": "DeafBench synthetic-v2",
        "models": [value["model"] for value in results],
        "metrics": [value["evaluations"][0]["metrics"] for value in results],
    }
    (report_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_synthetic_rehearsal(
    *,
    repo_root: Path,
    case_base: Path,
    records_root: Path,
    operator: str,
    protection_probe: Callable[[Path], ProtectionState],
    acl_restrictor: Callable[[Path], bool],
    delivered_at: datetime | None = None,
) -> RehearsalResult:
    """Exercise intake through deletion using frozen synthetic evidence only."""

    repo = Path(repo_root).resolve(strict=True)
    workspace = create_case_workspace(
        case_base,
        worktrees=discover_git_worktrees(repo),
    )
    record_dir = Path(records_root).resolve() / workspace.case_id
    incident = IncidentStop(record_dir / "incident-state.json")
    ledger = record_dir / "events.jsonl"
    append_event(ledger, case_id=workspace.case_id, event="case_creation")

    incident.run_gate(
        "access",
        lambda: protect_case_storage(
            workspace.root,
            protection_probe=protection_probe,
            acl_restrictor=acl_restrictor,
        ),
    )
    now = (delivered_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    incident.run_gate(
        "authorization",
        lambda: _authorize_synthetic_case(workspace.root, workspace.case_id, now),
    )
    decision = evaluate_intake(
        sensitivity_classification="synthetic",
        prohibited_categories={key: False for key in PROHIBITED_CATEGORIES},
    )
    if not decision.accepted:
        raise RuntimeError("synthetic rehearsal intake was unexpectedly rejected")
    source_audio = next((repo / "benchmarks/synthetic-v2/audio-synthetic").glob("*.wav"))
    shutil.copy2(source_audio, workspace.root / "input" / "authorized-synthetic.wav")
    append_event(ledger, case_id=workspace.case_id, event="validation")

    results = _validate_model_results(repo, workspace.case_id, ledger, incident)
    _write_demo_report(workspace.root, results)
    append_event(ledger, case_id=workspace.case_id, event="report_generation")
    append_event(ledger, case_id=workspace.case_id, event="delivery")
    schedule = schedule_after_delivery(now)
    append_event(
        ledger,
        case_id=workspace.case_id,
        event="retention_change",
        metadata={"deletion_deadline": schedule.deletion_deadline.isoformat()},
    )
    deletion = incident.run_gate(
        "deletion",
        lambda: logical_delete(
            workspace.root,
            case_id=workspace.case_id,
            repository_roots=(repo,),
        ),
    )
    append_event(ledger, case_id=workspace.case_id, event="deletion")
    digest = issue_deletion_certificate(
        record_dir / "deletion-certificate.json",
        case_id=workspace.case_id,
        result=deletion,
        operator=operator,
        deleted_at=schedule.deletion_deadline,
        retained_records=(
            "aggregate metrics",
            "reproducibility metadata",
            "deletion evidence",
        ),
    )
    return RehearsalResult(
        case_id=workspace.case_id,
        model_count=len(results),
        deletion_verified=deletion.verified,
        certificate_sha256=digest,
        ledger_verified=verify_ledger(ledger),
    )
