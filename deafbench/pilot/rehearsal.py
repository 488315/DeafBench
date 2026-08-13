"""Synthetic-only rehearsal of the customer-run zero-custody workflow."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from deafbench.pilot.export import CustomerExportResult, create_customer_export
from deafbench.pilot.zero_custody import load_execution_attestation


MODEL_RESULTS = (
    "qwen3-asr-1.7b.json",
    "parakeet-tdt-0.6b-v2.json",
    "granite-speech-4.1-2b.json",
)


@dataclass(frozen=True)
class RehearsalResult:
    model_count: int
    sample_count: int
    export_safe: bool
    signature_verified: bool
    manifest_sha256: str


def _synthetic_attestation(path: Path) -> None:
    value = {
        "schema_version": 1,
        "execution_mode": "customer_run",
        "customer_authorized_computer": True,
        "customer_audio_uploaded": False,
        "customer_audio_transferred_to_deafbench": False,
        "remote_shell_enabled": False,
        "unattended_access_enabled": False,
        "credentials_shared": False,
        "aggregate_only_export": True,
    }
    path.write_text(json.dumps(value), encoding="utf-8", newline="\n")


def run_synthetic_rehearsal(
    *, repo_root: Path, output_dir: Path, signing_key: Path
) -> RehearsalResult:
    """Exercise customer attestation through verified aggregate export."""

    repo = Path(repo_root).resolve(strict=True)
    result_root = repo / "experiments" / "model-results"
    with tempfile.TemporaryDirectory(prefix="deafbench-zero-custody-") as temporary:
        attestation = Path(temporary) / "execution-attestation.json"
        _synthetic_attestation(attestation)
        execution_attestation = load_execution_attestation(attestation)
        exported: CustomerExportResult = create_customer_export(
            repo_root=repo,
            result_paths=[result_root / name for name in MODEL_RESULTS],
            output_dir=Path(output_dir),
            signing_key=Path(signing_key),
            execution_attestation=execution_attestation,
        )
    return RehearsalResult(
        model_count=exported.model_count,
        sample_count=exported.sample_count,
        export_safe=True,
        signature_verified=True,
        manifest_sha256=exported.manifest_sha256,
    )
