import json
from pathlib import Path

import pytest

from deafbench.pilot.cli import main
from deafbench.pilot.audit import CustomerAuditResult
from deafbench.pilot.export import CustomerExportResult
from deafbench.pilot.rehearsal import RehearsalResult


def _attestation(path: Path) -> Path:
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
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_rehearsal_cli_uses_no_storage_or_access_controls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = {}

    def runner(**kwargs: object) -> RehearsalResult:
        captured.update(kwargs)
        return RehearsalResult(3, 25, True, True, "a" * 64)

    assert main(
        [
            "rehearse",
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "export"),
            "--signing-key",
            str(tmp_path / "private-key.pem"),
        ],
        rehearsal_runner=runner,
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["model_count"] == 3
    assert set(captured) == {"repo_root", "output_dir", "signing_key"}


def test_export_cli_requires_zero_custody_attestation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = {}

    def exporter(**kwargs: object) -> CustomerExportResult:
        captured.update(kwargs)
        return CustomerExportResult("b" * 64, 3, 25)

    args = [
        "export",
        "--repo-root",
        str(tmp_path),
        "--attestation",
        str(_attestation(tmp_path / "attestation.json")),
        "--result",
        str(tmp_path / "qwen.json"),
        "--result",
        str(tmp_path / "parakeet.json"),
        "--result",
        str(tmp_path / "granite.json"),
        "--output-dir",
        str(tmp_path / "export"),
        "--signing-key",
        str(tmp_path / "signing-key.pem"),
    ]

    assert main(args, exporter=exporter) == 0
    assert json.loads(capsys.readouterr().out)["sample_count"] == 25
    assert len(captured["result_paths"]) == 3
    assert len(captured["execution_attestation"].sha256) == 64


def test_export_cli_fails_before_export_when_attestation_is_unsafe(
    tmp_path: Path,
) -> None:
    attestation = json.loads(
        _attestation(tmp_path / "attestation.json").read_text(encoding="utf-8")
    )
    attestation["remote_shell_enabled"] = True
    (tmp_path / "attestation.json").write_text(
        json.dumps(attestation), encoding="utf-8"
    )
    called = False

    def exporter(**_: object) -> CustomerExportResult:
        nonlocal called
        called = True
        return CustomerExportResult("b" * 64, 3, 25)

    with pytest.raises(ValueError, match="zero-custody"):
        main(
            [
                "export",
                "--repo-root",
                str(tmp_path),
                "--attestation",
                str(tmp_path / "attestation.json"),
                "--result",
                str(tmp_path / "result.json"),
                "--output-dir",
                str(tmp_path / "export"),
                "--signing-key",
                str(tmp_path / "signing-key.pem"),
            ],
            exporter=exporter,
        )
    assert called is False


def test_audit_cli_evaluates_then_exports_local_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []
    result_paths = tuple(tmp_path / f"result-{index}.json" for index in range(3))

    def audit_runner(**kwargs: object) -> CustomerAuditResult:
        calls.append(("audit", kwargs))
        return CustomerAuditResult(result_paths, 4)

    def exporter(**kwargs: object) -> CustomerExportResult:
        calls.append(("export", kwargs))
        return CustomerExportResult("c" * 64, 3, 4)

    assert main(
        [
            "audit",
            "--repo-root",
            str(tmp_path / "repo"),
            "--case-root",
            str(tmp_path / "case"),
            "--attestation",
            str(_attestation(tmp_path / "attestation.json")),
            "--output-dir",
            str(tmp_path / "export"),
            "--signing-key",
            str(tmp_path / "signing-key.pem"),
        ],
        audit_runner=audit_runner,
        exporter=exporter,
    ) == 0

    assert [name for name, _ in calls] == ["audit", "export"]
    assert calls[1][1]["result_paths"] == list(result_paths)
    assert len(calls[1][1]["execution_attestation"].sha256) == 64
    assert json.loads(capsys.readouterr().out)["sample_count"] == 4
