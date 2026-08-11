import json
from pathlib import Path

import pytest

from deafbench.pilot.export_scan import assert_export_safe, scan_export_directory


def _write_json(root: Path, value: object) -> None:
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")


def test_export_scanner_accepts_aggregate_only_artifacts(tmp_path: Path) -> None:
    export = tmp_path / "export"
    _write_json(
        export,
        {
            "dataset_count": 25,
            "model_id": "Qwen/Qwen3-ASR-1.7B-hf",
            "metrics": {"wer_percent": 21.0},
        },
    )

    assert scan_export_directory(export) == ()


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"transcript": "the private spoken sentence"}, "prohibited field"),
        ({"sample_id": "customer-001"}, "prohibited field"),
        ({"speaker_identity": "Alex"}, "prohibited field"),
        ({"critical_value": "alpha seven nine"}, "prohibited field"),
        ({"notes": "api" + "_key=do-not-export"}, "possible secret"),
        ({"notes": "C:\\Private\\caller.wav"}, "local path or filename"),
        ({"notes": "..\\private\\malicious.FLAC"}, "local path or filename"),
        ({"notes": "/home/customer/private.ogg"}, "local path or filename"),
        ({"notes": "case-" + "1" * 32}, "case or sample identifier"),
        ({"notes": "core-019"}, "case or sample identifier"),
    ],
)
def test_export_scanner_rejects_sensitive_and_sample_level_json(
    tmp_path: Path, payload: dict[str, object], reason: str
) -> None:
    export = tmp_path / "export"
    _write_json(export, payload)

    findings = scan_export_directory(export)

    assert any(reason in finding.reason for finding in findings)
    with pytest.raises(ValueError, match="export blocked"):
        assert_export_safe(export)


def test_export_scanner_rejects_transcript_leakage_in_report(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    (export / "report.md").write_text(
        "Transcript: the customer said a private sentence.", encoding="utf-8"
    )

    assert scan_export_directory(export)[0].reason == "prohibited report content"


def test_export_scanner_rejects_unexpected_or_unreadable_artifacts(
    tmp_path: Path,
) -> None:
    export = tmp_path / "export"
    export.mkdir()
    (export / "raw.txt").write_text("private", encoding="utf-8")
    (export / "manifest.json").write_bytes(b"\xff")

    reasons = {finding.reason for finding in scan_export_directory(export)}

    assert reasons == {"unexpected export artifact", "unreadable export artifact"}
