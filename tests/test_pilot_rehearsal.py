import json
from pathlib import Path

from deafbench.pilot.export_scan import assert_export_safe
from deafbench.pilot.manifest import EXECUTION_NOTICE, verify_signed_manifest
from deafbench.pilot.rehearsal import run_synthetic_rehearsal


def test_synthetic_rehearsal_completes_zero_custody_export(tmp_path: Path) -> None:
    repo = Path(__file__).parents[1]
    output = tmp_path / "customer-export"

    result = run_synthetic_rehearsal(
        repo_root=repo,
        output_dir=output,
        signing_key=tmp_path / "private" / "signing-key.pem",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert result.model_count == 3
    assert result.sample_count == 25
    assert result.export_safe is True
    assert result.signature_verified is True
    assert len(result.manifest_sha256) == 64
    assert manifest["execution_notice"] == EXECUTION_NOTICE
    assert_export_safe(output)
    assert verify_signed_manifest(output / "manifest.json") is True
    assert {path.name for path in output.iterdir()} == {"manifest.json", "report.md"}
