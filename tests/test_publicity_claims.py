from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_KIT = ROOT / "docs" / "publicity" / "deafbench-launch-kit.md"
NON_SPEECH_REPORT = ROOT / "benchmarks" / "non-speech-v1" / "model-a-report.md"


def _metric(report: str, label: str) -> str:
    match = re.search(rf"\*\*{re.escape(label)}\*\* \| ([^|\n]+)", report)
    assert match is not None, f"missing frozen metric: {label}"
    return match.group(1).strip()


def test_launch_claim_matches_frozen_non_speech_evidence() -> None:
    launch = LAUNCH_KIT.read_text(encoding="utf-8")
    report = NON_SPEECH_REPORT.read_text(encoding="utf-8")

    assert _metric(report, "Word Error Rate (WER)") == "2.0%"
    assert _metric(report, "Non-Speech Information Recall") == "0.0% (0/19)"
    assert "2.0% speech WER" in launch
    assert "0 of 19 expected sound events" in launch
    assert "benchmarks/non-speech-v1/model-a-report.md" in launch


def test_launch_copy_preserves_public_evidence_boundaries() -> None:
    launch = " ".join(LAUNCH_KIT.read_text(encoding="utf-8").split())

    for disclosure in (
        "12-sample synthetic demonstration",
        "not a Hugging Face verified leaderboard result",
        "not a demographic fairness study",
        "deafbench==0.2.1",
        "metadata-only Hugging Face repository",
    ):
        assert disclosure in launch
