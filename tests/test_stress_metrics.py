import pytest

from deafbench.benchmark.stress_metrics import (
    evaluate_caption_timing,
    summarize_stress_results,
)


def _reference() -> dict[str, object]:
    return {
        "id": "stress-001",
        "text": "Meet Priya Shah at 8:30 PM",
        "critical": ["Priya Shah", "8:30 PM"],
        "critical_types": {
            "Priya Shah": "PROPER_NAME",
            "8:30 PM": "TIME",
        },
        "risk_categories": {
            "Priya Shah": "PROPER_NAME",
            "8:30 PM": "TIME",
        },
        "sounds": [],
    }


def test_stress_summary_keeps_clean_and_stressed_metrics_separate() -> None:
    summary = summarize_stress_results(
        [_reference()],
        [{"id": "stress-001", "text": "Meet Priya Shah at 8:30 PM"}],
        [{"id": "stress-001", "text": "Meet Priya at PM"}],
    )

    assert summary["clean"]["wer"] == 0.0
    assert summary["stressed"]["deletions"] == 2
    assert summary["stressed"]["deletion_share_percent"] == 100.0
    assert summary["stressed"]["critical_failures_by_risk"] == {
        "PROPER_NAME": 1,
        "TIME": 1,
    }
    assert summary["degradation"]["wer_points"] > 0.0
    assert summary["degradation"]["canonical_recall_points"] == -100.0


def test_stress_summary_counts_substitutions_and_insertions() -> None:
    reference = _reference()
    reference["text"] = "alpha beta gamma"
    reference["critical"] = ["alpha"]
    reference["critical_types"] = {}
    reference["risk_categories"] = {"alpha": "CODE"}

    summary = summarize_stress_results(
        [reference],
        [{"id": "stress-001", "text": "alpha beta gamma"}],
        [{"id": "stress-001", "text": "alpha delta gamma extra"}],
    )

    assert summary["stressed"]["substitutions"] == 1
    assert summary["stressed"]["insertions"] == 1
    assert summary["stressed"]["deletion_share_percent"] == 0.0


def test_caption_timing_reports_drift_past_accessibility_threshold() -> None:
    summary = evaluate_caption_timing(
        [
            {"token": "meet", "reference_ms": 100.0, "prediction_ms": 180.0},
            {"token": "priya", "reference_ms": 500.0, "prediction_ms": 1_050.0},
            {"token": "now", "reference_ms": 900.0, "prediction_ms": 300.0},
        ]
    )

    assert summary == {
        "tokens": 3,
        "median_absolute_drift_ms": 550.0,
        "maximum_absolute_drift_ms": 600.0,
        "over_500ms": 2,
        "over_500ms_percent": pytest.approx(66.6666666667),
    }


@pytest.mark.parametrize(
    "alignment",
    [
        [],
        [{"token": "", "reference_ms": 0.0, "prediction_ms": 0.0}],
        [{"token": "word", "reference_ms": -1.0, "prediction_ms": 0.0}],
        [{"token": "word", "reference_ms": 0.0, "prediction_ms": float("nan")}],
    ],
)
def test_caption_timing_rejects_unusable_alignments(alignment) -> None:
    with pytest.raises(ValueError, match="timing"):
        evaluate_caption_timing(alignment)


def test_stress_summary_rejects_missing_risk_mapping() -> None:
    reference = _reference()
    reference["risk_categories"] = {"Priya Shah": "PROPER_NAME"}

    with pytest.raises(ValueError, match="no declared risk category"):
        summarize_stress_results(
            [reference],
            [{"id": "stress-001", "text": "Meet Priya Shah at 8:30 PM"}],
            [{"id": "stress-001", "text": "Meet Priya Shah"}],
        )
