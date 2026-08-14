import pytest

from deafbench.report import generate_markdown_report


def test_report_keeps_dual_critical_metrics_and_word_error_counts():
    metrics = {
        "samples": 1,
        "wer": 50.0,
        "cer": 25.0,
        "orthographic_wer": 50.0,
        "normalized_wer": 25.0,
        "orthographic_cer": 25.0,
        "normalized_cer": 12.5,
        "normalization_policy": "deafbench-asr-normalization-v1",
        "substitutions": 1,
        "insertions": 0,
        "deletions": 0,
        "word_errors_by_sample": [
            {
                "id": "sample-1",
                "wer": 50.0,
                "substitutions": 1,
                "insertions": 0,
                "deletions": 0,
            }
        ],
        "critical_recall": 100.0,
        "canonical_critical_recall": 100.0,
        "strict_critical_recall": 0.0,
        "matched_critical": 1,
        "canonical_matched_critical": 1,
        "strict_matched_critical": 0,
        "total_critical": 1,
        "critical_failures": [],
        "non_speech_recall": None,
        "total_sounds": 0,
        "matched_sounds": 0,
    }

    report = generate_markdown_report(metrics, "refs.jsonl", "preds.jsonl")

    assert "Strict lexical critical recall** | 0.0% (0/1)" in report
    assert "Canonical semantic critical recall** | 100.0% (1/1)" in report
    assert "WER substitutions** | 1" in report
    assert "Orthographic WER** | 50.0%" in report
    assert "Normalized WER** | 25.0%" in report
    assert "Orthographic CER** | 25.0%" in report
    assert "Normalized CER** | 12.5%" in report
    assert "`deafbench-asr-normalization-v1`" in report
    assert "| `sample-1` | 50.0% | 1 | 0 | 0 |" in report


def test_report_labels_local_rtfx_as_environment_dependent() -> None:
    metrics = {
        "samples": 1,
        "wer": 0.0,
        "critical_recall": 100.0,
        "matched_critical": 0,
        "total_critical": 0,
        "non_speech_recall": None,
        "matched_sounds": 0,
        "total_sounds": 0,
        "critical_failures": [],
    }

    report = generate_markdown_report(
        metrics,
        "refs.jsonl",
        "preds.jsonl",
        performance={"local_rtfx": 12.5},
    )

    assert "Local throughput (RTFx)** | 12.50x" in report
    assert "audio seconds divided by inference wall seconds" in report
    assert "environment-dependent" in report


@pytest.mark.parametrize("local_rtfx", [0.0, -1.0, float("nan"), True, "fast"])
def test_report_rejects_invalid_local_rtfx(local_rtfx: object) -> None:
    metrics = {
        "samples": 1,
        "wer": 0.0,
        "critical_recall": 100.0,
        "matched_critical": 0,
        "total_critical": 0,
        "non_speech_recall": None,
        "matched_sounds": 0,
        "total_sounds": 0,
        "critical_failures": [],
    }

    with pytest.raises(ValueError, match="local_rtfx"):
        generate_markdown_report(
            metrics,
            "refs.jsonl",
            "preds.jsonl",
            performance={"local_rtfx": local_rtfx},
        )


def test_report_accepts_canonical_metrics_without_legacy_aliases() -> None:
    metrics = {
        "samples": 1,
        "wer": 0.0,
        "canonical_critical_recall": 100.0,
        "canonical_matched_critical": 1,
        "total_critical": 1,
        "critical_failures": [],
        "non_speech_recall": None,
        "matched_sounds": 0,
        "total_sounds": 0,
    }

    report = generate_markdown_report(metrics, "refs.jsonl", "preds.jsonl")

    assert "Canonical semantic critical recall** | 100.0% (1/1)" in report


def test_report_escapes_failure_table_cells():
    metrics = {
        "samples": 1,
        "wer": 0.0,
        "critical_recall": 0.0,
        "matched_critical": 0,
        "total_critical": 1,
        "non_speech_recall": 100.0,
        "matched_sounds": 0,
        "total_sounds": 0,
        "critical_failures": [
            {
                "id": "sample-1",
                "expected": "dose\\path|25\nmg",
                "predicted_text": "dose\\path|125\nmg",
            }
        ],
    }

    report = generate_markdown_report(metrics, "refs.jsonl", "preds.jsonl")

    assert "**dose\\\\path\\|25<br>mg**" in report
    assert "*dose\\\\path\\|125<br>mg*" in report


def test_report_marks_non_speech_unscored_as_na():
    metrics = {
        "samples": 1,
        "wer": 0.0,
        "critical_recall": 100.0,
        "matched_critical": 0,
        "total_critical": 0,
        "non_speech_recall": None,
        "matched_sounds": 0,
        "total_sounds": 0,
        "critical_failures": [],
    }

    report = generate_markdown_report(metrics, "refs.jsonl", "preds.jsonl")

    assert "| **Non-Speech Information Recall** | N/A |" in report


def test_report_lists_non_speech_failures():
    metrics = {
        "samples": 1,
        "wer": 0.0,
        "critical_recall": 100.0,
        "matched_critical": 0,
        "total_critical": 0,
        "non_speech_recall": 50.0,
        "matched_sounds": 1,
        "total_sounds": 2,
        "critical_failures": [],
        "non_speech_failures": [
            {
                "id": "ns-001",
                "expected": "[door closes]",
                "predicted_text": "Please wait here. [alarm]",
            }
        ],
    }

    report = generate_markdown_report(metrics, "refs.jsonl", "preds.jsonl")

    assert "## Non-Speech Information Failures" in report
    assert "Detected **1** non-speech information failure:" in report
    assert "| `ns-001` | **[door closes]** | *Please wait here. [alarm]* |" in report


def test_report_escapes_non_speech_failure_sample_ids():
    metrics = {
        "samples": 1,
        "wer": 0.0,
        "critical_recall": 100.0,
        "matched_critical": 0,
        "total_critical": 0,
        "non_speech_recall": 0.0,
        "matched_sounds": 0,
        "total_sounds": 1,
        "critical_failures": [],
        "non_speech_failures": [
            {
                "id": "ns|001\npart",
                "expected": "[alarm]",
                "predicted_text": "Please wait here.",
            }
        ],
    }

    report = generate_markdown_report(metrics, "refs.jsonl", "preds.jsonl")

    assert "| `ns\\|001<br>part` | **[alarm]** | *Please wait here.* |" in report


def test_report_uses_singular_critical_failure_wording():
    metrics = {
        "samples": 1,
        "wer": 0.0,
        "critical_recall": 0.0,
        "matched_critical": 0,
        "total_critical": 1,
        "non_speech_recall": None,
        "matched_sounds": 0,
        "total_sounds": 0,
        "critical_failures": [
            {
                "id": "ns-006",
                "expected": "nearest safe exit",
                "predicted_text": "Leave the area using the nearest safe access.",
            }
        ],
    }

    report = generate_markdown_report(metrics, "refs.jsonl", "preds.jsonl")

    assert "Detected **1** critical information failure:" in report
    assert "Detected **1** critical information failures:" not in report


def test_report_trims_failure_output_for_emphasis():
    metrics = {
        "samples": 1,
        "wer": 0.0,
        "critical_recall": 0.0,
        "matched_critical": 0,
        "total_critical": 1,
        "non_speech_recall": 0.0,
        "matched_sounds": 0,
        "total_sounds": 1,
        "critical_failures": [
            {
                "id": "sample-1",
                "expected": "safe exit",
                "predicted_text": "  nearest safe access  ",
            }
        ],
        "non_speech_failures": [
            {
                "id": "sample-1",
                "expected": "[alarm]",
                "predicted_text": "  nearest safe access  ",
            }
        ],
    }

    report = generate_markdown_report(metrics, "refs.jsonl", "preds.jsonl")

    assert report.count("*nearest safe access*") == 2
    assert "*  nearest safe access  *" not in report
