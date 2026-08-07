from deafbench.report import generate_markdown_report


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
