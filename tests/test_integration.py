import json

import pytest

from deafbench.metrics import evaluate_dataset
from deafbench.parser import align_records, parse_jsonl
from deafbench.report import generate_markdown_report


pytestmark = pytest.mark.integration


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_jsonl_to_markdown_report_pipeline(tmp_path):
    references_path = tmp_path / "references.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"

    _write_jsonl(
        references_path,
        [
            {
                "id": "dose",
                "text": "Dose is 25 milligrams.",
                "critical": ["25"],
                "sounds": [],
                "speaker": "A",
            },
            {
                "id": "door",
                "text": "Door closes. [door closes]",
                "critical": ["Door closes"],
                "sounds": ["[door closes]"],
                "speaker": "B",
            },
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "dose",
                "text": "Dose is 25 milligrams.",
                "latency_ms": 200,
                "speaker": "A",
            },
            {
                "id": "door",
                "text": "Door closes. [door closes]",
                "latency_ms": 400,
                "speaker": "B",
            },
        ],
    )

    references = parse_jsonl(str(references_path))
    predictions = parse_jsonl(str(predictions_path))
    aligned = align_records(references, predictions)
    metrics = evaluate_dataset(aligned)
    report = generate_markdown_report(
        metrics,
        str(references_path),
        str(predictions_path),
    )

    assert metrics["samples"] == 2
    assert metrics["wer"] == 0.0
    assert metrics["critical_recall"] == 100.0
    assert metrics["non_speech_recall"] == 100.0
    assert metrics["speaker_accuracy"] == 100.0
    assert metrics["median_latency_ms"] == 300.0
    assert "# DeafBench Evaluation Report" in report
    assert "No critical information failures detected" in report


def test_missing_prediction_is_scored_as_missing_critical_info(tmp_path):
    references_path = tmp_path / "references.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"

    _write_jsonl(
        references_path,
        [
            {"id": "a", "text": "alpha", "critical": ["alpha"]},
            {"id": "b", "text": "beta", "critical": ["beta"]},
        ],
    )
    _write_jsonl(
        predictions_path,
        [{"id": "b", "text": "beta"}],
    )

    aligned = align_records(
        parse_jsonl(str(references_path)),
        parse_jsonl(str(predictions_path)),
    )
    metrics = evaluate_dataset(aligned)

    assert aligned[0]["prediction"]["text"] == ""
    assert metrics["critical_recall"] == 50.0
    assert metrics["critical_failures"][0]["id"] == "a"
