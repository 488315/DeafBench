import pytest

from deafbench.metrics import (
    evaluate_critical_info,
    evaluate_dataset,
    evaluate_speaker_attribution,
)
from deafbench.parser import align_records


pytestmark = pytest.mark.unit


def test_speaker_attribution_skips_missing_reference_speaker():
    assert evaluate_speaker_attribution({}, {"speaker": "Speaker 1"}) is None


def test_speaker_attribution_rejects_missing_prediction_speaker():
    assert evaluate_speaker_attribution({"speaker": "Speaker 1"}, {}) is False


def test_critical_info_matches_standalone_numeric_term():
    result = evaluate_critical_info(
        {"critical": ["25"]},
        {"text": "The dose is 25 milligrams."},
    )

    assert result["matched"] == ["25"]
    assert result["missed"] == []


def test_align_records_uses_position_when_both_inputs_are_idless():
    references = [
        {"text": "first reference"},
        {"text": "second reference"},
    ]
    predictions = [
        {"text": "first prediction"},
        {"text": "second prediction"},
    ]

    aligned = align_records(references, predictions)

    assert aligned[0]["prediction"]["text"] == "first prediction"
    assert aligned[1]["prediction"]["text"] == "second prediction"


def test_align_records_uses_empty_prediction_for_missing_id():
    references = [
        {"id": "a", "text": "alpha"},
        {"id": "b", "text": "beta"},
    ]
    predictions = [{"id": "b", "text": "beta"}]

    aligned = align_records(references, predictions)

    assert aligned[0]["prediction"] == {"id": "a", "text": ""}
    assert aligned[1]["prediction"]["id"] == "b"


def test_evaluate_dataset_uses_even_latency_median():
    aligned = [
        {
            "reference": {"id": "a", "text": "alpha"},
            "prediction": {"id": "a", "text": "alpha", "latency_ms": 100},
        },
        {
            "reference": {"id": "b", "text": "beta"},
            "prediction": {"id": "b", "text": "beta", "latency_ms": 300},
        },
    ]

    metrics = evaluate_dataset(aligned)

    assert metrics["wer"] == 0.0
    assert metrics["median_latency_ms"] == 200.0
    assert metrics["speaker_accuracy"] is None
