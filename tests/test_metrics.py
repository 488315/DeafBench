import pytest
from deafbench.parser import normalize_text, align_records
from deafbench.metrics import (
    calculate_wer,
    evaluate_critical_info,
    evaluate_non_speech_info,
    evaluate_speaker_attribution,
    evaluate_dataset
)

def test_normalize_text():
    assert normalize_text("John Doe needs 25 milligrams!") == "john doe needs 25 milligrams"
    assert normalize_text("  [Door closes]  ") == "[door closes]"

def test_critical_info_eval():
    ref = {"critical": ["John Doe", "25 milligrams", "Friday"]}
    pred = {"text": "Guy needs 20 milligrams on Friday."}
    res = evaluate_critical_info(ref, pred)
    assert res["total"] == 3
    assert res["matched"] == ["Friday"]
    assert len(res["missed"]) == 2

def test_critical_info_does_not_match_numeric_substring():
    ref = {"critical": ["25"]}
    pred = {"text": "The dose is 125 milligrams."}
    res = evaluate_critical_info(ref, pred)
    assert res["matched"] == []
    assert res["missed"] == ["25"]

def test_non_speech_eval():
    ref = {"sounds": ["[alarm]", "[laughter]"]}
    pred = {"text": "Hear the [alarm] go off"}
    res = evaluate_non_speech_info(ref, pred)
    assert res["total"] == 2
    assert res["matched"] == ["[alarm]"]
    assert res["missed"] == ["[laughter]"]

def test_speaker_attribution():
    ref = {"speaker": "Speaker 1"}
    pred1 = {"speaker": "speaker 1"}
    pred2 = {"speaker": "Speaker 2"}
    assert evaluate_speaker_attribution(ref, pred1) is True
    assert evaluate_speaker_attribution(ref, pred2) is False

def test_evaluate_dataset():
    refs = [
        {"id": "s1", "text": "Hello world.", "critical": ["Hello"], "sounds": [], "speaker": "A"}
    ]
    preds = [
        {"id": "s1", "text": "Hello world.", "latency_ms": 500, "speaker": "A"}
    ]
    aligned = align_records(refs, preds)
    metrics = evaluate_dataset(aligned)
    assert metrics["samples"] == 1
    assert metrics["wer"] == 0.0
    assert metrics["critical_recall"] == 100.0
    assert metrics["speaker_accuracy"] == 100.0
    assert metrics["median_latency_ms"] == 500.0
