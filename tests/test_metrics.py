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

@pytest.mark.parametrize(
    ("critical", "prediction"),
    [
        ("twenty three", "The batch contains 23 samples."),
        ("47 dollars", "The total is $47.83."),
        ("83 cents", "The total is $47.83."),
        ("$47.83", "The total is forty seven dollars and eighty three cents."),
        ("$47.00", "The total is 47 dollars."),
        ("47 dollars", "The total is $47.00."),
        ("125 dollars", "The invoice total is one hundred and twenty five dollars."),
        ("9 AM", "The meeting starts at 9am."),
        ("9 AM", "The meeting starts at 9:00 AM."),
        ("9 AM", "The meeting starts at 09:00 AM."),
        ("11:45 PM", "The migration starts at 11.45pm."),
        ("11:45 PM", "The migration starts at eleven forty five PM."),
        ("2024", "The release shipped in twenty twenty four."),
        ("five point eight", "Install version 5.8 before opening the project."),
        (
            "version one point two point three point four",
            "Install version 1.2.3.4 before opening the project.",
        ),
        (
            "release one point two point three point four",
            "Install release 1.2.3.4 before opening the project.",
        ),
        ("192 dot 168 dot 1 dot 25", "The server address is 192.168.1.25."),
        ("seven four nine two six eight one", "The delivery number is 7492681."),
        ("one hundred twenty five dollars", "The invoice total is $125.40."),
        ("forty cents", "The invoice total is $125.40."),
        ("deployed version two point four", "The developer deployed version 2.4."),
    ],
)
def test_critical_info_matches_semantic_numeric_equivalents(critical, prediction):
    res = evaluate_critical_info({"critical": [critical]}, {"text": prediction})

    assert res["matched"] == [critical]
    assert res["missed"] == []

@pytest.mark.parametrize(
    ("critical", "prediction"),
    [
        ("2:15 PM", "The appointment is at 12.15pm."),
        ("47 dollars", "The total is $147.83."),
        ("version two point four", "The deployed version is 24."),
        ("192 dot 168 dot 1 dot 25", "The server address is 192.168.1.250."),
        ("56 PM", "The migration starts at eleven forty five PM."),
        ("44", "The release shipped in twenty twenty four."),
        ("3", "The values are one and two."),
    ],
)
def test_critical_info_keeps_distinct_numeric_values_separate(critical, prediction):
    res = evaluate_critical_info({"critical": [critical]}, {"text": prediction})

    assert res["matched"] == []
    assert res["missed"] == [critical]

@pytest.mark.parametrize(
    ("critical", "prediction"),
    [
        (
            "dev_user twenty three",
            "My username is dev underscore user 23 and the reset code is 481926.",
        ),
        (
            "alpha seven nine",
            "The connection code is Alpha79.",
        ),
        (
            "dev_user_23",
            "My username is dev underscore user underscore 23.",
        ),
    ],
)
def test_critical_info_matches_spoken_identifier_equivalents(critical, prediction):
    res = evaluate_critical_info({"critical": [critical]}, {"text": prediction})

    assert res["matched"] == [critical]
    assert res["missed"] == []

@pytest.mark.parametrize(
    ("critical", "prediction"),
    [
        ("Office Guest", "The Wi-Fi network name is OfficeGuest."),
        ("dev_user twenty three", "My username is devuser23."),
    ],
)
def test_identifier_normalization_keeps_meaningful_separators_strict(critical, prediction):
    res = evaluate_critical_info({"critical": [critical]}, {"text": prediction})

    assert res["matched"] == []
    assert res["missed"] == [critical]

def test_non_speech_eval():
    ref = {"sounds": ["[alarm]", "[laughter]"]}
    pred = {"text": "Hear the [alarm] go off"}
    res = evaluate_non_speech_info(ref, pred)
    assert res["total"] == 2
    assert res["matched"] == ["[alarm]"]
    assert res["missed"] == ["[laughter]"]

def test_non_speech_does_not_match_partial_token():
    ref = {"sounds": ["[alarm]"]}
    pred = {"text": "Hear the [alarm]-tone go off"}
    res = evaluate_non_speech_info(ref, pred)
    assert res["matched"] == []
    assert res["missed"] == ["[alarm]"]

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


def test_evaluate_dataset_leaves_non_speech_unscored_without_reference_sounds():
    refs = [
        {"id": "s1", "text": "Hello world.", "critical": [], "sounds": []}
    ]
    preds = [{"id": "s1", "text": "Hello world."}]

    metrics = evaluate_dataset(align_records(refs, preds))

    assert metrics["total_sounds"] == 0
    assert metrics["matched_sounds"] == 0
    assert metrics["non_speech_recall"] is None


def test_evaluate_dataset_collects_non_speech_failures():
    refs = [
        {
            "id": "ns-001",
            "text": "Please wait here.",
            "critical": [],
            "sounds": ["[alarm]", "[door closes]"],
        }
    ]
    preds = [{"id": "ns-001", "text": "Please wait here. [alarm]"}]

    metrics = evaluate_dataset(align_records(refs, preds))

    assert metrics["non_speech_failures"] == [
        {
            "id": "ns-001",
            "expected": "[door closes]",
            "predicted_text": "Please wait here. [alarm]",
        }
    ]


def test_evaluate_dataset_uses_aligned_id_for_non_speech_failure():
    refs = [
        {
            "text": "Please wait here.",
            "critical": [],
            "sounds": ["[alarm]"],
        }
    ]
    preds = [{"id": "sample-1", "text": "Please wait here."}]

    metrics = evaluate_dataset(align_records(refs, preds))

    assert metrics["non_speech_failures"] == [
        {
            "id": "sample-1",
            "expected": "[alarm]",
            "predicted_text": "Please wait here.",
        }
    ]
