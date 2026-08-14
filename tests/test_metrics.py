import pytest
from deafbench.parser import normalize_text, align_records
from deafbench.metrics import (
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


@pytest.mark.parametrize(
    (
        "sample_id",
        "entity_type",
        "expected",
        "prediction",
        "semantic_match",
    ),
    [
        pytest.param(
            "core-001",
            "TIME",
            "2:15 PM",
            "My appointment is Friday at 2 clear 15 p.m.",
            False,
            id="recognition-error-invalid-time-word",
        ),
        pytest.param(
            "core-006",
            "TIME",
            "8:30 PM",
            "Take 15 milligrams at 8 30 p.m., not 50 milligrams.",
            True,
            id="formatting-only-spoken-time",
        ),
        pytest.param(
            "core-009",
            "TIME",
            "4:45 PM",
            "Jordan needs 83927 Thursday at 4 core 5 p.m.",
            False,
            id="recognition-error-corrupted-time",
        ),
        pytest.param(
            "core-011",
            "USERNAME",
            "dev_user twenty three",
            "My username is devcusser23 and the code is 481926.",
            False,
            id="recognition-error-username-characters",
        ),
        pytest.param(
            "core-012",
            "TIME",
            "11:45 PM",
            "The migration starts at 11 o'clock 45pm.",
            True,
            id="semantic-equivalent-spoken-time",
        ),
        pytest.param(
            "core-016",
            "DIGIT_SEQUENCE",
            "seven four nine two six eight one",
            "The package delivery number is seven.",
            False,
            id="recognition-error-incomplete-digit-sequence",
        ),
        pytest.param(
            "core-019",
            "SSID",
            "Office Guest",
            "The Wi-Fi network name is Alpha's Guest.",
            False,
            id="recognition-error-different-wifi-name",
        ),
    ],
)
def test_reported_failures_use_strict_and_typed_semantic_scoring(
    sample_id,
    entity_type,
    expected,
    prediction,
    semantic_match,
):
    """Lock the seven baseline failures to their documented pass/fail reason."""
    result = evaluate_critical_info(
        {
            "id": sample_id,
            "critical": [expected],
            "critical_types": {expected: entity_type},
        },
        {"id": sample_id, "text": prediction},
    )

    assert result["strict_matched"] == []
    assert (result["canonical_matched"] == [expected]) is semantic_match
    assert (result["canonical_missed"] == [expected]) is not semantic_match


@pytest.mark.parametrize(
    ("entity_type", "expected", "prediction", "matches"),
    [
        ("TIME", "8:30 PM", "at eight thirty p.m.", True),
        ("TIME", "8:30 PM", "at 8 30 p.m.", True),
        ("TIME", "9 AM", "at nine a.m.", True),
        ("DIGIT_SEQUENCE", "seven four nine", "number 749", True),
        ("DIGIT_SEQUENCE", "seven four nine", "number 74", False),
        ("USERNAME", "dev_user twenty three", "dev underscore user 23", True),
        ("USERNAME", "dev_user twenty three", "devuser23", False),
        ("CODE", "alpha seven nine", "code Alpha79", True),
        ("CODE", "alpha seven nine", "code Alpha-79", False),
        ("PASSWORD", "481926", "password 481926", True),
        ("PASSWORD", "481926", "password is four eight one nine two six", True),
        ("PASSWORD", "481926", "password 48192", False),
        ("SSID", "Office Guest", "network officeguest", True),
        ("SSID", "Office Guest", "network Alpha Guest", False),
        ("PROPER_NAME", "Ada Lovelace", "speaker ada   lovelace", True),
        ("PROPER_NAME", "Ada Lovelace", "speaker Ava Lovelace", False),
    ],
)
def test_typed_entities_apply_only_their_allowed_normalization(
    entity_type, expected, prediction, matches
):
    result = evaluate_critical_info(
        {
            "critical": [expected],
            "critical_types": {expected: entity_type},
        },
        {"text": prediction},
    )

    assert (result["canonical_matched"] == [expected]) is matches


@pytest.mark.parametrize(
    "critical_types",
    [
        [],
        {"other": "TIME"},
        {"hello": "FUZZY"},
    ],
)
def test_critical_entity_contract_rejects_malformed_types(critical_types):
    with pytest.raises(ValueError, match="critical_types"):
        evaluate_critical_info(
            {"critical": ["hello"], "critical_types": critical_types},
            {"text": "hello"},
        )


def test_evaluate_dataset_reports_per_sample_and_aggregate_word_errors():
    refs = [
        {"id": "s1", "text": "alpha beta", "critical": [], "sounds": []},
        {"id": "s2", "text": "one two", "critical": [], "sounds": []},
    ]
    preds = [
        {"id": "s1", "text": "alpha gamma extra"},
        {"id": "s2", "text": "one"},
    ]

    metrics = evaluate_dataset(align_records(refs, preds))

    assert metrics["substitutions"] == 1
    assert metrics["insertions"] == 1
    assert metrics["deletions"] == 1
    assert metrics["wer"] == metrics["orthographic_wer"]
    assert metrics["cer"] == metrics["orthographic_cer"]
    assert metrics["normalization_policy"] == "deafbench-asr-normalization-v1"
    assert metrics["normalized_wer"] == metrics["orthographic_wer"]
    assert metrics["normalized_substitutions"] == 1
    assert metrics["normalized_insertions"] == 1
    assert metrics["normalized_deletions"] == 1
    assert metrics["word_errors_by_sample"] == [
        {
            "id": "s1",
            "wer": 100.0,
            "substitutions": 1,
            "insertions": 1,
            "deletions": 0,
        },
        {
            "id": "s2",
            "wer": 50.0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 1,
        },
    ]


def test_evaluate_dataset_exposes_normalized_accuracy_without_replacing_wer():
    refs = [{"id": "s1", "text": "Hello, WORLD", "critical": [], "sounds": []}]
    preds = [{"id": "s1", "text": "hello world"}]

    metrics = evaluate_dataset(align_records(refs, preds))

    assert metrics["wer"] == 100.0
    assert metrics["orthographic_wer"] == 100.0
    assert metrics["normalized_wer"] == 0.0
    assert metrics["cer"] == metrics["orthographic_cer"]
    assert metrics["normalized_cer"] == 0.0


def test_evaluate_dataset_leaves_non_speech_unscored_without_reference_sounds():
    refs = [
        {"id": "s1", "text": "Hello world.", "critical": [], "sounds": []}
    ]
    preds = [{"id": "s1", "text": "Hello world."}]

    metrics = evaluate_dataset(align_records(refs, preds))

    assert metrics["total_sounds"] == 0
    assert metrics["matched_sounds"] == 0
    assert metrics["non_speech_recall"] is None


def test_evaluate_dataset_preserves_sound_only_accessibility_scoring():
    refs = [{"id": "sound-1", "text": "", "critical": [], "sounds": ["[alarm]"]}]
    preds = [{"id": "sound-1", "text": "", "sounds": ["[alarm]"]}]

    metrics = evaluate_dataset(align_records(refs, preds))

    assert metrics["non_speech_recall"] == 100.0
    assert metrics["matched_sounds"] == 1
    assert metrics["orthographic_wer"] is None
    assert metrics["normalized_wer"] is None
    assert metrics["word_errors_by_sample"] == []


def test_evaluate_dataset_rejects_empty_normalized_reference():
    refs = [{"id": "bad-1", "text": "...", "critical": [], "sounds": []}]
    preds = [{"id": "bad-1", "text": "hallucination"}]

    with pytest.raises(ValueError, match="empty after ASR normalization"):
        evaluate_dataset(align_records(refs, preds))


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        (
            {"id": "bad-1", "text": "hello", "critical": [], "sounds": "alarm"},
            "reference sounds",
        ),
        (
            {"id": "bad-1", "text": "hello", "critical": [], "sounds": [""]},
            "reference sounds",
        ),
        (
            {"id": "bad-1", "text": "", "critical": [], "sounds": [""]},
            "reference sounds",
        ),
        (
            {"id": "bad-1", "text": 1, "critical": [], "sounds": []},
            "reference text",
        ),
    ],
)
def test_evaluate_dataset_rejects_malformed_reference_labels(reference, message):
    refs = [reference]
    preds = [{"id": "bad-1", "text": "hello"}]

    with pytest.raises(ValueError, match=message):
        evaluate_dataset(align_records(refs, preds))


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


def test_non_speech_eval_accepts_structured_prediction_sounds():
    ref = {"sounds": ["[alarm]", "[door closes]"]}
    pred = {
        "text": "Please remain seated.",
        "sounds": ["[alarm]"],
    }

    res = evaluate_non_speech_info(ref, pred)

    assert res["total"] == 2
    assert res["matched"] == ["[alarm]"]
    assert res["missed"] == ["[door closes]"]
