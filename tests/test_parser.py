import pytest

from deafbench.parser import align_records, parse_jsonl


def test_align_records_does_not_map_none_ids():
    references = [{"id": None, "text": "reference"}]
    predictions = [
        {"id": None, "text": "first"},
        {"id": None, "text": "second"},
    ]

    aligned = align_records(references, predictions)

    assert aligned[0]["prediction"]["text"] == "first"


def test_align_records_does_not_reuse_id_prediction_positionally():
    references = [
        {"id": "a", "text": "reference a"},
        {"id": "b", "text": "reference b"},
    ]
    predictions = [
        {"id": "b", "text": "prediction b"},
    ]

    aligned = align_records(references, predictions)

    assert aligned[0]["prediction"] == {"id": "a", "text": ""}
    assert aligned[1]["prediction"] == predictions[0]


def test_align_records_rejects_duplicate_reference_ids():
    references = [
        {"id": "a", "text": "first"},
        {"id": "a", "text": "second"},
    ]
    predictions = [{"id": "a", "text": "prediction"}]

    with pytest.raises(ValueError, match=r"duplicate reference id: a"):
        align_records(references, predictions)


def test_align_records_rejects_duplicate_prediction_ids():
    references = [{"id": "a", "text": "reference"}]
    predictions = [
        {"id": "a", "text": "first"},
        {"id": "a", "text": "second"},
    ]

    with pytest.raises(ValueError, match=r"duplicate prediction id: a"):
        align_records(references, predictions)


@pytest.mark.parametrize("value", ["[]", '"caption"', "null"])
def test_parse_jsonl_rejects_non_record_values(tmp_path, value):
    path = tmp_path / "input.jsonl"
    path.write_text(value + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 1"):
        parse_jsonl(str(path))


@pytest.mark.parametrize(
    "record",
    [
        '{"id":"s1","text":null}',
        '{"id":"s1","text":"hello","critical":"dose"}',
        '{"id":"s1","text":"hello","sounds":"alarm"}',
    ],
)
def test_parse_jsonl_rejects_invalid_field_types(tmp_path, record):
    path = tmp_path / "input.jsonl"
    path.write_text(record + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 1"):
        parse_jsonl(str(path))
