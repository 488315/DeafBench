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


@pytest.mark.parametrize("value", ["[]", '"caption"', "null"])
def test_parse_jsonl_rejects_non_record_values(tmp_path, value):
    path = tmp_path / "input.jsonl"
    path.write_text(value + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 1"):
        parse_jsonl(str(path))
