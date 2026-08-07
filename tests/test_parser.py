from deafbench.parser import align_records


def test_align_records_does_not_map_none_ids():
    references = [{"id": None, "text": "reference"}]
    predictions = [
        {"id": None, "text": "first"},
        {"id": None, "text": "second"},
    ]

    aligned = align_records(references, predictions)

    assert aligned[0]["prediction"]["text"] == "first"
