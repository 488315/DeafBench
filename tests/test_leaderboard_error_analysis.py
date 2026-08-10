import pytest

from deafbench.leaderboard.error_analysis import analyze_records


def _distance(reference, prediction, *, merge_compounds):
    assert merge_compounds is True
    assert isinstance(reference, tuple)
    assert isinstance(prediction, tuple)
    if reference == prediction:
        return {"ins": 0, "del": 0, "sub": 0}
    return {"ins": 1, "del": 0, "sub": 1}


def test_analysis_ranks_error_mass_and_preserves_diagnostic_text():
    records = [
        {
            "audio_filepath": "sample_0",
            "duration": 1.5,
            "text": "Doctor Ada speaks",
            "pred_text": "physician ada speaks",
        },
        {
            "audio_filepath": "sample_1",
            "duration": 2.0,
            "text": "Two rare names",
            "pred_text": "too names here",
        },
    ]

    result = analyze_records(
        records,
        normalize=lambda text: text.lower().replace("doctor", "dr"),
        edit_distance=_distance,
        limit=1,
    )

    assert result == {
        "analyzed_rows": 2,
        "reference_words": 6,
        "errors": {"del": 0, "ins": 2, "sub": 2},
        "top_errors": [
            {
                "row": 0,
                "audio_filepath": "sample_0",
                "duration": 1.5,
                "reference": "dr ada speaks",
                "prediction": "physician ada speaks",
                "reference_words": 3,
                "errors": {"del": 0, "ins": 1, "sub": 1},
                "wer": 66.67,
            }
        ],
    }


@pytest.mark.parametrize("limit", (0, -1))
def test_analysis_rejects_non_positive_limit(limit):
    with pytest.raises(ValueError, match="limit"):
        analyze_records([], normalize=str.lower, edit_distance=_distance, limit=limit)


def test_analysis_rejects_empty_official_reference():
    record = {
        "audio_filepath": "sample_0",
        "duration": 1.0,
        "text": "ignored",
        "pred_text": "words",
    }

    with pytest.raises(ValueError, match="empty normalized reference"):
        analyze_records(
            [record],
            normalize=lambda _: "",
            edit_distance=_distance,
        )
