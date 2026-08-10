from pathlib import Path

import pytest

from deafbench.benchmark.forced_alignment import (
    coverage_from_word_scores,
)


def test_alignment_coverage_counts_characters_at_predeclared_score_floor():
    words = ("meet", "eight", "thirty")
    scores = (
        (0.9, 0.9, 0.9, 0.9),
        (0.9, 0.9, 0.1, 0.9, 0.9),
        (0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
    )

    total, entities = coverage_from_word_scores(
        words,
        scores,
        {"8:30": (1, 3)},
        score_threshold=0.25,
    )

    assert total == pytest.approx(14 / 15)
    assert entities == {"8:30": pytest.approx(10 / 11)}


def test_alignment_coverage_rejects_shape_or_range_drift():
    with pytest.raises(ValueError, match="word count"):
        coverage_from_word_scores(("one",), (), {}, score_threshold=0.25)
    with pytest.raises(ValueError, match="character count"):
        coverage_from_word_scores(
            ("one",), ((0.9,),), {}, score_threshold=0.25
        )
    with pytest.raises(ValueError, match="entity word range"):
        coverage_from_word_scores(
            ("one",), ((0.9, 0.9, 0.9),), {"bad": (0, 2)}, score_threshold=0.25
        )


def test_alignment_coverage_rejects_invalid_score_floor():
    with pytest.raises(ValueError, match="threshold"):
        coverage_from_word_scores((), (), {}, score_threshold=1.1)
