import pytest

from deafbench.benchmark.independent_asr import collapse_ctc_labels


def test_ctc_decoder_collapses_repeats_blanks_and_word_boundaries():
    labels = ("-", "|", "A", "B")

    text = collapse_ctc_labels(
        (0, 1, 2, 2, 0, 3, 1, 1, 2),
        labels,
        blank=0,
    )

    assert text == "ab a"


def test_ctc_decoder_rejects_unknown_label_index():
    with pytest.raises(ValueError, match="label index"):
        collapse_ctc_labels((9,), ("-", "A"), blank=0)
