import pytest

from deafbench.asr_metrics import (
    NORMALIZATION_POLICY,
    evaluate_conventional_asr,
    normalize_asr_text,
)


def test_normalized_policy_changes_only_declared_surface_features() -> None:
    assert normalize_asr_text("  Café—WI-FI  ") == "café wi fi"
    assert normalize_asr_text("Meet at eight") != normalize_asr_text("Meet at 8")


def test_conventional_metrics_separate_orthographic_and_normalized_wer() -> None:
    metrics = evaluate_conventional_asr(
        ["Hello, WORLD"],
        ["hello world"],
    )

    assert metrics["normalization_policy"] == NORMALIZATION_POLICY
    assert metrics["orthographic_wer"] == 100.0
    assert metrics["normalized_wer"] == 0.0
    assert metrics["orthographic_substitutions"] == 2
    assert metrics["orthographic_insertions"] == 0
    assert metrics["orthographic_deletions"] == 0
    assert metrics["orthographic_cer"] > 0.0
    assert metrics["normalized_cer"] == 0.0


def test_normalized_wer_does_not_hide_number_word_errors() -> None:
    metrics = evaluate_conventional_asr(
        ["Meet at eight"],
        ["Meet at 8"],
    )

    assert metrics["normalized_wer"] == pytest.approx(100.0 / 3.0)
    assert metrics["normalized_substitutions"] == 1


def test_normalized_metrics_reject_empty_normalized_references() -> None:
    with pytest.raises(ValueError, match="empty after ASR normalization"):
        evaluate_conventional_asr(["..."], ["hallucination"])


@pytest.mark.parametrize(
    ("references", "predictions", "message"),
    [
        ([], [], "at least one"),
        (["one"], [], "same length"),
        (["one"], [1], "strings"),
        ("one", ["one"], "sequences of strings"),
        (["one"], "one", "sequences of strings"),
    ],
)
def test_conventional_metrics_reject_invalid_corpora(
    references: list[str], predictions: list[str], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        evaluate_conventional_asr(references, predictions)
