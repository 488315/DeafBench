"""Versioned conventional ASR measurements for DeafBench."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from typing import Any

import jiwer


NORMALIZATION_POLICY = "deafbench-asr-normalization-v1"


def normalize_asr_text(text: str) -> str:
    """Apply the disclosed v1 lexical ASR normalization policy."""
    if not isinstance(text, str):
        raise TypeError("ASR text must be a string")

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return " ".join(normalized.split())


def _validate_corpus(references: Sequence[str], predictions: Sequence[str]) -> None:
    if isinstance(references, str) or isinstance(predictions, str):
        raise TypeError("references and predictions must be sequences of strings")
    if not references:
        raise ValueError("ASR evaluation requires at least one reference")
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    if not all(isinstance(value, str) for value in (*references, *predictions)):
        raise TypeError("references and predictions must contain strings")


def _percent(value: float) -> float:
    result = float(value) * 100.0
    if not math.isfinite(result):
        raise ValueError("ASR metric calculation produced a non-finite value")
    return result


def _word_metrics(references: list[str], predictions: list[str]) -> dict[str, Any]:
    result = jiwer.process_words(references, predictions)
    return {
        "wer": _percent(result.wer),
        "substitutions": result.substitutions,
        "insertions": result.insertions,
        "deletions": result.deletions,
    }


def evaluate_conventional_asr(
    references: Sequence[str], predictions: Sequence[str]
) -> dict[str, Any]:
    """Return corpus-level orthographic and normalized ASR measurements."""
    _validate_corpus(references, predictions)
    orthographic_references = [" ".join(value.split()) for value in references]
    orthographic_predictions = [" ".join(value.split()) for value in predictions]
    normalized_references = [normalize_asr_text(value) for value in references]
    normalized_predictions = [normalize_asr_text(value) for value in predictions]

    empty_ids = [
        str(index)
        for index, reference in enumerate(normalized_references)
        if not reference
    ]
    if empty_ids:
        raise ValueError(
            "reference text is empty after ASR normalization at indexes: "
            + ", ".join(empty_ids)
        )

    orthographic = _word_metrics(orthographic_references, orthographic_predictions)
    normalized = _word_metrics(normalized_references, normalized_predictions)
    return {
        "normalization_policy": NORMALIZATION_POLICY,
        "orthographic_wer": orthographic["wer"],
        "normalized_wer": normalized["wer"],
        "orthographic_cer": _percent(
            jiwer.cer(orthographic_references, orthographic_predictions)
        ),
        "normalized_cer": _percent(
            jiwer.cer(normalized_references, normalized_predictions)
        ),
        "orthographic_substitutions": orthographic["substitutions"],
        "orthographic_insertions": orthographic["insertions"],
        "orthographic_deletions": orthographic["deletions"],
        "normalized_substitutions": normalized["substitutions"],
        "normalized_insertions": normalized["insertions"],
        "normalized_deletions": normalized["deletions"],
    }
