"""Utterance-level diagnostics using injected official ASR operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


_ERROR_KEYS = ("del", "ins", "sub")


def _text_field(record: Mapping[str, Any], field: str, row: int) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"row {row} has invalid {field}")
    return value


def analyze_records(
    records: Iterable[Mapping[str, Any]],
    *,
    normalize: Callable[[str], str],
    edit_distance: Callable[..., Mapping[str, int | float]],
    limit: int = 20,
) -> dict[str, Any]:
    """Rank rows by official normalized error mass."""
    if limit <= 0:
        raise ValueError("analysis limit must be positive")

    totals = {key: 0 for key in _ERROR_KEYS}
    diagnostics: list[dict[str, Any]] = []
    reference_words = 0
    analyzed_rows = 0

    for row, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"row {row} must be an object")
        reference = normalize(_text_field(record, "text", row))
        prediction = normalize(_text_field(record, "pred_text", row))
        reference_tokens = tuple(reference.split())
        if not reference_tokens:
            raise ValueError(f"row {row} has empty normalized reference")

        counts = edit_distance(
            reference_tokens,
            tuple(prediction.split()),
            merge_compounds=True,
        )
        errors = {key: int(counts[key]) for key in _ERROR_KEYS}
        if any(value < 0 for value in errors.values()):
            raise ValueError(f"row {row} returned negative error counts")

        error_count = sum(errors.values())
        word_count = len(reference_tokens)
        analyzed_rows += 1
        reference_words += word_count
        for key, value in errors.items():
            totals[key] += value

        if error_count:
            diagnostics.append(
                {
                    "row": row,
                    "audio_filepath": record.get("audio_filepath"),
                    "duration": record.get("duration"),
                    "reference": reference,
                    "prediction": prediction,
                    "reference_words": word_count,
                    "errors": errors,
                    "wer": round(100 * error_count / word_count, 2),
                }
            )

    diagnostics.sort(
        key=lambda item: (
            -sum(item["errors"].values()),
            -item["wer"],
            item["row"],
        )
    )
    return {
        "analyzed_rows": analyzed_rows,
        "reference_words": reference_words,
        "errors": totals,
        "top_errors": diagnostics[:limit],
    }
