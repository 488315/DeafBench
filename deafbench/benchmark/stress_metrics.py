"""Accessibility-specific summaries for paired clean and stressed ASR runs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from deafbench.metrics import evaluate_dataset
from deafbench.parser import align_records


def _evaluate(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = evaluate_dataset(
        align_records([dict(item) for item in references], [dict(item) for item in predictions])
    )
    edits = metrics["substitutions"] + metrics["insertions"] + metrics["deletions"]
    risk_by_id = {
        reference["id"]: reference["risk_categories"] for reference in references
    }
    failure_counts: Counter[str] = Counter()
    for failure in metrics["critical_failures"]:
        category = risk_by_id[failure["id"]].get(failure["expected"])
        if category is None:
            raise ValueError("Critical failure has no declared risk category")
        failure_counts[category] += 1
    return {
        "samples": metrics["samples"],
        "wer": metrics["wer"],
        "substitutions": metrics["substitutions"],
        "insertions": metrics["insertions"],
        "deletions": metrics["deletions"],
        "deletion_share_percent": metrics["deletions"] / edits * 100.0 if edits else 0.0,
        "strict_critical_recall": metrics["strict_critical_recall"],
        "canonical_critical_recall": metrics["canonical_critical_recall"],
        "critical_failures": metrics["critical_failures"],
        "critical_failures_by_risk": dict(sorted(failure_counts.items())),
        "word_errors_by_sample": metrics["word_errors_by_sample"],
    }


def summarize_stress_results(
    references: Sequence[Mapping[str, Any]],
    clean_predictions: Sequence[Mapping[str, Any]],
    stressed_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare paired runs without blending the clean and stressed lanes."""
    clean = _evaluate(references, clean_predictions)
    stressed = _evaluate(references, stressed_predictions)
    return {
        "clean": clean,
        "stressed": stressed,
        "degradation": {
            "wer_points": stressed["wer"] - clean["wer"],
            "strict_recall_points": (
                stressed["strict_critical_recall"]
                - clean["strict_critical_recall"]
            ),
            "canonical_recall_points": (
                stressed["canonical_critical_recall"]
                - clean["canonical_critical_recall"]
            ),
        },
    }


def evaluate_caption_timing(
    alignments: Sequence[Mapping[str, object]],
) -> dict[str, int | float]:
    """Measure absolute token timing drift against the 500 ms caption boundary."""
    if not alignments:
        raise ValueError("Caption timing requires at least one alignment")
    drifts: list[float] = []
    for alignment in alignments:
        token = alignment.get("token")
        reference_ms = alignment.get("reference_ms")
        prediction_ms = alignment.get("prediction_ms")
        if (
            not isinstance(token, str)
            or not token.strip()
            or isinstance(reference_ms, bool)
            or not isinstance(reference_ms, (int, float))
            or isinstance(prediction_ms, bool)
            or not isinstance(prediction_ms, (int, float))
            or not math.isfinite(float(reference_ms))
            or not math.isfinite(float(prediction_ms))
            or float(reference_ms) < 0.0
            or float(prediction_ms) < 0.0
        ):
            raise ValueError("Caption timing alignment is invalid")
        drifts.append(abs(float(prediction_ms) - float(reference_ms)))
    over_threshold = sum(drift > 500.0 for drift in drifts)
    return {
        "tokens": len(drifts),
        "median_absolute_drift_ms": median(drifts),
        "maximum_absolute_drift_ms": max(drifts),
        "over_500ms": over_threshold,
        "over_500ms_percent": over_threshold / len(drifts) * 100.0,
    }
