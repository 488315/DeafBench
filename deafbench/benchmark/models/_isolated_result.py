"""Validation shared by isolated model-adapter result boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def validated_records(
    payload: object,
    expected_ids: Sequence[str],
    *,
    worker_name: str,
) -> list[dict[str, object]]:
    """Validate complete, ordered predictions returned by an isolated worker."""
    if not isinstance(payload, list) or len(payload) != len(expected_ids):
        raise ValueError(f"{worker_name} worker returned incomplete predictions")
    records: list[dict[str, object]] = []
    for raw_record, expected_id in zip(payload, expected_ids, strict=True):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"{worker_name} worker returned an invalid prediction")
        sample_id = raw_record.get("id")
        text = raw_record.get("text")
        latency = raw_record.get("latency_ms")
        if (
            sample_id != expected_id
            or not isinstance(text, str)
            or isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError(f"{worker_name} worker returned an invalid prediction")
        records.append({"id": sample_id, "latency_ms": latency, "text": text})
    return records


def required_mapping(
    payload: Mapping[str, Any],
    field: str,
    *,
    worker_name: str,
) -> Mapping[str, object]:
    """Return required mapping metadata from an isolated worker result."""
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{worker_name} worker omitted {field}")
    return value
