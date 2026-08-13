import math

import pytest

from deafbench.benchmark.models._isolated_result import (
    required_mapping,
    validated_records,
)


def test_validated_records_preserves_ordered_predictions() -> None:
    records = validated_records(
        [{"id": "sample-1", "latency_ms": 1.25, "text": "caption"}],
        ["sample-1"],
        worker_name="Example",
    )

    assert records == [{"id": "sample-1", "latency_ms": 1.25, "text": "caption"}]


@pytest.mark.parametrize("latency", [math.nan, math.inf, -math.inf, -1.0, True])
def test_validated_records_rejects_unsafe_latency(latency: object) -> None:
    with pytest.raises(ValueError, match="Example worker returned an invalid prediction"):
        validated_records(
            [{"id": "sample-1", "latency_ms": latency, "text": "caption"}],
            ["sample-1"],
            worker_name="Example",
        )


def test_required_mapping_names_missing_worker_metadata() -> None:
    with pytest.raises(ValueError, match="Example worker omitted decoding"):
        required_mapping({}, "decoding", worker_name="Example")
