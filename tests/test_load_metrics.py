import pytest

from deafbench.benchmark.load_metrics import summarize_load_trial


def test_load_summary_reports_latency_throughput_and_resource_peaks() -> None:
    summary = summarize_load_trial(
        [
            {
                "audio_seconds": 10.0,
                "latency_ms": 2_000.0,
                "ttfb_ms": 200.0,
                "peak_vram_bytes": 2_000,
                "peak_cpu_percent": 70.0,
            },
            {
                "audio_seconds": 20.0,
                "latency_ms": 4_000.0,
                "ttfb_ms": 600.0,
                "peak_vram_bytes": 3_000,
                "peak_cpu_percent": 90.0,
            },
        ],
        concurrency=2,
        wall_seconds=5.0,
    )

    assert summary == {
        "requests": 2,
        "concurrency": 2,
        "audio_seconds": 30.0,
        "wall_seconds": 5.0,
        "throughput_requests_per_second": 0.4,
        "aggregate_rtf": pytest.approx(1 / 6),
        "aggregate_rtfx": 6.0,
        "median_latency_ms": 3_000.0,
        "p95_latency_ms": 3_900.0,
        "median_ttfb_ms": 400.0,
        "p95_ttfb_ms": 580.0,
        "ttfb_over_500ms": 1,
        "peak_vram_bytes": 3_000,
        "peak_cpu_percent": 90.0,
    }


@pytest.mark.parametrize(
    ("observations", "concurrency", "wall_seconds", "message"),
    [
        ([], 1, 1.0, "at least one observation"),
        ([{"audio_seconds": 1, "latency_ms": 1, "ttfb_ms": 1}], 0, 1.0, "concurrency"),
        ([{"audio_seconds": 1, "latency_ms": 1, "ttfb_ms": 1}], 1, 0.0, "wall time"),
        ([{"audio_seconds": 0, "latency_ms": 1, "ttfb_ms": 1}], 1, 1.0, "audio_seconds"),
        ([{"audio_seconds": 1, "latency_ms": -1, "ttfb_ms": 1}], 1, 1.0, "latency_ms"),
        ([{"audio_seconds": 1, "latency_ms": 1, "ttfb_ms": True}], 1, 1.0, "ttfb_ms"),
    ],
)
def test_load_summary_rejects_invalid_trials(
    observations,
    concurrency: int,
    wall_seconds: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_load_trial(
            observations,
            concurrency=concurrency,
            wall_seconds=wall_seconds,
        )


def test_load_summary_allows_unmeasured_optional_resources() -> None:
    summary = summarize_load_trial(
        [{"audio_seconds": 1.0, "latency_ms": 2.0, "ttfb_ms": 1.0}],
        concurrency=1,
        wall_seconds=0.5,
    )

    assert summary["p95_latency_ms"] == 2.0
    assert summary["peak_vram_bytes"] is None
    assert summary["peak_cpu_percent"] is None


def test_load_summary_rejects_impossible_cpu_measurement() -> None:
    with pytest.raises(ValueError, match="must not exceed 100"):
        summarize_load_trial(
            [
                {
                    "audio_seconds": 1.0,
                    "latency_ms": 2.0,
                    "ttfb_ms": 1.0,
                    "peak_cpu_percent": 101.0,
                }
            ],
            concurrency=1,
            wall_seconds=0.5,
        )
