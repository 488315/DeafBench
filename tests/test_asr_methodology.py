from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_methodology_discloses_normalization_and_metric_boundaries() -> None:
    methodology = " ".join(
        (ROOT / "docs/asr-evaluation-methodology.md")
        .read_text(encoding="utf-8")
        .split()
    )

    for required in (
        "deafbench-asr-normalization-v1",
        "Unicode NFKC normalization",
        "Unicode case folding",
        "eight` and `8` remain different",
        "Total audio seconds divided by inference wall seconds",
        "Open ASR Leaderboard compatibility lane",
        "must stay labeled separately",
    ):
        assert required in methodology
