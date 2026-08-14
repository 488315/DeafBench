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
        "Replacement of Unicode punctuation and symbol characters with spaces",
        "Whitespace collapse and trimming",
        "accumulated across the corpus",
        "Malformed text or sound labels fail closed",
        "finite, positive measurements",
        "legacy `wer` and `cer` fields are aliases of orthographic WER and CER",
        "eight` and `8` remain different",
        "Non-lexical records are excluded from WER and CER",
        "Total audio seconds divided by inference wall seconds",
        "Open ASR Leaderboard compatibility lane",
        "must stay labeled separately",
    ):
        assert required in methodology

    normalization_steps = [
        "Unicode NFKC normalization",
        "Unicode case folding",
        "Replacement of Unicode punctuation and symbol characters with spaces",
        "Whitespace collapse and trimming",
    ]
    assert [methodology.index(step) for step in normalization_steps] == sorted(
        methodology.index(step) for step in normalization_steps
    )
