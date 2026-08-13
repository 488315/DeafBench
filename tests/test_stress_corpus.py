from collections import Counter
import hashlib
from pathlib import Path

from deafbench.benchmark.stress_contract import (
    RISK_CATEGORIES,
    load_stress_cases,
)


CORPUS = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "accessibility-stress-v1"
    / "references.jsonl"
)
CORPUS_SHA256 = "b47adac789092a1b8094c450340b178d64a06cccaf124893e8c4c7622ae73b61"


def test_stress_corpus_bytes_are_frozen() -> None:
    assert hashlib.sha256(CORPUS.read_bytes()).hexdigest() == CORPUS_SHA256


def test_stress_corpus_has_unique_cases_and_clean_controls() -> None:
    cases = load_stress_cases(CORPUS)

    assert len(cases) == 24
    assert [case["id"] for case in cases] == [
        f"stress-{index:03d}" for index in range(1, 25)
    ]
    assert len({case["text"] for case in cases}) == len(cases)
    assert all(case["stressors"][0] == {"kind": "clean"} for case in cases)


def test_stress_corpus_covers_every_declared_risk_category() -> None:
    cases = load_stress_cases(CORPUS)
    counts = Counter(
        category
        for case in cases
        for category in case["risk_categories"].values()
    )

    assert set(counts) == RISK_CATEGORIES
    assert min(counts.values()) >= 2


def test_stress_corpus_covers_each_supported_stressor_family() -> None:
    cases = load_stress_cases(CORPUS)
    kinds = {
        stressor["kind"]
        for case in cases
        for stressor in case["stressors"]
    }

    assert kinds == {
        "additive_noise",
        "clean",
        "compression",
        "interstitial_noise",
        "long_pause",
        "overlap",
        "rate",
        "reverberation",
        "telephony",
    }
