import hashlib
import json
from pathlib import Path

import pytest

from deafbench.benchmark.spoken_reference import prepare_spoken_reference


_ROOT = Path(__file__).parents[1]
_REPLACEMENTS = {"core-001", "core-009", "core-011", "core-016"}


def _records():
    return {
        record["id"]: record
        for line in (_ROOT / "benchmarks/core-v1/references.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if (record := json.loads(line))["id"] in _REPLACEMENTS
    }


@pytest.mark.parametrize(
    ("sample_id", "term", "expected_words", "expected_markup"),
    [
        (
            "core-001",
            "2:15 PM",
            ("two", "fifteen", "p", "m"),
            '<sub alias="two fifteen p m">2:15 PM</sub>',
        ),
        (
            "core-009",
            "83927",
            ("eight", "three", "nine", "two", "seven"),
            '<say-as interpret-as="telephone">83927</say-as>',
        ),
        (
            "core-009",
            "4:45 PM",
            ("four", "forty", "five", "p", "m"),
            '<sub alias="four forty five p m">4:45 PM</sub>',
        ),
        (
            "core-011",
            "dev_user twenty three",
            ("dev", "underscore", "user", "twenty", "three"),
            '<sub alias="dev underscore user twenty three">dev_user twenty three</sub>',
        ),
        (
            "core-011",
            "481926",
            ("four", "eight", "one", "nine", "two", "six"),
            '<say-as interpret-as="telephone">481926</say-as>',
        ),
        (
            "core-016",
            "seven four nine two six eight one",
            ("seven", "four", "nine", "two", "six", "eight", "one"),
            '<sub alias="seven four nine two six eight one">seven four nine two six eight one</sub>',
        ),
    ],
)
def test_replacement_entities_have_explicit_spoken_forms(
    sample_id: str,
    term: str,
    expected_words: tuple[str, ...],
    expected_markup: str,
):
    record = _records()[sample_id]

    prepared = prepare_spoken_reference(record["text"], record["critical_types"])
    start, end = prepared.entity_word_ranges[term]

    assert prepared.words[start:end] == expected_words
    assert prepared.spoken_aliases[term] == " ".join(expected_words)
    assert expected_markup in prepared.ssml
    assert prepared.reference_sha256 == hashlib.sha256(
        record["text"].encode("utf-8")
    ).hexdigest()


def test_spoken_reference_rejects_missing_or_overlapping_typed_terms():
    with pytest.raises(ValueError, match="not present"):
        prepare_spoken_reference("hello", {"missing": "PROPER_NAME"})
    with pytest.raises(ValueError, match="overlap"):
        prepare_spoken_reference("abc", {"abc": "CODE", "bc": "CODE"})


def test_spoken_reference_renders_untyped_numbers_as_individual_digits():
    prepared = prepare_spoken_reference(
        "Before 23, use A79 with dev_user after 45.",
        {"A79": "CODE", "dev_user": "USERNAME"},
    )

    assert '<sub alias="two three">23</sub>' in prepared.ssml
    assert '<sub alias="four five">45</sub>' in prepared.ssml
    assert '<say-as interpret-as="telephone">A79</say-as>' in prepared.ssml
    assert 'alias="dev underscore user"' in prepared.ssml
    assert prepared.words == (
        "before",
        "two",
        "three",
        "use",
        "seven",
        "nine",
        "with",
        "dev",
        "underscore",
        "user",
        "after",
        "four",
        "five",
    )
