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
    ("sample_id", "term", "expected_words"),
    [
        ("core-001", "2:15 PM", ("two", "fifteen", "p", "m")),
        (
            "core-009",
            "83927",
            ("eight", "three", "nine", "two", "seven"),
        ),
        ("core-009", "4:45 PM", ("four", "forty", "five", "p", "m")),
        (
            "core-011",
            "dev_user twenty three",
            ("dev", "underscore", "user", "twenty", "three"),
        ),
        (
            "core-011",
            "481926",
            ("four", "eight", "one", "nine", "two", "six"),
        ),
        (
            "core-016",
            "seven four nine two six eight one",
            ("seven", "four", "nine", "two", "six", "eight", "one"),
        ),
    ],
)
def test_replacement_entities_have_explicit_spoken_forms(
    sample_id: str,
    term: str,
    expected_words: tuple[str, ...],
):
    record = _records()[sample_id]

    prepared = prepare_spoken_reference(record["text"], record["critical_types"])
    start, end = prepared.entity_word_ranges[term]

    assert prepared.words[start:end] == expected_words
    assert prepared.spoken_aliases[term] == " ".join(expected_words)
    if record["critical_types"][term] in {"DIGIT_SEQUENCE", "CODE", "PASSWORD"} and any(
        character.isdigit() for character in term
    ):
        assert f'<say-as interpret-as="telephone">{term}</say-as>' in prepared.ssml
    else:
        assert f'alias="{" ".join(expected_words)}"' in prepared.ssml
    assert prepared.reference_sha256 == hashlib.sha256(
        record["text"].encode("utf-8")
    ).hexdigest()


def test_spoken_reference_rejects_missing_or_overlapping_typed_terms():
    with pytest.raises(ValueError, match="not present"):
        prepare_spoken_reference("hello", {"missing": "PROPER_NAME"})
    with pytest.raises(ValueError, match="overlap"):
        prepare_spoken_reference("abc", {"abc": "CODE", "bc": "CODE"})
