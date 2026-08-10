"""Reference-preserving spoken forms for synthesis and forced alignment."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html import escape
from types import MappingProxyType
from typing import Mapping

from deafbench.critical_entities import ENTITY_TYPES


_DIGITS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
_SMALL_NUMBERS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
}


@dataclass(frozen=True)
class SpokenReference:
    """One exact reference plus its explicit acoustic rendering contract."""

    reference_sha256: str
    words: tuple[str, ...]
    entity_word_ranges: Mapping[str, tuple[int, int]]
    spoken_aliases: Mapping[str, str]
    ssml: str


def _number_words(value: int) -> tuple[str, ...]:
    if value in _SMALL_NUMBERS:
        return tuple(_SMALL_NUMBERS[value].split())
    tens, ones = divmod(value, 10)
    if 2 <= tens <= 5 and ones:
        return (_SMALL_NUMBERS[tens * 10], _SMALL_NUMBERS[ones])
    raise ValueError(f"number cannot be rendered by the spoken profile: {value}")


def _normalized_words(value: str) -> tuple[str, ...]:
    value = value.casefold().replace("_", " underscore ")
    value = re.sub(
        r"\d",
        lambda match: f" {_DIGITS[int(match.group(0))]} ",
        value,
    )
    return tuple(re.findall(r"[a-z]+(?:'[a-z]+)?", value))


def _time_alias(term: str) -> tuple[str, ...]:
    match = re.fullmatch(
        r"\s*(\d{1,2}):([0-5]\d)\s*([ap])\.?\s*m\.?\s*",
        term,
        re.IGNORECASE,
    )
    if match is None:
        words = _normalized_words(term)
        if not words:
            raise ValueError(f"unsupported TIME spoken form: {term}")
        return words
    hour = int(match.group(1))
    if not 1 <= hour <= 12:
        raise ValueError(f"unsupported TIME hour: {term}")
    minute = int(match.group(2))
    minute_words = ("oh", _DIGITS[minute]) if minute < 10 else _number_words(minute)
    return (*_number_words(hour), *minute_words, match.group(3).casefold(), "m")


def _spoken_alias(term: str, entity_type: str) -> tuple[str, ...]:
    if entity_type == "TIME":
        return _time_alias(term)
    if entity_type in {"DIGIT_SEQUENCE", "CODE", "PASSWORD"} and re.search(
        r"\d", term
    ):
        return tuple(_DIGITS[int(digit)] for digit in re.findall(r"\d", term))
    words = _normalized_words(term)
    if not words:
        raise ValueError(f"empty spoken form for typed entity: {term}")
    return words


def _typed_spans(
    reference_text: str,
    critical_types: Mapping[str, str],
) -> tuple[tuple[int, int, str, str], ...]:
    unknown = set(critical_types.values()) - ENTITY_TYPES
    if unknown:
        raise ValueError(f"unknown critical entity types: {sorted(unknown)}")
    spans: list[tuple[int, int, str, str]] = []
    for term, entity_type in critical_types.items():
        if reference_text.count(term) != 1:
            raise ValueError(f"typed critical term is not present exactly once: {term}")
        start = reference_text.index(term)
        spans.append((start, start + len(term), term, entity_type))
    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise ValueError("typed critical terms overlap")
    return tuple(spans)


def prepare_spoken_reference(
    reference_text: str,
    critical_types: Mapping[str, str],
) -> SpokenReference:
    """Create deterministic SSML and alignment words without changing the reference."""
    spans = _typed_spans(reference_text, critical_types)
    words: list[str] = []
    ssml_parts = ['<speak version="1.0" xml:lang="en-US">']
    entity_ranges: dict[str, tuple[int, int]] = {}
    aliases: dict[str, str] = {}
    cursor = 0
    for start, end, term, entity_type in spans:
        prefix = reference_text[cursor:start]
        words.extend(_normalized_words(prefix))
        ssml_parts.append(escape(prefix))

        alias_words = _spoken_alias(term, entity_type)
        range_start = len(words)
        words.extend(alias_words)
        entity_ranges[term] = (range_start, len(words))
        alias = " ".join(alias_words)
        aliases[term] = alias
        ssml_parts.append(f'<sub alias="{escape(alias, quote=True)}">{escape(term)}</sub>')
        cursor = end

    suffix = reference_text[cursor:]
    words.extend(_normalized_words(suffix))
    ssml_parts.extend((escape(suffix), "</speak>"))
    return SpokenReference(
        reference_sha256=hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
        words=tuple(words),
        entity_word_ranges=MappingProxyType(entity_ranges),
        spoken_aliases=MappingProxyType(aliases),
        ssml="".join(ssml_parts),
    )
