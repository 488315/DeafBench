"""Typed, fail-closed comparison for critical caption entities."""

from __future__ import annotations

import re
from collections.abc import Callable

from .parser import normalize_text


ENTITY_TYPES = frozenset(
    {
        "TIME",
        "DIGIT_SEQUENCE",
        "USERNAME",
        "CODE",
        "PASSWORD",
        "SSID",
        "PROPER_NAME",
    }
)

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_NUMBER_WORDS = {
    **_DIGIT_WORDS,
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
}
_DIGIT_WORD_PATTERN = "(?:" + "|".join(_DIGIT_WORDS) + ")"
_HOUR_WORD_PATTERN = "(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_MINUTE_WORD_PATTERN = (
    "(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    "twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    "nineteen|twenty(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?|"
    "thirty(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?|"
    "forty(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?|"
    "fifty(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?)"
)


def strict_contains(term: str, prediction: str) -> bool:
    """Match the case-folded surface form without semantic rewrites."""
    surface_term = re.sub(r"\s+", " ", term.casefold()).strip()
    surface_prediction = re.sub(r"\s+", " ", prediction.casefold()).strip()
    if not surface_term:
        return False
    return bool(
        re.search(rf"(?<!\w){re.escape(surface_term)}(?!\w)", surface_prediction)
    )


def _number_value(value: str) -> int | None:
    tokens = re.split(r"[ -]+", value.casefold())
    if any(token not in _NUMBER_WORDS for token in tokens):
        return None
    numbers = [int(_NUMBER_WORDS[token]) for token in tokens]
    if len(numbers) == 2 and numbers[0] >= 20 and numbers[0] % 10 == 0:
        return numbers[0] + numbers[1]
    if len(numbers) == 1:
        return numbers[0]
    return None


def _canonical_time(hour: int, minute: int, meridiem: str) -> str | None:
    if not 1 <= hour <= 12 or not 0 <= minute <= 59:
        return None
    return f"{hour}:{minute:02d}{meridiem.casefold()}"


def _time_values(text: str) -> set[str]:
    normalized = re.sub(r"\b([ap])\s*\.?\s*m\.?(?!\w)", r"\1m", text.casefold())
    values: set[str] = set()
    numeric_pattern = re.compile(
        r"(?<!\w)(\d{1,2})(?:\s*[:.]\s*|\s+o['’]?clock\s+|\s+)(\d{1,2})\s*(am|pm)\b"
    )
    for match in numeric_pattern.finditer(normalized):
        value = _canonical_time(int(match.group(1)), int(match.group(2)), match.group(3))
        if value is not None:
            values.add(value)
    for match in re.finditer(r"(?<!\w)(\d{1,2})\s*(am|pm)\b", normalized):
        value = _canonical_time(int(match.group(1)), 0, match.group(2))
        if value is not None:
            values.add(value)
    for match in re.finditer(
        rf"(?<!\w)({_HOUR_WORD_PATTERN})\s*(am|pm)\b", normalized
    ):
        hour = _number_value(match.group(1))
        if hour is not None:
            value = _canonical_time(hour, 0, match.group(2))
            if value is not None:
                values.add(value)
    spoken_pattern = re.compile(
        rf"(?<!\w)({_HOUR_WORD_PATTERN})[ -]+({_MINUTE_WORD_PATTERN})\s*(am|pm)\b"
    )
    for match in spoken_pattern.finditer(normalized):
        hour = _number_value(match.group(1))
        minute = _number_value(match.group(2))
        if hour is not None and minute is not None:
            value = _canonical_time(hour, minute, match.group(3))
            if value is not None:
                values.add(value)
    return values


def _canonical_digit_sequence(value: str) -> str | None:
    tokens = re.findall(r"[a-z]+|\d+", value.casefold())
    if not tokens:
        return None
    digits: list[str] = []
    for token in tokens:
        if token.isdigit():
            digits.append(token)
        elif token in _DIGIT_WORDS:
            digits.append(_DIGIT_WORDS[token])
        else:
            return None
    return "".join(digits)


def _digit_sequence_values(text: str) -> set[str]:
    values = set(re.findall(r"(?<!\w)\d+(?!\w)", text))
    spoken = re.compile(
        rf"(?<!\w){_DIGIT_WORD_PATTERN}(?:[ -]+{_DIGIT_WORD_PATTERN})*(?!\w)",
        re.IGNORECASE,
    )
    values.update(
        canonical
        for match in spoken.finditer(text)
        if (canonical := _canonical_digit_sequence(match.group(0))) is not None
    )
    separated_digits = re.compile(r"(?<!\w)\d(?:[ -]+\d)+(?!\w)")
    values.update(
        re.sub(r"[ -]+", "", match.group(0))
        for match in separated_digits.finditer(text)
    )
    return values


def _replace_spoken_number(match: re.Match[str]) -> str:
    value = _number_value(match.group(0))
    return str(value) if value is not None else match.group(0)


def _canonical_username_text(text: str) -> str:
    value = text.casefold()
    value = re.sub(r"\bunderscore\b", "_", value)
    number_pattern = "(?:" + "|".join(_NUMBER_WORDS) + ")"
    value = re.sub(
        rf"\b{number_pattern}(?:[ -]+{number_pattern})?\b",
        _replace_spoken_number,
        value,
    )
    value = re.sub(r"\s*_\s*", "_", value)
    value = re.sub(r"(?<=[a-z_])\s+(?=\d)", "", value)
    return value


def _username_matches(term: str, prediction: str) -> bool:
    expected = _canonical_username_text(term).strip()
    candidate_text = _canonical_username_text(prediction)
    return bool(
        expected
        and re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", candidate_text)
    )


def _spacing_case_matches(term: str, prediction: str) -> bool:
    chunks = term.casefold().split()
    if not chunks:
        return False
    pattern = r"\s*".join(re.escape(chunk) for chunk in chunks)
    return bool(re.search(rf"(?<!\w){pattern}(?!\w)", prediction.casefold()))


def _exact_code_matches(term: str, prediction: str) -> bool:
    """Compare codes literally after case-folding and spoken-digit conversion."""
    expected = term.casefold().strip()
    for word, digit in _DIGIT_WORDS.items():
        expected = re.sub(rf"\b{word}\b", digit, expected)
    expected = re.sub(r"\s+", "", expected)
    if not expected:
        return False

    candidates = set(
        re.findall(r"(?<!\w)(?:[a-z]+\d+|\d+)(?!\w)", prediction.casefold())
    )
    spoken_code = re.compile(
        rf"(?<!\w)(?:[a-z]+[ ]+)?{_DIGIT_WORD_PATTERN}"
        rf"(?:[ ]+{_DIGIT_WORD_PATTERN})*(?!\w)",
        re.IGNORECASE,
    )
    for match in spoken_code.finditer(prediction):
        candidate = match.group(0).casefold()
        for word, digit in _DIGIT_WORDS.items():
            candidate = re.sub(rf"\b{word}\b", digit, candidate)
        candidates.add(re.sub(r"\s+", "", candidate))
    return expected in candidates


_MATCHERS: dict[str, Callable[[str, str], bool]] = {
    "USERNAME": _username_matches,
    "CODE": _exact_code_matches,
    "PASSWORD": _exact_code_matches,
    "SSID": _spacing_case_matches,
    "PROPER_NAME": _spacing_case_matches,
}


def canonical_contains(term: str, prediction: str, entity_type: str | None) -> bool:
    """Compare one entity with only the normalization allowed by its type."""
    if entity_type == "TIME":
        expected = _time_values(term)
        return len(expected) == 1 and not expected.isdisjoint(_time_values(prediction))
    if entity_type == "DIGIT_SEQUENCE":
        expected = _canonical_digit_sequence(term)
        return expected is not None and expected in _digit_sequence_values(prediction)
    if entity_type in _MATCHERS:
        return _MATCHERS[entity_type](term, prediction)
    normalized_term = normalize_text(term)
    normalized_prediction = normalize_text(prediction)
    return bool(
        normalized_term
        and re.search(
            rf"(?<!\w){re.escape(normalized_term)}(?!\w)", normalized_prediction
        )
    )
