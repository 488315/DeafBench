"""Deterministic customer-facing classification for caption audit findings."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FindingClassification:
    primary_category: str
    related_factors: tuple[str, ...]
    severity: str
    impact: str
    recommendation: str


_category_labels = {
    "codes_passwords_login_information": "Codes, passwords & login information",
    "times_dates_appointments": "Times, dates & appointments",
    "people_companies_place_names": "People, companies & place names",
    "contact_internet_information": "Contact & internet information",
    "money_numbers_measurements": "Money, numbers & measurements",
    "computer_repair_technical_instructions": "Computer, repair & technical instructions",
    "words_that_change_meaning": "Words that change the meaning",
    "health_medicine_safety": "Health, medicine & safety",
    "directions_location_instructions": "Directions & location instructions",
    "who_is_speaking": "Who is speaking",
    "important_sounds": "Important sounds",
    "censorship_changed_words": "Censorship & changed words",
    "other_important_word_errors": "Other important word errors",
    "other_critical_information": "Other critical information",
    "caption_timing_completeness": "Captions that are too late, too early or missing",
    "accents_dialects_multiple_languages": "Accents, dialects & multiple languages",
    "punctuation_symbols_formatting": "Punctuation, symbols & formatting that change meaning",
    "made_up_captions_wrong_context": "Made-up captions & wrong context",
}

_severity_labels = {
    "no_real_impact": "No real impact",
    "minor": "Minor",
    "moderate": "Moderate",
    "major": "Major",
    "critical": "Critical",
}

_health_terms = re.compile(
    r"\b(?:dose|dosage|medicine|medication|milligram|milligrams|mg|allergy|allergies|"
    r"hospital|doctor|medical|emergency|evacuat(?:e|ion)|safety|danger|hazard|chemical)\b",
    re.IGNORECASE,
)
_code_terms = re.compile(
    r"\b(?:code|password|pin|verification|confirm(?:ation)?|login|log in|recovery|"
    r"account|order|ticket|case|serial|license key|security key|otp)\b",
    re.IGNORECASE,
)
_technical_terms = re.compile(
    r"\b(?:command|file|filename|server|api|error|version|software|program|git|"
    r"systemctl|configuration|config|diagnostic|trouble code|port)\b",
    re.IGNORECASE,
)
_direction_terms = re.compile(
    r"\b(?:left|right|north|south|east|west|upstairs|downstairs|above|below|"
    r"inside|outside|entrance|exit|floor|lane|route|turn|clockwise|counterclockwise)\b",
    re.IGNORECASE,
)
_negation_terms = re.compile(
    r"\b(?:no|not|never|can't|cannot|don't|do not|shouldn't|must not|except|unless|"
    r"before|after|more than|less than|at least|at most|only)\b",
    re.IGNORECASE,
)


def category_values() -> tuple[str, ...]:
    return tuple(_category_labels)


def category_label(value: str) -> str:
    return _category_labels.get(value, value.replace("_", " ").capitalize())


def severity_label(value: str) -> str:
    return _severity_labels.get(value, value.replace("_", " ").capitalize())


def factor_label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _missing_negation(reference: str, prediction: str) -> bool:
    reference_terms = {
        match.group(0).casefold() for match in _negation_terms.finditer(reference)
    }
    prediction_terms = {
        match.group(0).casefold() for match in _negation_terms.finditer(prediction)
    }
    return bool(reference_terms - prediction_terms)


def _decimal_changed(reference: str, prediction: str) -> bool:
    reference_decimals = set(re.findall(r"(?<!\w)-?\d+\.\d+(?!\w)", reference))
    if not reference_decimals:
        return False
    prediction_numbers = set(re.findall(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)", prediction))
    return reference_decimals.isdisjoint(prediction_numbers)


def _primary_category(
    *,
    expected: str,
    reference_text: str,
    predicted_text: str,
    entity_type: str | None,
    finding_kind: str,
) -> str:
    context = f"{reference_text} {expected}"
    if finding_kind == "sound":
        return "important_sounds"
    if finding_kind == "speaker":
        return "who_is_speaking"
    if _health_terms.search(context):
        return "health_medicine_safety"
    if _direction_terms.search(context):
        return "directions_location_instructions"
    if entity_type in {"CODE", "PASSWORD"} or (
        entity_type == "DIGIT_SEQUENCE" and _code_terms.search(context)
    ):
        return "codes_passwords_login_information"
    if entity_type == "TIME":
        return "times_dates_appointments"
    if entity_type == "PROPER_NAME":
        return "people_companies_place_names"
    if entity_type in {"USERNAME", "SSID"}:
        return "contact_internet_information"
    if _technical_terms.search(context):
        return "computer_repair_technical_instructions"
    if _missing_negation(reference_text, predicted_text):
        return "words_that_change_meaning"
    if "****" in predicted_text and "****" not in reference_text:
        return "censorship_changed_words"
    if entity_type == "DIGIT_SEQUENCE" or re.search(r"\d", expected):
        return "money_numbers_measurements"
    return "other_critical_information"


def _severity(
    category: str,
    *,
    expected: str,
    reference_text: str,
    finding_kind: str,
) -> str:
    context = f"{reference_text} {expected}"
    if category == "health_medicine_safety":
        return "critical"
    if category == "important_sounds":
        return "critical" if re.search(r"alarm|siren|emergency|warning", expected, re.I) else "major"
    if category == "codes_passwords_login_information":
        return "critical" if re.search(r"password|security|recovery|login|log in", context, re.I) else "major"
    if category in {
        "times_dates_appointments",
        "contact_internet_information",
        "money_numbers_measurements",
        "computer_repair_technical_instructions",
        "words_that_change_meaning",
        "directions_location_instructions",
        "who_is_speaking",
        "caption_timing_completeness",
        "made_up_captions_wrong_context",
    }:
        return "major"
    if category == "people_companies_place_names":
        return "moderate"
    if finding_kind in {"critical", "sound", "speaker"}:
        return "major"
    return "moderate"


def _impact(category: str) -> str:
    return {
        "codes_passwords_login_information": "The caption changes information the user may need to authenticate, identify a case, or complete a transaction.",
        "times_dates_appointments": "The caption changes time-sensitive information, so the user could act at the wrong time or miss a deadline.",
        "people_companies_place_names": "The caption changes who or what is being identified, which can make the instruction ambiguous or incorrect.",
        "contact_internet_information": "The caption changes information the user may need to contact someone or connect to a service.",
        "money_numbers_measurements": "The caption changes a numeric value or measurement, so the displayed quantity no longer matches the spoken information.",
        "computer_repair_technical_instructions": "The caption changes technical information that may be required to reproduce a command, setting, or troubleshooting step.",
        "words_that_change_meaning": "The caption changes a word that controls the meaning or condition of the instruction.",
        "health_medicine_safety": "The caption changes health or safety information that a user may rely on when deciding what to do.",
        "directions_location_instructions": "The caption changes a direction or location instruction, so the user could go to or use the wrong place.",
        "who_is_speaking": "The caption does not preserve who said the statement, which can change how the conversation is understood.",
        "important_sounds": "The caption omits or changes meaningful audio information that is available to hearing users.",
        "censorship_changed_words": "The caption alters or removes spoken wording in a way that changes the usable information.",
        "caption_timing_completeness": "The caption is not available at the time the information is needed.",
        "made_up_captions_wrong_context": "The caption presents information that was not present in the source audio.",
        "other_critical_information": "The caption changes information marked as important for this audit.",
    }.get(category, "The caption error changes information that may affect understanding or task completion.")


def _recommendation(category: str) -> str:
    return {
        "codes_passwords_login_information": "Review alphanumeric token preservation, numeric decoding, and caption post-processing around codes and identifiers.",
        "times_dates_appointments": "Review time and date normalization, number preservation, and AM/PM handling.",
        "people_companies_place_names": "Review named-entity preservation and any vocabulary or contextual biasing used for names.",
        "contact_internet_information": "Review handling of usernames, network names, separators, and other contact or connection identifiers.",
        "money_numbers_measurements": "Review numeric-token preservation, decimal handling, units, and post-processing around quantities.",
        "computer_repair_technical_instructions": "Review technical vocabulary, punctuation preservation, and post-processing around commands and identifiers.",
        "words_that_change_meaning": "Review preservation of negation, conditions, directional terms, and short meaning-changing words.",
        "health_medicine_safety": "Review safety-critical vocabulary, negation, numeric preservation, and domain-specific post-processing before relying on these captions.",
        "directions_location_instructions": "Review directional vocabulary and number preservation in route or location instructions.",
        "who_is_speaking": "Review diarization, speaker-turn boundaries, and speaker-label assignment.",
        "important_sounds": "Review sound-event detection, event-to-caption mapping, and suppression of meaningful non-speech audio.",
        "censorship_changed_words": "Review content filtering and post-processing rules that alter recognized speech.",
        "caption_timing_completeness": "Review streaming latency, segmentation, caption lifetime, and synchronization behavior.",
        "made_up_captions_wrong_context": "Review hallucination controls, silence handling, context carry-over, and decoder reset behavior.",
        "other_critical_information": "Review the failed sample and add a specific preservation rule if this failure pattern repeats.",
    }.get(category, "Review the failed sample and the caption processing stage that changed the information.")


def classify_failure(
    *,
    expected: str,
    reference_text: str,
    predicted_text: str,
    entity_type: str | None = None,
    finding_kind: str = "critical",
    error_kinds: tuple[str, ...] = (),
) -> FindingClassification:
    category = _primary_category(
        expected=expected,
        reference_text=reference_text,
        predicted_text=predicted_text,
        entity_type=entity_type,
        finding_kind=finding_kind,
    )
    factors: list[str] = []
    if entity_type:
        factors.append(entity_type.casefold())
    if _missing_negation(reference_text, predicted_text):
        factors.append("negation")
    if _decimal_changed(reference_text, predicted_text):
        factors.append("decimal_point")
    if category == "health_medicine_safety" and re.search(
        r"\b(?:mg|milligram|milligrams|dose|dosage)\b", reference_text, re.I
    ):
        factors.append("dosage")
    if category == "directions_location_instructions":
        factors.append("direction")
    if finding_kind == "speaker":
        factors.append("speaker_attribution")
    if finding_kind == "sound":
        factors.append("sound_event")
    factors.extend(error_kinds)
    if _missing_negation(reference_text, predicted_text) or (
        category == "directions_location_instructions"
        and reference_text.casefold() != predicted_text.casefold()
    ):
        factors.append("meaning_reversal")
    return FindingClassification(
        primary_category=category,
        related_factors=tuple(dict.fromkeys(factors)),
        severity=_severity(
            category,
            expected=expected,
            reference_text=reference_text,
            finding_kind=finding_kind,
        ),
        impact=_impact(category),
        recommendation=_recommendation(category),
    )
