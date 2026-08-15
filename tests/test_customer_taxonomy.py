import pytest

from deafbench.pilot.taxonomy import (
    category_label,
    category_values,
    classify_failure,
    factor_label,
    severity_label,
    severity_values,
)


pytestmark = pytest.mark.unit


def test_taxonomy_exposes_all_customer_categories() -> None:
    values = category_values()

    assert len(values) == 18
    assert values[0] == "codes_passwords_login_information"
    assert values[-1] == "made_up_captions_wrong_context"


def test_health_impact_wins_over_numeric_error_mechanism() -> None:
    finding = classify_failure(
        expected="0.5 milligrams",
        reference_text="Do not take more than 0.5 milligrams.",
        predicted_text="Take more than 5 milligrams.",
        error_kinds=("delete", "substitute"),
    )

    assert finding.primary_category == "health_medicine_safety"
    assert finding.severity == "critical"
    assert "negation" in finding.related_factors
    assert "decimal_point" in finding.related_factors
    assert "dosage" in finding.related_factors
    assert "meaning_reversal" in finding.related_factors
    assert "delete" in finding.related_factors
    assert "substitute" in finding.related_factors
    assert "health" in finding.impact.casefold()
    assert "safety-critical" in finding.recommendation.casefold()


def test_code_failure_uses_domain_category_and_consequence_severity() -> None:
    finding = classify_failure(
        expected="83927",
        reference_text="Your confirmation code is 83927.",
        predicted_text="Your confirmation code is 83972.",
        entity_type="CODE",
        error_kinds=("substitute",),
    )

    assert finding.primary_category == "codes_passwords_login_information"
    assert finding.severity == "major"
    assert finding.related_factors == ("code", "substitute")


def test_sound_and_speaker_failures_are_first_class_categories() -> None:
    sound = classify_failure(
        expected="[smoke alarm]",
        reference_text="",
        predicted_text="",
        finding_kind="sound",
    )
    speaker = classify_failure(
        expected="Alex",
        reference_text="Do not deploy it.",
        predicted_text="Do not deploy it.",
        finding_kind="speaker",
    )

    assert sound.primary_category == "important_sounds"
    assert sound.severity == "critical"
    assert "sound_event" in sound.related_factors
    assert speaker.primary_category == "who_is_speaking"
    assert speaker.severity == "major"
    assert "speaker_attribution" in speaker.related_factors


@pytest.mark.parametrize(
    ("expected", "reference", "prediction", "entity_type", "category"),
    [
        ("Friday at 3:15 PM", "Meet Friday at 3:15 PM.", "Meet Friday at 3:50 PM.", "TIME", "times_dates_appointments"),
        ("Nguyen", "Send the paperwork to Nguyen.", "Send the paperwork to win.", "PROPER_NAME", "people_companies_place_names"),
        ("Office Guest", "Join Wi-Fi Office Guest.", "Join Wi-Fi Office Guess.", "SSID", "contact_internet_information"),
        ("git checkout --detach", "Run git checkout --detach.", "Run get checkout detach.", None, "computer_repair_technical_instructions"),
        ("do not", "Do not restart the server.", "Restart the server.", None, "computer_repair_technical_instructions"),
        ("kill", "The Unix command is kill.", "The Unix command is ****.", None, "computer_repair_technical_instructions"),
        ("12.6", "The battery should read 12.6 volts.", "The battery should read 126 volts.", None, "money_numbers_measurements"),
        ("red release handle", "Use the red release handle.", "Use the release handle.", None, "other_critical_information"),
        ("left", "Turn left after the second light.", "Turn right after the second light.", None, "directions_location_instructions"),
    ],
)
def test_taxonomy_routes_common_real_world_domains(
    expected: str,
    reference: str,
    prediction: str,
    entity_type: str | None,
    category: str,
) -> None:
    finding = classify_failure(
        expected=expected,
        reference_text=reference,
        predicted_text=prediction,
        entity_type=entity_type,
        error_kinds=("substitute",),
    )

    assert finding.primary_category == category
    assert finding.impact
    assert finding.recommendation


def test_negation_detection_uses_token_boundaries() -> None:
    finding = classify_failure(
        expected="do not enter",
        reference_text="Do not enter the room.",
        predicted_text="The donation center is open.",
        error_kinds=("delete",),
    )

    assert "negation" in finding.related_factors


def test_censorship_category_applies_when_no_higher_impact_domain_matches() -> None:
    finding = classify_failure(
        expected="profanity",
        reference_text="The quoted word is profanity.",
        predicted_text="The quoted word is ****.",
    )

    assert finding.primary_category == "censorship_changed_words"


def test_customer_labels_are_sentence_case_not_machine_values() -> None:
    assert category_label("health_medicine_safety") == "Health, medicine & safety"
    assert category_label("future_category") == "Future category"
    assert severity_label("no_real_impact") == "No real impact"
    assert severity_label("critical") == "Critical"
    assert severity_label("future_level") == "Future level"
    assert severity_values() == (
        "no_real_impact",
        "minor",
        "moderate",
        "major",
        "critical",
    )
    assert factor_label("decimal_point") == "Decimal point"
    assert "_" not in category_label("codes_passwords_login_information")
