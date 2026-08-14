import math
import re
from typing import List, Dict, Any, Optional
import jiwer
from .asr_metrics import (
    NORMALIZATION_POLICY,
    evaluate_conventional_asr,
)
from .parser import normalize_text
from .critical_entities import ENTITY_TYPES, canonical_contains, strict_contains

_NUMBER_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
}
_DIGIT_WORDS = {
    "zero": "0",
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
_DIGIT_WORD_PATTERN = "(?:" + "|".join(_DIGIT_WORDS) + ")"
_NUMBER_WORD_PATTERN = "(?:" + "|".join((*_NUMBER_VALUES, *_NUMBER_SCALES)) + ")"
_HOUR_WORD_PATTERN = "(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_ONES_WORD_PATTERN = "(?:one|two|three|four|five|six|seven|eight|nine)"
_TEEN_WORD_PATTERN = "(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
_TENS_WORD_PATTERN = "(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_MINUTE_WORD_PATTERN = (
    rf"(?:zero|{_ONES_WORD_PATTERN}|{_TEEN_WORD_PATTERN}|"
    rf"{_TENS_WORD_PATTERN}(?:[\s-]+{_ONES_WORD_PATTERN})?)"
)
_YEAR_SUFFIX_PATTERN = (
    rf"(?:{_TEEN_WORD_PATTERN}|{_TENS_WORD_PATTERN}(?:[\s-]+{_ONES_WORD_PATTERN})?)"
)


def _parse_number_words(words: List[str]) -> int:
    total = 0
    current = 0
    for word in words:
        if word in _NUMBER_VALUES:
            current += _NUMBER_VALUES[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        else:
            total += max(current, 1) * _NUMBER_SCALES[word]
            current = 0
    return total + current


def _replace_money(match: re.Match[str]) -> str:
    dollars = int(match.group(1))
    cents = match.group(2)
    normalized = f"{dollars} dollars"
    if cents is not None:
        cents_value = int(cents.ljust(2, "0"))
        if cents_value:
            normalized += f" {cents_value} cents"
    return normalized


def _replace_numeric_meridiem_time(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2))
    meridiem = match.group(3)
    if minute == 0:
        return f"{hour} {meridiem}"
    return f"{hour} colon {minute:02d} {meridiem}"


def _replace_numeric_hour_meridiem(match: re.Match[str]) -> str:
    return f"{int(match.group(1))} {match.group(2)}"


def _replace_numeric_time(match: re.Match[str]) -> str:
    return f"{int(match.group(1))} colon {int(match.group(2)):02d}"


def _replace_spoken_time(match: re.Match[str]) -> str:
    hour = _NUMBER_VALUES[match.group("hour")]
    minute_words = re.split(r"[\s-]+", match.group("minute"))
    minute = _parse_number_words(minute_words)
    meridiem = match.group("meridiem")
    if minute == 0:
        return f"{hour} {meridiem}"
    return f"{hour} colon {minute:02d} {meridiem}"


def _replace_spoken_year(match: re.Match[str]) -> str:
    century = _NUMBER_VALUES[match.group("century")] * 100
    suffix_words = re.split(r"[\s-]+", match.group("suffix"))
    return str(century + _parse_number_words(suffix_words))


def _replace_spoken_digits(match: re.Match[str]) -> str:
    words = re.split(r"[\s-]+", match.group(0))
    return "".join(_DIGIT_WORDS[word] for word in words)


def _replace_number_words(match: re.Match[str]) -> str:
    words = re.split(r"[\s-]+", match.group(0))
    return str(_parse_number_words(words))


def _replace_ipv4(match: re.Match[str]) -> str:
    candidate = match.group(0)
    octets = candidate.split(".")
    if all(0 <= int(octet) <= 255 for octet in octets):
        return candidate.replace(".", " dot ")
    return candidate


def _replace_dotted_version(match: re.Match[str]) -> str:
    return match.group(0).replace(".", " point ")


def _replace_contextual_version(match: re.Match[str]) -> str:
    version = match.group(2).replace(".", " point ")
    return f"{match.group(1)} {version}"


def _normalize_critical_text(text: str) -> str:
    """Normalize semantically equivalent numeric forms for critical matching."""
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\b([ap])\s*\.?\s*m\.?(?!\w)", r"\1m", text)
    text = re.sub(
        rf"\b(hundred|thousand|million)\s+and\s+(?={_NUMBER_WORD_PATTERN}\b)",
        r"\1 ",
        text,
    )
    text = re.sub(r"\$(\d+)(?:\.(\d{1,2}))?", _replace_money, text)
    text = re.sub(
        r"\b(\d{1,2})[.:](\d{2})\s*(am|pm)\b",
        _replace_numeric_meridiem_time,
        text,
    )
    text = re.sub(
        r"\b(\d{1,2})\s*(am|pm)\b",
        _replace_numeric_hour_meridiem,
        text,
    )
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _replace_numeric_time, text)
    text = re.sub(
        rf"\b(?P<hour>{_HOUR_WORD_PATTERN})[\s-]+"
        rf"(?P<minute>{_MINUTE_WORD_PATTERN})\s*"
        r"(?P<meridiem>am|pm)\b",
        _replace_spoken_time,
        text,
    )
    text = re.sub(
        rf"\b(?P<century>nineteen|twenty)[\s-]+"
        rf"(?P<suffix>{_YEAR_SUFFIX_PATTERN})\b",
        _replace_spoken_year,
        text,
    )
    text = re.sub(
        r"\b(version|release)\s+(\d+(?:\.\d+)+)\b",
        _replace_contextual_version,
        text,
    )
    text = re.sub(
        r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
        _replace_ipv4,
        text,
    )
    text = re.sub(r"\b\d+(?:\.\d+)+\b", _replace_dotted_version, text)
    text = re.sub(
        rf"\b{_DIGIT_WORD_PATTERN}(?:[\s-]+{_DIGIT_WORD_PATTERN})+\b",
        _replace_spoken_digits,
        text,
    )
    text = re.sub(
        rf"\b{_NUMBER_WORD_PATTERN}(?:[\s-]+{_NUMBER_WORD_PATTERN})*\b",
        _replace_number_words,
        text,
    )
    text = re.sub(
        r"\b(dollars?)\s+and\s+(?=\d+\s+cents?\b)",
        r"\1 ",
        text,
    )
    return normalize_text(text)


def _normalize_identifier_text(text: str) -> str:
    """Normalize spoken identifier separators without erasing literal ones."""
    text = re.sub(
        r"(?<=[a-z0-9])\s+underscore\s+(?=[a-z0-9])",
        "_",
        text.lower(),
    )
    return _normalize_critical_text(text)


def _contains_normalized_term(norm_term: str, norm_pred: str) -> bool:
    if not norm_term:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(norm_term)}(?!\w)", norm_pred))


def _looks_like_identifier_term(term: str, norm_term: str) -> bool:
    lowered = term.lower()
    if "_" in term or " underscore " in lowered:
        return True
    return bool(re.fullmatch(r"[a-z]+\s*\d+", norm_term))


def _contains_identifier_term(norm_term: str, norm_pred: str) -> bool:
    if _contains_normalized_term(norm_term, norm_pred):
        return True

    compact_match = re.fullmatch(r"([a-z]+)\s*(\d+)", norm_term)
    if not compact_match:
        return False

    prefix, digits = compact_match.groups()
    return bool(
        re.search(
            rf"(?<!\w){re.escape(prefix)}\s*{re.escape(digits)}(?!\w)",
            norm_pred,
        )
    )


def calculate_wer(references: List[str], predictions: List[str]) -> float:
    """Calculate Word Error Rate (WER) using jiwer."""
    if not references:
        return 0.0
    # Clean empty strings or replace with placeholder space
    refs = [r if r.strip() else " " for r in references]
    preds = [p if p.strip() else " " for p in predictions]
    try:
        wer = jiwer.wer(refs, preds)
        return float(wer)
    except Exception:
        return float('nan')


def _word_error_counts(reference: str, prediction: str) -> Dict[str, Any]:
    """Return WER and edit counts for one reference/prediction pair."""
    ref = reference if reference.strip() else " "
    pred = prediction if prediction.strip() else " "
    result = jiwer.process_words(ref, pred)
    return {
        "wer": float(result.wer) * 100.0,
        "substitutions": result.substitutions,
        "insertions": result.insertions,
        "deletions": result.deletions,
    }


def evaluate_critical_info(reference_item: Dict[str, Any], prediction_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate critical information recall for a single item.
    Returns details on matched, missed, and total critical terms, plus specific failure details.
    """
    critical_terms = reference_item.get("critical", [])
    critical_types = reference_item.get("critical_types", {})
    if (
        not isinstance(critical_types, dict)
        or not all(
            isinstance(term, str)
            and isinstance(entity_type, str)
            and term in critical_terms
            and entity_type in ENTITY_TYPES
            for term, entity_type in critical_types.items()
        )
    ):
        raise ValueError("Invalid critical_types entity mapping")
    pred_text = prediction_item.get("text", "")
    norm_pred = _normalize_critical_text(pred_text)
    identifier_norm_pred = None
    
    matched = []
    missed = []
    failures = []
    strict_matched = []
    strict_missed = []
    
    for term in critical_terms:
        if strict_contains(term, pred_text):
            strict_matched.append(term)
        else:
            strict_missed.append(term)

        entity_type = critical_types.get(term)
        if entity_type is not None:
            is_match = canonical_contains(term, pred_text, entity_type)
        else:
            # Preserve the legacy contract for datasets that have not yet added
            # explicit entity types. Typed datasets never use this heuristic.
            norm_term = _normalize_critical_text(term)
            is_match = _contains_normalized_term(norm_term, norm_pred)

            if not is_match and _looks_like_identifier_term(term, norm_term):
                if identifier_norm_pred is None:
                    identifier_norm_pred = _normalize_identifier_text(pred_text)
                identifier_norm_term = _normalize_identifier_text(term)
                is_match = _contains_identifier_term(identifier_norm_term, identifier_norm_pred)

        if is_match:
            matched.append(term)
        else:
            missed.append(term)
            failures.append({
                "expected": term,
                "predicted_text": pred_text
            })
            
    return {
        "total": len(critical_terms),
        "matched": matched,
        "missed": missed,
        "failures": failures,
        "strict_matched": strict_matched,
        "strict_missed": strict_missed,
        "canonical_matched": matched,
        "canonical_missed": missed,
    }


def evaluate_non_speech_info(reference_item: Dict[str, Any], prediction_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate non-speech information (sound events e.g. [laughter], [door closes]).
    """
    sounds = reference_item.get("sounds", [])
    pred_text = prediction_item.get("text", "")
    norm_pred = normalize_text(pred_text)
    pred_sounds = prediction_item.get("sounds", [])
    norm_pred_sounds = {
        normalize_text(sound)
        for sound in pred_sounds
        if isinstance(sound, str)
    }
    
    matched = []
    missed = []
    
    for sound in sounds:
        norm_sound = normalize_text(sound)
        text_match = bool(
            norm_sound
            and re.search(rf"(?<!\w){re.escape(norm_sound)}(?!\w)", norm_pred)
        )
        if norm_sound and (norm_sound in norm_pred_sounds or text_match):
            matched.append(sound)
        else:
            missed.append(sound)
            
    return {
        "total": len(sounds),
        "matched": matched,
        "missed": missed
    }


def evaluate_speaker_attribution(reference_item: Dict[str, Any], prediction_item: Dict[str, Any]) -> Optional[bool]:
    """
    Evaluate speaker attribution if speaker tag is present in reference.
    Returns None if the reference omits a speaker, False if the prediction omits or mismatches it, and True if correct.
    """
    ref_speaker = reference_item.get("speaker")
    if ref_speaker is None:
        return None
    pred_speaker = prediction_item.get("speaker")
    if pred_speaker is None:
        return False
    return normalize_text(str(ref_speaker)) == normalize_text(str(pred_speaker))


def _is_sound_only_reference(reference: Dict[str, Any]) -> bool:
    sounds = reference.get("sounds")
    return (
        not reference.get("text", "").strip()
        and isinstance(sounds, list)
        and bool(sounds)
        and all(isinstance(sound, str) and sound.strip() for sound in sounds)
    )


def evaluate_dataset(aligned_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run full metric evaluation over aligned dataset.
    """
    lexical_pairs = []
    for item in aligned_data:
        if _is_sound_only_reference(item["reference"]):
            continue
        reference_text = item["reference"].get("text", "")
        lexical_pairs.append((reference_text, item["prediction"].get("text", "")))
    if lexical_pairs:
        conventional_asr = evaluate_conventional_asr(
            [pair[0] for pair in lexical_pairs],
            [pair[1] for pair in lexical_pairs],
        )
    else:
        conventional_asr = {
            "normalization_policy": NORMALIZATION_POLICY,
            "orthographic_wer": None,
            "normalized_wer": None,
            "orthographic_cer": None,
            "normalized_cer": None,
            "orthographic_substitutions": 0,
            "orthographic_insertions": 0,
            "orthographic_deletions": 0,
            "normalized_substitutions": 0,
            "normalized_insertions": 0,
            "normalized_deletions": 0,
        }
    word_errors_by_sample = []
    substitutions = conventional_asr["orthographic_substitutions"]
    insertions = conventional_asr["orthographic_insertions"]
    deletions = conventional_asr["orthographic_deletions"]
    
    total_critical = 0
    matched_critical = 0
    strict_matched_critical = 0
    all_critical_failures = []
    all_strict_critical_failures = []
    
    total_sounds = 0
    matched_sounds = 0
    all_sound_failures = []
    
    speaker_evals = []
    latencies = []
    
    for item in aligned_data:
        ref = item["reference"]
        pred = item["prediction"]
        sample_id = ref.get("id")
        if sample_id is None:
            sample_id = pred.get("id")
        if sample_id is None:
            sample_id = "unknown"
        
        # Critical
        crit_res = evaluate_critical_info(ref, pred)
        total_critical += crit_res["total"]
        matched_critical += len(crit_res["matched"])
        strict_matched_critical += len(crit_res["strict_matched"])
        for fail in crit_res["failures"]:
            all_critical_failures.append({
                "id": sample_id,
                "expected": fail["expected"],
                "predicted_text": fail["predicted_text"]
            })
        for expected in crit_res["strict_missed"]:
            all_strict_critical_failures.append({
                "id": sample_id,
                "expected": expected,
                "predicted_text": pred.get("text", ""),
            })

        if not _is_sound_only_reference(ref):
            sample_errors = _word_error_counts(
                ref.get("text", ""), pred.get("text", "")
            )
            word_errors_by_sample.append({"id": sample_id, **sample_errors})
            
        # Non-speech
        sound_res = evaluate_non_speech_info(ref, pred)
        total_sounds += sound_res["total"]
        matched_sounds += len(sound_res["matched"])
        for sound in sound_res["missed"]:
            all_sound_failures.append({
                "id": sample_id,
                "expected": sound,
                "predicted_text": pred.get("text", "")
            })
        
        # Speaker
        spk_res = evaluate_speaker_attribution(ref, pred)
        if spk_res is not None:
            speaker_evals.append(spk_res)
            
        # Latency
        if "latency_ms" in pred and pred["latency_ms"] is not None:
            try:
                latency_ms = float(pred["latency_ms"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid latency_ms for sample {sample_id}: expected a finite non-negative number"
                ) from exc
            if not math.isfinite(latency_ms) or latency_ms < 0:
                raise ValueError(
                    f"Invalid latency_ms for sample {sample_id}: expected a finite non-negative number"
                )
            latencies.append(latency_ms)
            
    crit_recall = (matched_critical / total_critical * 100.0) if total_critical > 0 else 100.0
    strict_crit_recall = (strict_matched_critical / total_critical * 100.0) if total_critical > 0 else 100.0
    sound_recall = (matched_sounds / total_sounds * 100.0) if total_sounds > 0 else None
    speaker_acc = (sum(speaker_evals) / len(speaker_evals) * 100.0) if speaker_evals else None
    
    median_latency = None
    if latencies:
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        if n % 2 == 1:
            median_latency = sorted_lat[n // 2]
        else:
            median_latency = (sorted_lat[n // 2 - 1] + sorted_lat[n // 2]) / 2.0
            
    return {
        "samples": len(aligned_data),
        "wer": conventional_asr["orthographic_wer"],
        "cer": conventional_asr["orthographic_cer"],
        **conventional_asr,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "word_errors_by_sample": word_errors_by_sample,
        "critical_recall": crit_recall,
        "canonical_critical_recall": crit_recall,
        "strict_critical_recall": strict_crit_recall,
        "total_critical": total_critical,
        "matched_critical": matched_critical,
        "canonical_matched_critical": matched_critical,
        "strict_matched_critical": strict_matched_critical,
        "critical_failures": all_critical_failures,
        "strict_critical_failures": all_strict_critical_failures,
        "non_speech_recall": sound_recall,
        "total_sounds": total_sounds,
        "matched_sounds": matched_sounds,
        "non_speech_failures": all_sound_failures,
        "speaker_accuracy": speaker_acc,
        "median_latency_ms": median_latency
    }
