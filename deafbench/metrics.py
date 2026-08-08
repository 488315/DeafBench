import math
import re
from typing import List, Dict, Any, Optional
import jiwer
from .parser import normalize_text

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
        normalized += f" {int(cents.ljust(2, '0'))} cents"
    return normalized


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


def _normalize_critical_text(text: str) -> str:
    """Normalize semantically equivalent numeric forms for critical matching."""
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\b([ap])\s*\.?\s*m\.?(?!\w)", r"\1m", text)
    text = re.sub(r"\$(\d+)(?:\.(\d{1,2}))?", _replace_money, text)
    text = re.sub(
        r"\b(\d{1,2})[.:](\d{2})\s*(am|pm)\b",
        r"\1 colon \2 \3",
        text,
    )
    text = re.sub(r"\b(\d{1,2})\s*(am|pm)\b", r"\1 \2", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1 colon \2", text)
    text = re.sub(
        r"(?<=\bversion\s)\d+(?:\.\d+)+\b",
        _replace_dotted_version,
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
    return normalize_text(text)


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


def evaluate_critical_info(reference_item: Dict[str, Any], prediction_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate critical information recall for a single item.
    Returns details on matched, missed, and total critical terms, plus specific failure details.
    """
    critical_terms = reference_item.get("critical", [])
    pred_text = prediction_item.get("text", "")
    norm_pred = _normalize_critical_text(pred_text)
    
    matched = []
    missed = []
    failures = []
    
    for term in critical_terms:
        norm_term = _normalize_critical_text(term)
        if norm_term and re.search(rf"(?<!\w){re.escape(norm_term)}(?!\w)", norm_pred):
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
        "failures": failures
    }


def evaluate_non_speech_info(reference_item: Dict[str, Any], prediction_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate non-speech information (sound events e.g. [laughter], [door closes]).
    """
    sounds = reference_item.get("sounds", [])
    pred_text = prediction_item.get("text", "")
    norm_pred = normalize_text(pred_text)
    
    matched = []
    missed = []
    
    for sound in sounds:
        norm_sound = normalize_text(sound)
        if norm_sound and re.search(rf"(?<!\w){re.escape(norm_sound)}(?!\w)", norm_pred):
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


def evaluate_dataset(aligned_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run full metric evaluation over aligned dataset.
    """
    ref_texts = [item["reference"].get("text", "") for item in aligned_data]
    pred_texts = [item["prediction"].get("text", "") for item in aligned_data]
    
    wer = calculate_wer(ref_texts, pred_texts)
    
    total_critical = 0
    matched_critical = 0
    all_critical_failures = []
    
    total_sounds = 0
    matched_sounds = 0
    
    speaker_evals = []
    latencies = []
    
    for item in aligned_data:
        ref = item["reference"]
        pred = item["prediction"]
        sample_id = ref.get("id", "unknown")
        
        # Critical
        crit_res = evaluate_critical_info(ref, pred)
        total_critical += crit_res["total"]
        matched_critical += len(crit_res["matched"])
        for fail in crit_res["failures"]:
            all_critical_failures.append({
                "id": sample_id,
                "expected": fail["expected"],
                "predicted_text": fail["predicted_text"]
            })
            
        # Non-speech
        sound_res = evaluate_non_speech_info(ref, pred)
        total_sounds += sound_res["total"]
        matched_sounds += len(sound_res["matched"])
        
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
    sound_recall = (matched_sounds / total_sounds * 100.0) if total_sounds > 0 else 100.0
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
        "wer": wer * 100.0,
        "critical_recall": crit_recall,
        "total_critical": total_critical,
        "matched_critical": matched_critical,
        "critical_failures": all_critical_failures,
        "non_speech_recall": sound_recall,
        "total_sounds": total_sounds,
        "matched_sounds": matched_sounds,
        "speaker_accuracy": speaker_acc,
        "median_latency_ms": median_latency
    }
