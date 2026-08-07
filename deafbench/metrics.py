import math
import re
from typing import List, Dict, Any, Optional
import jiwer
from .parser import normalize_text

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
    norm_pred = normalize_text(pred_text)
    
    matched = []
    missed = []
    failures = []
    
    for term in critical_terms:
        norm_term = normalize_text(term)
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
        if norm_sound and norm_sound in norm_pred:
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
