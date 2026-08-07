import json
import re
from typing import Dict, List, Any, Optional

def normalize_text(text: str) -> str:
    """Normalize text for basic string matching (lowercase, clean spaces & punctuation)."""
    text = text.lower()
    # remove punctuation except brackets used in sound events
    text = re.sub(r'[^\w\s\[\]]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Parse a JSONL file into a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid JSON record at line {line_num} in {filepath}: expected object")
                if "text" in item and not isinstance(item["text"], str):
                    raise ValueError(f"Invalid JSON record at line {line_num} in {filepath}: text must be a string")
                for field in ("critical", "sounds"):
                    if field in item and not isinstance(item[field], list):
                        raise ValueError(
                            f"Invalid JSON record at line {line_num} in {filepath}: {field} must be a list"
                        )
                data.append(item)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {filepath}: {e}")
    return data

def _validate_unique_ids(items: List[Dict[str, Any]], label: str) -> None:
    seen = set()
    for item in items:
        item_id = item.get("id")
        if item_id is None:
            continue
        if item_id in seen:
            raise ValueError(f"duplicate {label} id: {item_id}")
        seen.add(item_id)

def align_records(references: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Align reference and prediction items by ID or position."""
    _validate_unique_ids(references, "reference")
    _validate_unique_ids(predictions, "prediction")

    use_id_alignment = any(item.get("id") is not None for item in references + predictions)
    pred_map = {p.get("id"): p for p in predictions if p.get("id") is not None}
    
    aligned = []
    for idx, ref in enumerate(references):
        ref_id = ref.get("id", f"sample-{idx+1}")
        if use_id_alignment:
            pred = pred_map.get(ref_id)
        else:
            pred = predictions[idx] if idx < len(predictions) else None
        if pred is None:
            pred = {"id": ref_id, "text": ""}
        aligned.append({"reference": ref, "prediction": pred})
    return aligned
