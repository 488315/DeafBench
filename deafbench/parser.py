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
                data.append(item)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {filepath}: {e}")
    return data

def align_records(references: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Align reference and prediction items by ID or position."""
    pred_map = {p.get("id"): p for p in predictions if p.get("id") is not None}
    
    aligned = []
    for idx, ref in enumerate(references):
        ref_id = ref.get("id", f"sample-{idx+1}")
        pred = pred_map.get(ref_id)
        if pred is None and idx < len(predictions):
            pred = predictions[idx]
        if pred is None:
            pred = {"id": ref_id, "text": ""}
        aligned.append({"reference": ref, "prediction": pred})
    return aligned
