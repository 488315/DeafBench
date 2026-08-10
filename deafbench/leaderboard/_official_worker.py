"""Isolated import worker for the external official evaluator checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_RESULT_MARKER = "DEAFBENCH_OFFICIAL_RESULT="


def _read_payload() -> dict:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("worker payload must be an object")
    return payload


def _normalize(payload: dict) -> dict:
    from normalizer.data_utils import normalizer

    texts = payload.get("texts")
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("texts must be a list of strings")
    return {"normalized": [normalizer(text) for text in texts]}


def _score(payload: dict) -> dict:
    from normalizer.eval_utils import score_results

    composite, datasets = score_results(
        payload["results_dir"],
        payload["model_id"],
        families=["public"],
    )
    return {
        "composite_wer": dict(composite),
        "datasets": datasets,
    }


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("action", choices=("normalize", "score"))
    parsed = parser.parse_args(args)

    sys.path.insert(0, str(parsed.checkout.resolve()))
    payload = _read_payload()
    result = _normalize(payload) if parsed.action == "normalize" else _score(payload)
    print(_RESULT_MARKER + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
