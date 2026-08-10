"""Isolated import worker for the external official evaluator checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_RESULT_MARKER = "DEAFBENCH_OFFICIAL_RESULT="
_PUBLIC_EXPECTED_ROWS = {
    # The pinned loader removes 90 references that normalize to empty/ignore.
    "hf-audio-open-asr-leaderboard_ami_cleaned_test": 7715,
    # The pinned loader removes four references that normalize to empty/ignore.
    "hf-audio-open-asr-leaderboard_earnings22_test": 2737,
    # The pinned loader removes 11 references that normalize to empty/ignore.
    "hf-audio-open-asr-leaderboard_gigaspeech_cleaned_test": 18757,
    "hf-audio-open-asr-leaderboard_librispeech_test.clean": 2620,
    "hf-audio-open-asr-leaderboard_librispeech_test.other": 2939,
    "hf-audio-open-asr-leaderboard_spgispeech_test": 39341,
    "hf-audio-open-asr-leaderboard_voxpopuli_cleaned_aa_test": 628,
}


def _mean_wer_by_model(datasets: dict) -> dict[str, float]:
    """Match the mean that upstream prints but does not return."""
    wers: dict[str, list[float]] = {}
    for result_key, metrics in datasets.items():
        model_id = result_key.split("|", 1)[0].strip()
        wers.setdefault(model_id, []).append(metrics["wer"])
    return {
        model_id: round(sum(values) / len(values), 2)
        for model_id, values in wers.items()
    }


def _evaluation_status(observed_rows: dict[str, int]) -> dict:
    complete = observed_rows == _PUBLIC_EXPECTED_ROWS
    return {
        "status": "complete" if complete else "partial",
        "completed_sets": sum(
            observed_rows.get(dataset_id) == rows
            for dataset_id, rows in _PUBLIC_EXPECTED_ROWS.items()
        ),
        "expected_sets": len(_PUBLIC_EXPECTED_ROWS),
        "observed_rows": observed_rows,
        "expected_rows": _PUBLIC_EXPECTED_ROWS,
    }


def _public_manifest_rows(results_dir: str, model_id: str) -> dict[str, int]:
    observed = {}
    expected_prefix = "MODEL_" + model_id.replace("/", "-") + "_DATASET_"
    for result_file in Path(results_dir).rglob("*.jsonl"):
        if not result_file.stem.startswith(expected_prefix):
            continue
        dataset_id = result_file.stem.removeprefix(expected_prefix)
        if dataset_id in _PUBLIC_EXPECTED_ROWS:
            if dataset_id in observed:
                raise ValueError(f"duplicate public manifest: {dataset_id}")
            with result_file.open(encoding="utf-8") as handle:
                observed[dataset_id] = sum(1 for line in handle if line.strip())
    return observed


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

    upstream_sum, datasets = score_results(
        payload["results_dir"],
        payload["model_id"],
        families=["public"],
    )
    observed_rows = _public_manifest_rows(
        payload["results_dir"],
        payload["model_id"],
    )
    evaluation = _evaluation_status(observed_rows)
    mean_key = "composite_wer" if evaluation["status"] == "complete" else "partial_mean_wer"
    return {
        mean_key: _mean_wer_by_model(datasets),
        "upstream_wer_sum": dict(upstream_sum),
        "datasets": datasets,
        "evaluation": evaluation,
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
