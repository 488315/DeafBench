import json
from pathlib import Path


_ROOT = Path(__file__).parents[1]
_RESULTS = _ROOT / "experiments" / "open-asr" / "results"
_MANIFEST = (
    "MODEL_soundsgoodai-Zipformer-cr-ctc-transducer-XL-290M_"
    "DATASET_hf-audio-open-asr-leaderboard_librispeech_test.clean.jsonl"
)
_STABLE_FIELDS = ("audio_filepath", "duration", "text", "pred_text")


def _records(path: Path, limit: int) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for _, line in zip(range(limit), stream)]


def test_zipformer_predictions_repeat_across_independent_batch_sizes():
    diagnostic = _records(_RESULTS / _MANIFEST, 2)
    full_run = _records(_RESULTS / "full-public" / _MANIFEST, 2)

    assert len(diagnostic) == len(full_run) == 2
    assert [row["time"] for row in diagnostic] != [row["time"] for row in full_run]
    assert [
        tuple(row[field] for field in _STABLE_FIELDS) for row in diagnostic
    ] == [tuple(row[field] for field in _STABLE_FIELDS) for row in full_run]


def test_six_set_error_analysis_reproduces_official_aggregates():
    score = json.loads(
        (_RESULTS / "zipformer-public-6set-score.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (_RESULTS / "zipformer-public-6set-errors.json").read_text(encoding="utf-8")
    )

    assert score["evaluation"]["completed_sets"] == 6
    for score_key, metrics in score["datasets"].items():
        dataset_id = score_key.split(" | ", 1)[1]
        assert analysis["datasets"][dataset_id]["errors"] == {
            key: metrics[key] for key in ("del", "ins", "sub")
        }
