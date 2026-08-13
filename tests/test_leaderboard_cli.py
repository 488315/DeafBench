import json

from _official_checkout import commit_fake_official_evaluator
from deafbench.cli import main
from deafbench.leaderboard._official_worker import _PUBLIC_EXPECTED_ROWS


_CLI_SCORE_STUB = (
    "def score_results(directory, model_id=None, **kwargs):\n"
    "    return {model_id: 4.25}, "
    "{f'{model_id} | fake_test': {'wer': 4.25}}\n"
)


def _fake_checkout(tmp_path):
    return commit_fake_official_evaluator(tmp_path, score_stub=_CLI_SCORE_STUB)


def test_leaderboard_score_writes_json(tmp_path, monkeypatch):
    checkout, revision = _fake_checkout(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    output = tmp_path / "score.json"
    monkeypatch.setattr(
        "deafbench.leaderboard.official.OPEN_ASR_EVALUATOR_REVISION",
        revision,
    )

    exit_code = main(
        [
            "leaderboard",
            "score",
            str(results),
            "--official-repo",
            str(checkout),
            "--model-id",
            "owner/model",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "partial_mean_wer": {"owner/model": 4.25},
        "datasets": {"owner/model | fake_test": {"wer": 4.25}},
        "upstream_wer_sum": {"owner/model": 4.25},
        "evaluation": {
            "status": "partial",
            "completed_sets": 0,
            "expected_sets": 7,
            "observed_rows": {},
            "expected_rows": _PUBLIC_EXPECTED_ROWS,
        },
    }


def test_leaderboard_score_reports_invalid_checkout(tmp_path, capsys):
    results = tmp_path / "results"
    results.mkdir()

    exit_code = main(
        [
            "leaderboard",
            "score",
            str(results),
            "--official-repo",
            str(tmp_path / "missing"),
            "--model-id",
            "owner/model",
        ]
    )

    assert exit_code == 1
    assert "official evaluator checkout does not exist" in capsys.readouterr().err


def test_leaderboard_analyze_writes_ranked_errors(tmp_path, monkeypatch):
    checkout, revision = _fake_checkout(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    dataset_id = "hf-audio-open-asr-leaderboard_librispeech_test.clean"
    manifest = results / f"MODEL_owner-model_DATASET_{dataset_id}.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_filepath": "sample_0",
                "duration": 1.0,
                "text": "Hello world",
                "pred_text": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        "deafbench.leaderboard.official.OPEN_ASR_EVALUATOR_REVISION",
        revision,
    )

    exit_code = main(
        [
            "leaderboard",
            "analyze",
            str(results),
            "--official-repo",
            str(checkout),
            "--model-id",
            "owner/model",
            "--limit",
            "1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["datasets"][dataset_id]["errors"] == {
        "del": 1,
        "ins": 0,
        "sub": 0,
    }
    assert result["datasets"][dataset_id]["top_errors"][0]["row"] == 0
