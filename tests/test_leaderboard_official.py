import json
import subprocess

import pytest

from _official_checkout import commit_fake_official_evaluator
from deafbench.leaderboard._official_worker import (
    _PUBLIC_EXPECTED_ROWS,
    _evaluation_status,
    _mean_wer_by_model,
    _public_manifest_rows,
)
from deafbench.leaderboard import _official_worker
from deafbench.leaderboard.official import (
    OfficialEvaluator,
    OfficialEvaluatorError,
)


_OFFICIAL_NORMALIZER_STUB = (
    "normalizer = lambda text: text.lower().replace('doctor', 'dr')\n"
)
_OFFICIAL_SCORE_STUB = (
    "def score_results(directory, model_id=None, **kwargs):\n"
    "    return {model_id: 12.5}, "
    "{f'{model_id} | fake_test': {'wer': 12.5, 'rtfx': 3.0}}\n"
)


def _commit_fake_evaluator(tmp_path):
    return commit_fake_official_evaluator(
        tmp_path,
        score_stub=_OFFICIAL_SCORE_STUB,
        normalizer_stub=_OFFICIAL_NORMALIZER_STUB,
    )


def test_evaluator_rejects_a_checkout_at_the_wrong_revision(tmp_path):
    checkout, revision = _commit_fake_evaluator(tmp_path)
    evaluator = OfficialEvaluator(checkout, expected_revision="0" * 40)

    with pytest.raises(OfficialEvaluatorError, match="revision"):
        evaluator.validate()

    assert revision != "0" * 40


def test_worker_validates_checkout_before_reading_payload(tmp_path, monkeypatch):
    checkout, _ = _commit_fake_evaluator(tmp_path)
    monkeypatch.setattr(
        _official_worker,
        "_read_payload",
        lambda: pytest.fail("worker read untrusted input before validation"),
    )

    with pytest.raises(OfficialEvaluatorError, match="revision"):
        _official_worker.main(
            [
                "--checkout",
                str(checkout),
                "--expected-revision",
                "0" * 40,
                "normalize",
            ]
        )


def test_evaluator_rejects_modified_tracked_source(tmp_path):
    checkout, revision = _commit_fake_evaluator(tmp_path)
    evaluator = OfficialEvaluator(checkout, expected_revision=revision)
    (checkout / "normalizer" / "data_utils.py").write_text(
        "normalizer = lambda text: 'modified'\n",
        encoding="utf-8",
    )

    with pytest.raises(OfficialEvaluatorError, match="modified"):
        evaluator.validate()


def test_evaluator_delegates_normalization_and_scoring_to_pinned_source(tmp_path):
    checkout, revision = _commit_fake_evaluator(tmp_path)
    evaluator = OfficialEvaluator(checkout, expected_revision=revision)

    assert evaluator.normalize(["Doctor Ada", "HELLO"]) == ["dr ada", "hello"]

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    score = evaluator.score(results_dir, "owner/model")

    assert score == {
        "partial_mean_wer": {"owner/model": 12.5},
        "upstream_wer_sum": {"owner/model": 12.5},
        "datasets": {
            "owner/model | fake_test": {
                "rtfx": 3.0,
                "wer": 12.5,
            }
        },
        "evaluation": {
            "status": "partial",
            "completed_sets": 0,
            "expected_sets": 7,
            "observed_rows": {},
            "expected_rows": _PUBLIC_EXPECTED_ROWS,
        },
    }


def test_evaluator_analyzes_rows_with_pinned_normalization(tmp_path):
    checkout, revision = _commit_fake_evaluator(tmp_path)
    evaluator = OfficialEvaluator(checkout, expected_revision=revision)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    dataset_id = "hf-audio-open-asr-leaderboard_librispeech_test.clean"
    manifest = results_dir / f"MODEL_owner-model_DATASET_{dataset_id}.jsonl"
    records = [
        {
            "audio_filepath": "sample_0",
            "duration": 1.0,
            "time": 0.1,
            "text": "Doctor Ada",
            "pred_text": "doctor ada",
        },
        {
            "audio_filepath": "sample_1",
            "duration": 2.0,
            "time": 0.2,
            "text": "Hello world",
            "pred_text": "hello",
        },
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    analysis = evaluator.analyze(results_dir, "owner/model", limit=1)

    assert analysis == {
        "datasets": {
            dataset_id: {
                "analyzed_rows": 2,
                "reference_words": 4,
                "errors": {"del": 1, "ins": 0, "sub": 0},
                "top_errors": [
                    {
                        "row": 1,
                        "audio_filepath": "sample_1",
                        "duration": 2.0,
                        "reference": "hello world",
                        "prediction": "hello",
                        "reference_words": 2,
                        "errors": {"del": 1, "ins": 0, "sub": 0},
                        "wer": 50.0,
                    }
                ],
            }
        }
    }


def test_evaluator_rejects_analysis_without_public_manifests(tmp_path):
    checkout, revision = _commit_fake_evaluator(tmp_path)
    evaluator = OfficialEvaluator(checkout, expected_revision=revision)
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    with pytest.raises(OfficialEvaluatorError, match="public manifests"):
        evaluator.analyze(results_dir, "owner/model")


@pytest.mark.parametrize("limit", (0, 1.5, True))
def test_evaluator_rejects_invalid_analysis_limit(tmp_path, limit):
    checkout, revision = _commit_fake_evaluator(tmp_path)
    evaluator = OfficialEvaluator(checkout, expected_revision=revision)
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    with pytest.raises(OfficialEvaluatorError, match="limit"):
        evaluator.analyze(results_dir, "owner/model", limit=limit)


def test_evaluator_rejects_missing_results_directory(tmp_path):
    checkout, revision = _commit_fake_evaluator(tmp_path)
    evaluator = OfficialEvaluator(checkout, expected_revision=revision)

    with pytest.raises(OfficialEvaluatorError, match="results directory"):
        evaluator.score(tmp_path / "missing", "owner/model")


def test_evaluator_rejects_worker_output_without_result_marker(tmp_path, monkeypatch):
    checkout, revision = _commit_fake_evaluator(tmp_path)
    evaluator = OfficialEvaluator(checkout, expected_revision=revision)
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    completed = subprocess.CompletedProcess([], 0, stdout="noise\n", stderr="")
    monkeypatch.setattr(evaluator, "validate", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(OfficialEvaluatorError, match="result marker"):
        evaluator.score(results_dir, "owner/model")


def test_evaluator_times_out_at_the_configured_deadline(tmp_path, monkeypatch):
    checkout, revision = _commit_fake_evaluator(tmp_path)
    evaluator = OfficialEvaluator(
        checkout,
        expected_revision=revision,
        timeout_seconds=0.25,
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(evaluator, "validate", lambda: None)

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 0.25
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(OfficialEvaluatorError, match="timed out"):
        evaluator.score(results_dir, "owner/model")


@pytest.mark.parametrize("timeout_seconds", (0, -1, float("inf"), True))
def test_evaluator_rejects_nonfinite_or_nonpositive_timeout(
    tmp_path,
    timeout_seconds,
):
    checkout, revision = _commit_fake_evaluator(tmp_path)

    with pytest.raises(OfficialEvaluatorError, match="positive finite"):
        OfficialEvaluator(
            checkout,
            expected_revision=revision,
            timeout_seconds=timeout_seconds,
        )


def test_worker_returns_the_mean_that_upstream_only_prints():
    datasets = {
        "owner/model | first": {"wer": 1.31},
        "owner/model | second": {"wer": 4.31},
        "other/model | first": {"wer": 2.0},
    }

    assert _mean_wer_by_model(datasets) == {
        "owner/model": 2.81,
        "other/model": 2.0,
    }


def test_worker_requires_all_exact_public_row_counts_for_composite():
    assert _PUBLIC_EXPECTED_ROWS[
        "hf-audio-open-asr-leaderboard_ami_cleaned_test"
    ] == 7715
    assert _PUBLIC_EXPECTED_ROWS[
        "hf-audio-open-asr-leaderboard_earnings22_test"
    ] == 2737
    assert _PUBLIC_EXPECTED_ROWS[
        "hf-audio-open-asr-leaderboard_gigaspeech_cleaned_test"
    ] == 18757
    assert _evaluation_status(dict(_PUBLIC_EXPECTED_ROWS))["status"] == "complete"

    incomplete = dict(_PUBLIC_EXPECTED_ROWS)
    incomplete["hf-audio-open-asr-leaderboard_librispeech_test.clean"] -= 1
    status = _evaluation_status(incomplete)

    assert status["status"] == "partial"
    assert status["completed_sets"] == 6


def test_worker_scans_only_requested_model_and_rejects_duplicates(tmp_path):
    dataset_id = "hf-audio-open-asr-leaderboard_librispeech_test.clean"
    requested = tmp_path / f"MODEL_owner-model_DATASET_{dataset_id}.jsonl"
    wrong_model = tmp_path / f"MODEL_other-model_DATASET_{dataset_id}.jsonl"
    requested.write_text("{}\n{}\n", encoding="utf-8")
    wrong_model.write_text("{}\n{}\n{}\n", encoding="utf-8")

    assert _public_manifest_rows(tmp_path, "owner/model") == {dataset_id: 2}

    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    (duplicate_dir / requested.name).write_text("{}\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate public manifest"):
        _public_manifest_rows(tmp_path, "owner/model")
