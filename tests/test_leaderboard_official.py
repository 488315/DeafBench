import json
import subprocess

import pytest

from deafbench.leaderboard._official_worker import _mean_wer_by_model
from deafbench.leaderboard.official import (
    OfficialEvaluator,
    OfficialEvaluatorError,
)


def _commit_fake_evaluator(tmp_path):
    checkout = tmp_path / "official"
    normalizer = checkout / "normalizer"
    normalizer.mkdir(parents=True)
    (normalizer / "__init__.py").write_text("", encoding="utf-8")
    (normalizer / "normalizer.py").write_text("", encoding="utf-8")
    (normalizer / "data_utils.py").write_text(
        "normalizer = lambda text: text.lower().replace('doctor', 'dr')\n",
        encoding="utf-8",
    )
    (normalizer / "eval_utils.py").write_text(
        "def score_results(directory, model_id=None, **kwargs):\n"
        "    return {model_id: 12.5}, "
        "{f'{model_id} | fake_test': {'wer': 12.5, 'rtfx': 3.0}}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=DeafBench Tests",
            "-c",
            "user.email=tests@deafbench.invalid",
            "commit",
            "-qm",
            "test evaluator",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return checkout, revision


def test_evaluator_rejects_a_checkout_at_the_wrong_revision(tmp_path):
    checkout, revision = _commit_fake_evaluator(tmp_path)
    evaluator = OfficialEvaluator(checkout, expected_revision="0" * 40)

    with pytest.raises(OfficialEvaluatorError, match="revision"):
        evaluator.validate()

    assert revision != "0" * 40


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
        "composite_wer": {"owner/model": 12.5},
        "upstream_wer_sum": {"owner/model": 12.5},
        "datasets": {
            "owner/model | fake_test": {
                "rtfx": 3.0,
                "wer": 12.5,
            }
        },
    }


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
