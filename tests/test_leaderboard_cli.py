import json
import subprocess

from deafbench.cli import main


def _fake_checkout(tmp_path):
    checkout = tmp_path / "official"
    normalizer = checkout / "normalizer"
    normalizer.mkdir(parents=True)
    for name in ("__init__.py", "normalizer.py"):
        (normalizer / name).write_text("", encoding="utf-8")
    (normalizer / "data_utils.py").write_text(
        "normalizer = lambda text: text.lower()\n",
        encoding="utf-8",
    )
    (normalizer / "eval_utils.py").write_text(
        "def score_results(directory, model_id=None, **kwargs):\n"
        "    return {model_id: 4.25}, "
        "{f'{model_id} | fake_test': {'wer': 4.25}}\n",
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
        "composite_wer": {"owner/model": 4.25},
        "datasets": {"owner/model | fake_test": {"wer": 4.25}},
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
