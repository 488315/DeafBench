import builtins
import importlib
import sys
import types
from pathlib import Path

import pytest

from deafbench import cli


pytestmark = pytest.mark.functional


def test_benchmark_launcher_calls_installed_runner(monkeypatch):
    calls = []
    runner = types.ModuleType("deafbench.benchmark.runner")
    runner.main = lambda args: calls.append(args) or 7  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deafbench.benchmark.runner", runner)

    status = cli._run_benchmark(["core-v1", "--model", "whisper"])

    assert status == 7
    assert calls == [["core-v1", "--model", "whisper"]]


def test_benchmark_launcher_reraises_missing_runtime_dependency(monkeypatch):
    original_import = builtins.__import__

    def import_with_missing_dependency(
        name,
        globalns=None,
        localns=None,
        fromlist=(),
        level=0,
    ) -> object:
        if name == "benchmark.runner" and level == 1:
            raise ModuleNotFoundError(name="model_runtime")
        return original_import(name, globalns, localns, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_missing_dependency)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        cli._run_benchmark([])

    assert exc_info.value.name == "model_runtime"


def test_benchmark_command_forwards_defaults_to_lazy_launcher(monkeypatch):
    calls = []
    monkeypatch.setitem(cli.__dict__, "_run_benchmark", calls.append)

    cli.main(["benchmark", "core-v1", "--model", "whisper"])

    assert calls == [[
        "core-v1",
        "--model", "whisper",
        "--audio-source", "auto",
        "--scene-profile", "default-v1",
        "--seed", "42",
    ]]


def test_benchmark_command_returns_launcher_status(monkeypatch):
    monkeypatch.setitem(cli.__dict__, "_run_benchmark", lambda _args: 9)

    assert cli.main(["benchmark", "core-v1", "--model", "whisper"]) == 9


def test_compare_does_not_import_benchmark_runner(tmp_path):
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    references.write_text('{"id":"s1","text":"hello"}\n', encoding="utf-8")
    predictions.write_text('{"id":"s1","text":"hello"}\n', encoding="utf-8")
    sys.modules.pop("deafbench.benchmark.runner", None)

    cli.main(["compare", str(references), str(predictions)])

    assert "deafbench.benchmark.runner" not in sys.modules


def test_benchmark_command_prints_complete_terminal_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner_module = importlib.import_module("deafbench.benchmark.runner")
    BenchmarkResult = runner_module.BenchmarkResult
    run_dir = tmp_path / "benchmarks/non-speech-v1/runs/whisper-at/synthetic"
    result = BenchmarkResult(
        "synthetic",
        run_dir / "predictions.jsonl",
        run_dir / "report.md",
        run_dir / "run.json",
        {
            "samples": 1,
            "wer": 0.0,
            "critical_recall": 100.0,
            "non_speech_recall": 100.0,
            "speaker_accuracy": None,
            "median_latency_ms": None,
            "critical_failures": [],
        },
    )
    monkeypatch.setattr(runner_module, "run_benchmark", lambda _config: result)

    status = cli.main([
        "benchmark",
        "non-speech-v1",
        "--model",
        "whisper-at",
        "--audio-source",
        "synthetic",
        "--repo-root",
        str(tmp_path),
    ])

    assert status == 0
    output = capsys.readouterr().out
    for expected in (
        "Dataset: non-speech-v1",
        "Model: whisper-at",
        "Audio source: synthetic",
        "WER",
        "Critical Information",
        "Predictions:",
        "Report:",
    ):
        assert expected in output
    assert "WER                          0.0%" in output
    assert "Strict Critical Information     100.0%" in output
    assert "Canonical Critical Information  100.0%" in output
    assert "Non-Speech Information     100.0%" in output
    assert f"Predictions: {result.predictions}" in output
    assert f"Report: {result.report}" in output
