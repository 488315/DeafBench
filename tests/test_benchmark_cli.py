import builtins
import sys
import types

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
