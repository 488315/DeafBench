import json
import subprocess

import pytest

from deafbench.benchmark.models import _isolated


def test_worker_scrubs_secrets_and_forces_offline_mode(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("PATH", "runtime-path")
    captured: dict[str, object] = {}

    def fake_run(command, **options):
        captured["command"] = command
        captured.update(options)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='noise\nDEAFBENCH_MODEL_RESULT={"records": []}\n',
            stderr="",
        )

    monkeypatch.setattr(_isolated.subprocess, "run", fake_run)

    result = _isolated.invoke_isolated_worker(
        "deafbench.benchmark.models._example_worker",
        {"snapshot": "/model"},
    )

    assert result == {"records": []}
    assert json.loads(captured["input"]) == {"snapshot": "/model"}
    environment = captured["env"]
    assert environment["PATH"] == "runtime-path"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "HF_TOKEN" not in environment
    assert "shell" not in captured


def test_worker_rejects_external_module() -> None:
    with pytest.raises(_isolated.IsolatedModelError, match="unsafe isolated worker"):
        _isolated.invoke_isolated_worker("example.worker", {})


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("", "did not return exactly one result marker"),
        (
            "DEAFBENCH_MODEL_RESULT={}\nDEAFBENCH_MODEL_RESULT={}\n",
            "did not return exactly one result marker",
        ),
        ("DEAFBENCH_MODEL_RESULT=not-json\n", "returned malformed JSON"),
    ],
)
def test_worker_rejects_invalid_result(monkeypatch, stdout, message) -> None:
    monkeypatch.setattr(
        _isolated.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout, ""),
    )

    with pytest.raises(_isolated.IsolatedModelError, match=message):
        _isolated.invoke_isolated_worker(
            "deafbench.benchmark.models._example_worker",
            {},
        )
