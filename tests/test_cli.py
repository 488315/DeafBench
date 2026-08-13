import runpy

import pytest

from deafbench.cli import main


pytestmark = pytest.mark.functional


def _write_minimal_inputs(tmp_path):
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    references.write_text(
        '{"id":"s1","text":"hello","critical":["hello"]}\n',
        encoding="utf-8",
    )
    predictions.write_text(
        '{"id":"s1","text":"hello"}\n',
        encoding="utf-8",
    )
    return references, predictions


def test_compare_command_prints_summary(tmp_path, capsys):
    references, predictions = _write_minimal_inputs(tmp_path)

    main(["compare", str(references), str(predictions)])

    output = capsys.readouterr().out
    assert "DeafBench v0.1" in output
    assert "Samples: 1" in output
    assert "Strict Critical Information" in output
    assert "Canonical Critical Information" in output
    assert "Non-Speech Information       N/A" in output


def test_report_command_writes_markdown(tmp_path, capsys):
    references, predictions = _write_minimal_inputs(tmp_path)
    output_path = tmp_path / "report.md"

    main([
        "report",
        str(references),
        str(predictions),
        "--output",
        str(output_path),
    ])

    assert output_path.exists()
    assert "# DeafBench Evaluation Report" in output_path.read_text(encoding="utf-8")
    assert f"Report successfully saved to {output_path}" in capsys.readouterr().out


def test_report_write_error_exits_cleanly(tmp_path, capsys):
    references, predictions = _write_minimal_inputs(tmp_path)
    output_dir = tmp_path / "report-dir"
    output_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        main([
            "report",
            str(references),
            str(predictions),
            "--output",
            str(output_dir),
        ])

    assert exc_info.value.code == 1
    assert "Error writing report:" in capsys.readouterr().err


@pytest.mark.parametrize("latency", ['"not-a-number"', "-1", '"nan"', '"inf"'])
def test_compare_rejects_invalid_latency(tmp_path, capsys, latency):
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    references.write_text('{"id":"s1","text":"hello"}\n', encoding="utf-8")
    predictions.write_text(
        f'{{"id":"s1","text":"hello","latency_ms":{latency}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["compare", str(references), str(predictions)])

    assert exc_info.value.code == 1
    assert "Invalid latency_ms for sample s1" in capsys.readouterr().err


def test_recorder_command_forwards_dataset_to_lazy_launcher(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "deafbench.cli._run_recorder",
        calls.append,
    )

    main(["recorder", "--dataset", "non-speech-v1"])

    assert calls == [["--dataset", "non-speech-v1"]]


def test_recorder_command_returns_launcher_status(monkeypatch):
    monkeypatch.setattr(
        "deafbench.cli._run_recorder",
        lambda _recorder_args: 7,
    )

    assert main(["recorder"]) == 7


def test_audit_command_forwards_nested_action(monkeypatch):
    calls = []
    monkeypatch.setattr("deafbench.cli._run_audit", calls.append)

    main(["audit", "rehearse", "--repo-root", "."])

    assert calls == [["rehearse", "--repo-root", "."]]


def test_audit_command_returns_launcher_status(monkeypatch):
    monkeypatch.setattr("deafbench.cli._run_audit", lambda _audit_args: 7)

    assert main(["audit", "run"]) == 7


def test_audit_help_is_owned_by_audit_cli(monkeypatch):
    calls = []
    monkeypatch.setattr("deafbench.cli._run_audit", calls.append)

    main(["audit", "--help"])

    assert calls == [["--help"]]


def test_dev_corpus_command_forwards_nested_action(monkeypatch):
    calls = []
    monkeypatch.setattr("deafbench.cli._run_dev_corpus", calls.append)

    main(["dev-corpus", "materialize", "--repo-root", "."])

    assert calls == [["materialize", "--repo-root", "."]]


def test_dev_corpus_command_returns_launcher_status(monkeypatch):
    monkeypatch.setattr("deafbench.cli._run_dev_corpus", lambda _args: 7)

    assert main(["dev-corpus", "materialize"]) == 7


def test_module_entrypoint_propagates_main_status(monkeypatch):
    monkeypatch.setattr("deafbench.cli.main", lambda _args=None: 7)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("deafbench", run_name="__main__")

    assert exc_info.value.code == 7
