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
    assert "Critical Information" in output


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
