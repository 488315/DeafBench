import pytest

from deafbench.cli import main


def test_report_write_error_exits_cleanly(tmp_path, capsys):
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "report-dir"

    references.write_text('{"id":"s1","text":"hello"}\n', encoding="utf-8")
    predictions.write_text('{"id":"s1","text":"hello"}\n', encoding="utf-8")
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
