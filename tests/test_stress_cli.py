import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deafbench import cli
from deafbench.benchmark import stress_cli
from deafbench.benchmark.models import ModelRunInfo


def test_root_cli_delegates_stress_arguments(monkeypatch):
    received = []
    monkeypatch.setattr(cli, "_run_stress", lambda arguments: received.extend(arguments) or 0)

    assert cli.main(["stress", "--implemented-only"]) == 0
    assert received == ["--implemented-only"]


def test_stress_cli_runs_installed_workflow(monkeypatch, tmp_path, capsys):
    received = {}

    def fake_run(references, clean_audio, destination, model, **options):
        received.update(
            references=references,
            clean_audio=clean_audio,
            destination=destination,
            model=model,
            options=options,
        )
        return {"sample_count": 20}

    monkeypatch.setattr(stress_cli, "run_stress_benchmark", fake_run)
    output = tmp_path / "run"

    assert stress_cli.main(
        [
            "--references",
            "references.jsonl",
            "--clean-audio",
            "audio-clean",
            "--output",
            str(output),
            "--model",
            "faster-whisper",
            "--implemented-only",
        ]
    ) == 0

    assert received["references"] == Path("references.jsonl")
    assert received["model"] == "faster-whisper"
    assert received["options"]["implemented_only"] is True
    assert f"Result: {output / 'evaluation' / 'result.json'}" in capsys.readouterr().out


def test_stress_cli_reports_validation_error(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ValueError("unsupported stressors")

    monkeypatch.setattr(stress_cli, "run_stress_benchmark", fail)

    with pytest.raises(SystemExit, match="unsupported stressors"):
        stress_cli.main(
            [
                "--references",
                "references.jsonl",
                "--clean-audio",
                "audio-clean",
                "--output",
                "run",
                "--model",
                "faster-whisper",
            ]
        )


def test_run_stress_benchmark_executes_supported_case_atomically(tmp_path):
    references = tmp_path / "references.jsonl"
    references.write_text(
        json.dumps(
            {
                "id": "stress-001",
                "text": "Code 481926",
                "critical": ["481926"],
                "critical_types": {"481926": "PASSWORD"},
                "risk_categories": {"481926": "CODE"},
                "sounds": [],
                "stressors": [
                    {"kind": "clean"},
                    {
                        "kind": "additive_noise",
                        "profile": "street-noise",
                        "snr_db": 0.0,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audio = tmp_path / "audio-clean"
    audio.mkdir()
    time = np.arange(8_000, dtype=np.float32) / 16_000
    sf.write(audio / "stress-001.wav", 0.1 * np.sin(2 * np.pi * 220 * time), 16_000)

    def runner(audio_dir, _references, predictions):
        predictions.write_text(
            json.dumps(
                {
                    "id": "stress-001",
                    "text": "Code 481926" if audio_dir.name == "clean" else "Code 481926",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return ModelRunInfo("test", "example/test", "revision-1")

    destination = tmp_path / "run"
    result = stress_cli.run_stress_benchmark(
        references,
        audio,
        destination,
        "faster-whisper",
        implemented_only=True,
        model_runner=runner,
    )

    assert result["sample_count"] == 1
    assert (destination / "prepared" / "preparation-manifest.json").is_file()
    assert (destination / "evaluation" / "result.json").is_file()


def test_run_stress_benchmark_rejects_conflicting_selection(tmp_path):
    with pytest.raises(ValueError, match="either implemented-only"):
        stress_cli.run_stress_benchmark(
            tmp_path / "references.jsonl",
            tmp_path / "audio",
            tmp_path / "run",
            "faster-whisper",
            case_ids=["stress-001"],
            implemented_only=True,
            model_runner=lambda *_args: ModelRunInfo("test", "test"),
        )


def test_run_stress_benchmark_preserves_existing_destination(tmp_path):
    destination = tmp_path / "run"
    destination.mkdir()

    with pytest.raises(ValueError, match="destination already exists"):
        stress_cli.run_stress_benchmark(
            tmp_path / "references.jsonl",
            tmp_path / "audio",
            destination,
            "faster-whisper",
        )
