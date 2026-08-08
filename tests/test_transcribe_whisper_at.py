import json
import os
import subprocess
import sys
import types
import wave
from pathlib import Path

import pytest

from tools import transcribe_whisper_at
from tools.transcribe_whisper_at import (
    extract_audio_tags,
    resolve_dataset_paths,
    transcribe_directory,
)


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48_000)
        wav_file.writeframes(b"\x00" * 16)


def test_resolve_dataset_paths_writes_model_b(tmp_path):
    references, audio_dir, output = resolve_dataset_paths(tmp_path, "non-speech-v1")

    dataset_dir = tmp_path / "benchmarks" / "non-speech-v1"
    assert references == dataset_dir / "references.jsonl"
    assert audio_dir == dataset_dir / "audio"
    assert output == dataset_dir / "model-b.jsonl"


def test_extract_audio_tags_maps_audioset_labels_to_deafbench_sounds():
    parsed = [
        {
            "time": {"start": 0, "end": 10},
            "audio tags": [
                ("Speech", 2.0),
                ("Alarm", 1.8),
                ("Slam", 1.5),
                ("Telephone bell ringing", 1.2),
            ],
        },
        {
            "time": {"start": 10, "end": 20},
            "audio tags": [
                ("Knock", 1.1),
                ("Siren", 0.9),
                ("Beep, bleep", 0.7),
                ("Alarm", 0.5),
            ],
        },
    ]

    raw_tags, sounds = extract_audio_tags(parsed)

    assert raw_tags == [
        "Speech",
        "Alarm",
        "Slam",
        "Telephone bell ringing",
        "Knock",
        "Siren",
        "Beep, bleep",
    ]
    assert sounds == [
        "[alarm]",
        "[door closes]",
        "[phone rings]",
        "[knock]",
        "[siren]",
        "[error notification]",
    ]


def test_extract_audio_tags_keeps_broad_labels_raw_only():
    parsed = [
        {
            "audio tags": [
                ("Door", 1.5),
                ("Sliding door", 1.2),
                ("Telephone", 1.0),
            ]
        }
    ]

    raw_tags, sounds = extract_audio_tags(parsed)

    assert raw_tags == ["Door", "Sliding door", "Telephone"]
    assert sounds == []


def test_transcribe_directory_writes_structured_model_b_predictions(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "ns-002.wav")
    _write_wav(audio_dir / "ns-001.wav")
    output = tmp_path / "model-b.jsonl"

    def transcribe(path: Path):
        return {
            "text": f"Transcript for {path.stem}",
            "sounds": ["[alarm]"] if path.stem == "ns-001" else [],
            "audio_tags": ["Alarm"] if path.stem == "ns-001" else ["Speech"],
        }

    records = transcribe_directory(audio_dir, output, transcribe)

    assert records == [
        {
            "id": "ns-001",
            "text": "Transcript for ns-001",
            "sounds": ["[alarm]"],
            "audio_tags": ["Alarm"],
        },
        {
            "id": "ns-002",
            "text": "Transcript for ns-002",
            "sounds": [],
            "audio_tags": ["Speech"],
        },
    ]
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == records


def test_main_keeps_audio_tags_separate_from_asr_text(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "benchmarks" / "non-speech-v1"
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir(parents=True)
    _write_wav(audio_dir / "ns-001.wav")
    (dataset_dir / "references.jsonl").write_text(
        '{"id":"ns-001","text":"Please remain seated.","sounds":["[alarm]"]}\n',
        encoding="utf-8",
    )

    calls = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            calls["path"] = path
            calls["transcribe_kwargs"] = kwargs
            return {"text": " Please remain seated. "}

    fake_module = types.SimpleNamespace()
    fake_module.load_model = lambda name: (calls.__setitem__("model", name) or FakeModel())

    def parse_at_label(result, **kwargs):
        calls["parse_result"] = result
        calls["parse_kwargs"] = kwargs
        return [{"audio tags": [("Speech", 2.0), ("Alarm", 1.5)]}]

    fake_module.parse_at_label = parse_at_label
    monkeypatch.setitem(sys.modules, "whisper_at", fake_module)

    transcribe_whisper_at.main([
        "--repo-root",
        str(tmp_path),
        "--dataset",
        "non-speech-v1",
        "--model",
        "tiny.en",
    ])

    records = [
        json.loads(line)
        for line in (dataset_dir / "model-b.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "id": "ns-001",
            "text": " Please remain seated. ",
            "sounds": ["[alarm]"],
            "audio_tags": ["Speech", "Alarm"],
        }
    ]
    assert calls["model"] == "tiny.en"
    assert calls["transcribe_kwargs"]["language"] == "en"
    assert calls["transcribe_kwargs"]["task"] == "transcribe"
    assert calls["transcribe_kwargs"]["at_time_res"] == 10.0
    assert calls["parse_kwargs"]["top_k"] == 5
    assert calls["parse_kwargs"]["p_threshold"] == -1.0


def test_main_rejects_invalid_dataset_before_import(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "whisper_at", object())

    with pytest.raises(SystemExit) as exc_info:
        transcribe_whisper_at.main([
            "--repo-root",
            str(tmp_path),
            "--dataset",
            "C:temp",
        ])

    assert exc_info.value.code == 2
    assert "Invalid dataset name" in capsys.readouterr().err


def test_script_runs_directly_without_repo_root_on_python_path(tmp_path):
    script = Path(transcribe_whisper_at.__file__).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = ""

    result = subprocess.run(
        [sys.executable, "-S", str(script), "--dataset", "C:temp"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Invalid dataset name" in result.stderr


@pytest.mark.parametrize("value", ["0", "-0.4", "0.5"])
def test_main_rejects_invalid_at_time_res_before_model_loading(
    tmp_path,
    monkeypatch,
    capsys,
    value,
):
    monkeypatch.setitem(sys.modules, "whisper_at", object())

    with pytest.raises(SystemExit) as exc_info:
        transcribe_whisper_at.main([
            "--repo-root",
            str(tmp_path),
            "--dataset",
            "core-v1",
            "--at-time-res",
            value,
        ])

    assert exc_info.value.code == 2
    assert "at-time-res" in capsys.readouterr().err.lower()
