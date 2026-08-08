import json
from pathlib import Path

import pytest

from tools import transcribe_whisper
from tools.transcribe_whisper import transcribe_directory


def test_transcribe_directory_writes_sorted_prediction_jsonl(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "core-002.wav").write_bytes(b"wav")
    (audio_dir / "core-001.wav").write_bytes(b"wav")
    output = tmp_path / "model-a.jsonl"

    transcripts = {
        "core-001.wav": "  First transcript.  ",
        "core-002.wav": "\tSecond transcript.\n",
    }

    records = transcribe_directory(
        audio_dir,
        output,
        lambda path: transcripts[path.name],
    )

    assert records == [
        {"id": "core-001", "text": "  First transcript.  "},
        {"id": "core-002", "text": "\tSecond transcript.\n"},
    ]
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == records


def test_transcribe_directory_rejects_empty_audio_dir(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    output = tmp_path / "model-a.jsonl"

    with pytest.raises(FileNotFoundError, match="No core WAV files found"):
        transcribe_directory(audio_dir, output, lambda path: "unused")

    assert not output.exists()


def test_default_paths_are_anchored_to_repo_root():
    repo_root = Path(transcribe_whisper.__file__).resolve().parents[1]

    assert transcribe_whisper.AUDIO_DIR == repo_root / "benchmarks" / "core-v1" / "audio"
    assert transcribe_whisper.OUTPUT == repo_root / "benchmarks" / "core-v1" / "model-a.jsonl"
