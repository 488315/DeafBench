import json
import wave
from pathlib import Path

import pytest

from tools import transcribe_whisper
from tools.transcribe_whisper import resolve_dataset_paths, transcribe_directory


def _write_wav(path, *, channels=1, sample_width=2, sample_rate=48_000):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * channels * sample_width * 8)


def test_transcribe_directory_writes_sorted_prediction_jsonl(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "core-002.wav")
    _write_wav(audio_dir / "core-001.wav")
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


def test_transcribe_directory_accepts_non_core_sample_ids(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "ns-002.wav")
    _write_wav(audio_dir / "ns-001.wav")
    output = tmp_path / "model-a.jsonl"

    records = transcribe_directory(
        audio_dir,
        output,
        lambda path: f"Transcript for {path.stem}",
    )

    assert records == [
        {"id": "ns-001", "text": "Transcript for ns-001"},
        {"id": "ns-002", "text": "Transcript for ns-002"},
    ]


def test_transcribe_directory_rejects_empty_audio_dir(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    output = tmp_path / "model-a.jsonl"

    with pytest.raises(FileNotFoundError, match="No WAV files found"):
        transcribe_directory(audio_dir, output, lambda path: "unused")

    assert not output.exists()


def test_transcribe_directory_rejects_reference_audio_id_mismatch(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "core-001.wav")
    _write_wav(audio_dir / "core-002.wav")
    references = tmp_path / "references.jsonl"
    references.write_text(
        '{"id":"core-001","text":"First"}\n'
        '{"id":"core-003","text":"Third"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "model-a.jsonl"

    def fail_transcribe(_path):
        raise AssertionError("transcription must not start for an invalid dataset")

    with pytest.raises(ValueError, match="Reference/audio ID mismatch"):
        transcribe_directory(
            audio_dir,
            output,
            fail_transcribe,
            references=references,
        )

    assert not output.exists()


def test_transcribe_directory_preserves_existing_output_on_write_failure(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "core-001.wav")
    output = tmp_path / "model-a.jsonl"
    output.write_text("previous successful output\n", encoding="utf-8")

    def fail_dumps(*_args, **_kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(transcribe_whisper.json, "dumps", fail_dumps)

    with pytest.raises(RuntimeError, match="serialization failed"):
        transcribe_directory(audio_dir, output, lambda _path: "Transcript")

    assert output.read_text(encoding="utf-8") == "previous successful output\n"


def test_transcribe_directory_rejects_nonstandard_wav_format(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(
        audio_dir / "core-001.wav",
        channels=2,
        sample_rate=44_100,
    )
    output = tmp_path / "model-a.jsonl"

    with pytest.raises(ValueError, match="Invalid WAV format"):
        transcribe_directory(audio_dir, output, lambda _path: "Transcript")

    assert not output.exists()


def test_default_paths_are_anchored_to_repo_root():
    repo_root = Path(transcribe_whisper.__file__).resolve().parents[1]

    assert transcribe_whisper.AUDIO_DIR == repo_root / "benchmarks" / "core-v1" / "audio"
    assert transcribe_whisper.OUTPUT == repo_root / "benchmarks" / "core-v1" / "model-a.jsonl"


def test_resolve_dataset_paths_supports_non_speech_v1(tmp_path):
    references, audio_dir, output = resolve_dataset_paths(tmp_path, "non-speech-v1")

    dataset_dir = tmp_path / "benchmarks" / "non-speech-v1"
    assert references == dataset_dir / "references.jsonl"
    assert audio_dir == dataset_dir / "audio"
    assert output == dataset_dir / "model-a.jsonl"
