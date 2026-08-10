import builtins
import json
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.models import faster_whisper as adapter
from deafbench.benchmark.models.faster_whisper import run_faster_whisper


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    references = tmp_path / "references.jsonl"
    references.write_text(
        '{"id":"sample-001","text":"Reference caption"}\n',
        encoding="utf-8",
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    with wave.open(str(audio_dir / "sample-001.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\x00\x00" * 8)
    return references, audio_dir


def test_faster_whisper_writes_complete_segment_transcript(
    tmp_path: Path,
) -> None:
    references, audio_dir = _write_dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    calls: dict[str, object] = {}

    @dataclass(frozen=True)
    class Segment:
        text: str

    class FakeModel:
        def transcribe(self, path: str, **kwargs: object) -> tuple[object, None]:
            calls["path"] = path
            calls["transcribe_kwargs"] = kwargs
            return iter((Segment(" Hello"), Segment(" world."))), None

    class FakeBackend:
        def WhisperModel(self, model_id: str, **kwargs: object) -> FakeModel:
            calls["model_id"] = model_id
            calls["model_kwargs"] = kwargs
            return FakeModel()

    info = run_faster_whisper(
        audio_dir,
        references,
        output,
        backend=FakeBackend(),
    )

    assert info == ModelRunInfo("faster-whisper", "small.en")
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"id": "sample-001", "text": " Hello world."}
    ]
    assert calls["model_kwargs"] == {
        "device": "cpu",
        "compute_type": "int8",
    }
    assert calls["transcribe_kwargs"] == {
        "beam_size": 5,
        "language": "en",
    }


def test_faster_whisper_reports_missing_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_runtime(name: str, *args: object, **kwargs: object) -> object:
        if name == "faster_whisper":
            raise ModuleNotFoundError(name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_runtime)

    with pytest.raises(RuntimeError, match="pip install -U faster-whisper"):
        adapter._load_backend()


def test_faster_whisper_failure_preserves_previous_output(
    tmp_path: Path,
) -> None:
    references, audio_dir = _write_dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    output.write_text("previous predictions\n", encoding="utf-8")

    @dataclass(frozen=True)
    class InvalidSegment:
        text: None = None

    class FakeModel:
        def transcribe(self, *_args: object, **_kwargs: object) -> tuple[object, None]:
            return iter((InvalidSegment(),)), None

    class FakeBackend:
        def WhisperModel(self, *_args: object, **_kwargs: object) -> FakeModel:
            return FakeModel()

    with pytest.raises(ValueError, match="expected text to be a string"):
        run_faster_whisper(
            audio_dir,
            references,
            output,
            backend=FakeBackend(),
        )

    assert output.read_text(encoding="utf-8") == "previous predictions\n"
