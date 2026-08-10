import json
import wave
from dataclasses import dataclass
from pathlib import Path

from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.models.distil_whisper import run_distil_whisper


def test_distil_whisper_uses_distilled_inference_contract(
    tmp_path: Path,
) -> None:
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
    output = tmp_path / "predictions.jsonl"
    calls: dict[str, object] = {}

    @dataclass(frozen=True)
    class Segment:
        text: str

    class FakeModel:
        def transcribe(self, path: str, **kwargs: object) -> tuple[object, None]:
            calls["path"] = path
            calls["transcribe_kwargs"] = kwargs
            return iter((Segment(" Distilled caption."),)), None

    class FakeBackend:
        def WhisperModel(self, model_id: str, **kwargs: object) -> FakeModel:
            calls["model_id"] = model_id
            calls["model_kwargs"] = kwargs
            return FakeModel()

    info = run_distil_whisper(
        audio_dir,
        references,
        output,
        backend=FakeBackend(),
    )

    assert info == ModelRunInfo("distil-whisper", "distil-large-v3")
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"id": "sample-001", "text": " Distilled caption."}
    ]
    assert calls["model_id"] == "distil-large-v3"
    assert calls["transcribe_kwargs"] == {
        "beam_size": 5,
        "language": "en",
        "condition_on_previous_text": False,
    }
