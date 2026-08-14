import json
import wave
from dataclasses import dataclass
from pathlib import Path

from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.models.distil_whisper import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    run_distil_whisper,
)


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
    clock_values = iter((1.0, 1.25))

    @dataclass(frozen=True)
    class Segment:
        text: str

    class FakeModel:
        def transcribe(self, path: str, **kwargs: object) -> tuple[object, object]:
            calls["path"] = path
            calls["transcribe_kwargs"] = kwargs
            info = type("Info", (), {"duration": 2.0})()
            return iter((Segment(" Distilled caption."),)), info

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
        clock=lambda: next(clock_values),
    )

    assert info == ModelRunInfo(
        "distil-whisper",
        DEFAULT_MODEL,
        revision=DEFAULT_MODEL_REVISION,
        decoding={
            "beam_size": 5,
            "compute_type": "int8",
            "condition_on_previous_text": False,
            "device": "cpu",
            "language": "en",
        },
        performance={
            "local_rtfx": 8.0,
            "median_latency_ms": 250.0,
            "peak_vram_bytes": 0,
        },
    )
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {
            "id": "sample-001",
            "latency_ms": 250.0,
            "text": " Distilled caption.",
        }
    ]
    assert calls["model_id"] == DEFAULT_MODEL
    assert calls["model_kwargs"] == {
        "compute_type": "int8",
        "device": "cpu",
        "revision": DEFAULT_MODEL_REVISION,
    }
    assert calls["transcribe_kwargs"] == {
        "beam_size": 5,
        "language": "en",
        "condition_on_previous_text": False,
    }
