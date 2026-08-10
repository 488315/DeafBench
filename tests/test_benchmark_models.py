import json
import math
import wave
from pathlib import Path

import pytest

from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.models.whisper import run_whisper
from deafbench.benchmark.models.whisper_at import (
    extract_audio_tags,
    run_whisper_at,
)


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\x00\x00" * 8)


def _dataset(
    tmp_path: Path,
    *sample_ids: str,
    sounds: list[str] | None = None,
) -> tuple[Path, Path]:
    references = tmp_path / "references.jsonl"
    references.write_text(
        "".join(
            json.dumps(
                {
                    "id": sample_id,
                    "text": f"Reference for {sample_id}",
                    "sounds": sounds or [],
                }
            )
            + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for sample_id in reversed(sample_ids):
        _write_wav(audio_dir / f"{sample_id}.wav")
    return references, audio_dir


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_whisper_adapter_writes_sorted_model_a_predictions(
    tmp_path: Path,
) -> None:
    references, audio_dir = _dataset(tmp_path, "core-002", "core-001")
    output = tmp_path / "predictions.jsonl"
    calls: dict[str, object] = {}

    class FakeModel:
        def transcribe(self, path: str, **kwargs: object) -> dict[str, str]:
            calls.setdefault("paths", []).append(path)  # type: ignore[union-attr]
            calls["kwargs"] = kwargs
            return {"text": f" Synthetic {Path(path).stem} "}

    class FakeBackend:
        def load_model(self, name: str) -> FakeModel:
            calls["model"] = name
            return FakeModel()

    info = run_whisper(
        audio_dir,
        references,
        output,
        backend=FakeBackend(),
    )

    assert info == ModelRunInfo("whisper", "turbo")
    assert _records(output) == [
        {"id": "core-001", "text": " Synthetic core-001 "},
        {"id": "core-002", "text": " Synthetic core-002 "},
    ]
    assert calls["model"] == "turbo"
    assert calls["kwargs"] == {
        "language": "en",
        "task": "transcribe",
        "verbose": False,
    }


def test_whisper_rejects_id_mismatch_before_model_loading(
    tmp_path: Path,
) -> None:
    references, audio_dir = _dataset(tmp_path, "core-001")
    _write_wav(audio_dir / "extra.wav")

    class FailBackend:
        def load_model(self, _name: str) -> object:
            raise AssertionError("model must not load for invalid audio")

    with pytest.raises(ValueError, match="complete audio set"):
        run_whisper(
            audio_dir,
            references,
            tmp_path / "predictions.jsonl",
            backend=FailBackend(),
        )


def test_whisper_failure_preserves_previous_predictions(tmp_path: Path) -> None:
    references, audio_dir = _dataset(tmp_path, "core-001", "core-002")
    output = tmp_path / "predictions.jsonl"
    output.write_text("previous predictions\n", encoding="utf-8")

    class FailingModel:
        def transcribe(self, path: str, **_kwargs: object) -> dict[str, str]:
            if Path(path).stem == "core-002":
                raise RuntimeError("inference failed")
            return {"text": "first"}

    class FakeBackend:
        def load_model(self, _name: str) -> FailingModel:
            return FailingModel()

    with pytest.raises(RuntimeError, match="inference failed"):
        run_whisper(
            audio_dir,
            references,
            output,
            backend=FakeBackend(),
        )

    assert output.read_text(encoding="utf-8") == "previous predictions\n"


def test_whisper_at_keeps_sounds_out_of_text(tmp_path: Path) -> None:
    references, audio_dir = _dataset(
        tmp_path,
        "ns-001",
        sounds=["[alarm]"],
    )
    output = tmp_path / "predictions.jsonl"
    calls: dict[str, object] = {}

    class FakeModel:
        def transcribe(self, path: str, **kwargs: object) -> dict[str, str]:
            calls["path"] = path
            calls["transcribe_kwargs"] = kwargs
            return {"text": " Please remain seated. "}

    class FakeBackend:
        def load_model(self, name: str) -> FakeModel:
            calls["model"] = name
            return FakeModel()

        def parse_at_label(
            self,
            result: dict[str, str],
            **kwargs: object,
        ) -> list[dict[str, object]]:
            calls["parse_result"] = result
            calls["parse_kwargs"] = kwargs
            return [{"audio tags": [("Speech", 2.0), ("Alarm", 1.5)]}]

    info = run_whisper_at(
        audio_dir,
        references,
        output,
        backend=FakeBackend(),
    )
    record = _records(output)[0]

    assert info == ModelRunInfo("whisper-at", "medium.en")
    assert record == {
        "id": "ns-001",
        "text": " Please remain seated. ",
        "sounds": ["[alarm]"],
        "audio_tags": ["Speech", "Alarm"],
    }
    assert "[alarm]" not in str(record["text"])
    assert calls["model"] == "medium.en"
    assert calls["transcribe_kwargs"] == {
        "language": "en",
        "task": "transcribe",
        "verbose": False,
        "at_time_res": 10.0,
    }
    assert calls["parse_kwargs"] == {
        "language": "follow_asr",
        "top_k": 5,
        "p_threshold": -1.0,
        "include_class_list": list(range(527)),
    }


def test_extract_audio_tags_preserves_mapping_and_raw_only_labels() -> None:
    raw_tags, sounds = extract_audio_tags(
        [
            {
                "audio tags": [
                    ("Door", 3.0),
                    ("Sliding door", 2.5),
                    ("Telephone", 2.0),
                    ("Alarm", 1.5),
                    ("Slam", 1.0),
                    ("Telephone bell ringing", 0.5),
                ]
            }
        ]
    )

    assert raw_tags == [
        "Door",
        "Sliding door",
        "Telephone",
        "Alarm",
        "Slam",
        "Telephone bell ringing",
    ]
    assert sounds == ["[alarm]", "[door closes]", "[phone rings]"]


@pytest.mark.parametrize("value", [0.0, -0.4, 0.5, math.inf, math.nan])
def test_whisper_at_rejects_invalid_time_resolution_before_model_loading(
    tmp_path: Path,
    value: float,
) -> None:
    references, audio_dir = _dataset(tmp_path, "ns-001")

    class FailBackend:
        def load_model(self, _name: str) -> object:
            raise AssertionError("model must not load for invalid resolution")

    with pytest.raises(ValueError, match="positive multiple of 0.4"):
        run_whisper_at(
            audio_dir,
            references,
            tmp_path / "predictions.jsonl",
            at_time_res=value,
            backend=FailBackend(),
        )
