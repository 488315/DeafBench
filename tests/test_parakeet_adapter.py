import builtins
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from deafbench.benchmark.models import parakeet as parakeet_adapter
from deafbench.benchmark.models.parakeet import run_parakeet


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\x00\x00" * 48_000)


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    references = tmp_path / "references.jsonl"
    references.write_text(
        '{"id":"sample-002","text":"second"}\n' '{"id":"sample-001","text":"first"}\n',
        encoding="utf-8",
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "sample-002.wav")
    _write_wav(audio_dir / "sample-001.wav")
    return references, audio_dir


@pytest.mark.parametrize(
    ("clock_values", "timing_error"),
    [
        ((1.0, 1.1, 2.0, 2.2), None),
        ((1.0, 1.0, 2.0, 2.0), "timing must be positive"),
    ],
)
def test_parakeet_adapter_pins_archive_and_reports_performance(
    tmp_path: Path,
    clock_values: tuple[float, ...],
    timing_error: str | None,
) -> None:
    references, audio_dir = _dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    downloads: list[dict[str, object]] = []
    restores: list[dict[str, object]] = []
    peak_resets: list[bool] = []

    class FakeModel:
        def __init__(self) -> None:
            self.device_moves: list[str] = []
            self.evaluated = False
            self.calls: list[tuple[list[str], int, bool]] = []

        def to(self, device: str) -> None:
            self.device_moves.append(device)

        def eval(self) -> None:
            self.evaluated = True

        def transcribe(
            self,
            paths: list[str],
            *,
            batch_size: int,
            timestamps: bool,
        ) -> list[object]:
            self.calls.append((paths, batch_size, timestamps))
            return [SimpleNamespace(text=f"transcript {len(self.calls)}")]

    model = FakeModel()

    class ModelFactory:
        @staticmethod
        def restore_from(**options: object) -> FakeModel:
            restores.append(options)
            return model

    clock_values = iter(clock_values)
    backend = SimpleNamespace(
        ASRModel=ModelFactory,
        clock=lambda: next(clock_values),
        hf_hub_download=lambda **options: (
            downloads.append(options) or "pinned-model.nemo"
        ),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                max_memory_allocated=lambda: 987_654,
                reset_peak_memory_stats=lambda: peak_resets.append(True),
            ),
            device=lambda name: name,
            inference_mode=nullcontext,
        ),
    )

    if timing_error is not None:
        output.write_text("previous predictions\n", encoding="utf-8")
        with pytest.raises(ValueError, match=timing_error):
            run_parakeet(audio_dir, references, output, backend=backend)
        assert output.read_text(encoding="utf-8") == "previous predictions\n"
        return
    info = run_parakeet(audio_dir, references, output, backend=backend)

    assert downloads == [
        {
            "repo_id": "nvidia/parakeet-tdt-0.6b-v2",
            "filename": "parakeet-tdt-0.6b-v2.nemo",
            "revision": "ae9ad07059c7c739ffaf932226a8fe64ae2620b0",
        }
    ]
    assert restores == [{"restore_path": "pinned-model.nemo", "map_location": "cuda"}]
    assert model.device_moves == ["cuda"]
    assert model.evaluated is True
    assert peak_resets == [True]
    assert [Path(call[0][0]).stem for call in model.calls] == [
        "sample-001",
        "sample-002",
    ]
    assert all(call[1:] == (1, True) for call in model.calls)
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"id": "sample-001", "latency_ms": 100.0, "text": "transcript 1"},
        {"id": "sample-002", "latency_ms": 200.0, "text": "transcript 2"},
    ]
    assert info.name == "parakeet-tdt-0.6b-v2"
    assert info.model_id == "nvidia/parakeet-tdt-0.6b-v2"
    assert info.revision == "ae9ad07059c7c739ffaf932226a8fe64ae2620b0"
    assert info.decoding == {
        "archive": "parakeet-tdt-0.6b-v2.nemo",
        "batch_size": 1,
        "device": "cuda",
        "timestamps": True,
    }
    assert info.performance == {
        "local_rtfx": pytest.approx(6.6666666667),
        "median_latency_ms": 150.0,
        "peak_vram_bytes": 987_654,
        "timing_scope": "decode_only_excludes_model_load",
    }


def test_parakeet_adapter_preserves_output_on_malformed_result(
    tmp_path: Path,
) -> None:
    references, audio_dir = _dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    output.write_text("previous predictions\n", encoding="utf-8")

    model = SimpleNamespace(
        to=lambda _device: None,
        eval=lambda: None,
        transcribe=lambda *_args, **_kwargs: [],
    )
    backend = SimpleNamespace(
        ASRModel=SimpleNamespace(restore_from=lambda **_options: model),
        clock=lambda: 1.0,
        hf_hub_download=lambda **_options: "pinned-model.nemo",
        torch=SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            device=lambda name: name,
            inference_mode=nullcontext,
        ),
    )

    with pytest.raises(ValueError, match="Invalid Parakeet transcription"):
        run_parakeet(audio_dir, references, output, backend=backend)

    assert output.read_text(encoding="utf-8") == "previous predictions\n"


def test_parakeet_adapter_reports_missing_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_backend(name: str, *args: object, **kwargs: object) -> object:
        if name == "nemo.collections.asr.models":
            raise ModuleNotFoundError(name="nemo")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_backend)

    with pytest.raises(RuntimeError, match=r"deafbench\[parakeet-asr\]"):
        parakeet_adapter._load_backend()
