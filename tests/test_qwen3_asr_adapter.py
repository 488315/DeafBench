import json
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy
from scipy.signal import resample_poly

from deafbench.benchmark.models.qwen3_asr import run_qwen3_asr
from deafbench.model_registry import ModelRegistryError


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\x00\x00" * 16)


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    references = tmp_path / "references.jsonl"
    references.write_text(
        '{"id":"sample-002","text":"second"}\n'
        '{"id":"sample-001","text":"first"}\n',
        encoding="utf-8",
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "sample-002.wav")
    _write_wav(audio_dir / "sample-001.wav")
    return references, audio_dir


class _FakeInputs(dict[str, object]):
    def __init__(self) -> None:
        super().__init__(input_ids=SimpleNamespace(shape=(1, 3)))
        self.moves: list[tuple[object, object]] = []

    def to(self, device: object, dtype: object) -> "_FakeInputs":
        self.moves.append((device, dtype))
        return self


class _FakeOutput:
    def __getitem__(self, key: object) -> str:
        assert key == (slice(None), slice(3, None))
        return "generated-ids"


class _FakeProcessor:
    feature_extractor = SimpleNamespace(sampling_rate=16_000)

    def __init__(self) -> None:
        self.audio: list[object] = []
        self.inputs: list[_FakeInputs] = []

    def apply_transcription_request(
        self,
        *,
        audio: object,
        language: str,
    ) -> _FakeInputs:
        assert language == "English"
        inputs = _FakeInputs()
        self.audio.append(audio)
        self.inputs.append(inputs)
        return inputs

    def decode(self, tokens: str, *, return_format: str) -> list[str]:
        assert tokens == "generated-ids"
        assert return_format == "transcription_only"
        return [f"transcript {len(self.audio)}"]


class _FakeModel:
    device = "cuda"
    dtype = "bfloat16"

    def __init__(self) -> None:
        self.moved_to: list[str] = []
        self.evaluated = False
        self.generations: list[int] = []

    def to(self, device: str) -> "_FakeModel":
        self.moved_to.append(device)
        return self

    def eval(self) -> None:
        self.evaluated = True

    def generate(self, **inputs: object) -> _FakeOutput:
        self.generations.append(inputs.pop("max_new_tokens"))  # type: ignore[arg-type]
        assert "input_ids" in inputs
        return _FakeOutput()


def test_qwen_adapter_pins_safe_runtime_and_writes_sorted_predictions(
    tmp_path: Path,
) -> None:
    references, audio_dir = _dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    processor = _FakeProcessor()
    model = _FakeModel()
    processor_options: list[tuple[str, dict[str, object]]] = []
    model_options: list[tuple[str, dict[str, object]]] = []
    clock_values = iter((1.0, 1.1, 2.0, 2.2))
    peak_resets: list[bool] = []

    class ProcessorFactory:
        @staticmethod
        def from_pretrained(model_id: str, **options: object) -> _FakeProcessor:
            processor_options.append((model_id, options))
            return processor

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_id: str, **options: object) -> _FakeModel:
            model_options.append((model_id, options))
            return model

    backend = SimpleNamespace(
        AutoProcessor=ProcessorFactory,
        AutoModelForMultimodalLM=ModelFactory,
        clock=lambda: next(clock_values),
        numpy=numpy,
        resample_poly=resample_poly,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                max_memory_allocated=lambda: 123_456,
                reset_peak_memory_stats=lambda: peak_resets.append(True),
            ),
            device=lambda name: name,
            inference_mode=nullcontext,
        ),
    )

    info = run_qwen3_asr(
        audio_dir,
        references,
        output,
        backend=backend,
    )

    revision = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"
    expected_load = (
        "Qwen/Qwen3-ASR-0.6B-hf",
        {"revision": revision, "trust_remote_code": False},
    )
    assert processor_options == [expected_load]
    assert model_options == [expected_load]
    assert model.moved_to == ["cuda"]
    assert peak_resets == [True]
    assert model.evaluated is True
    assert model.generations == [256, 256]
    assert all(inputs.moves == [("cuda", "bfloat16")] for inputs in processor.inputs)
    assert all(isinstance(audio, numpy.ndarray) for audio in processor.audio)
    assert all(audio.shape == (6,) for audio in processor.audio)
    assert all(audio.dtype == numpy.float32 for audio in processor.audio)
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"id": "sample-001", "latency_ms": 100.0, "text": "transcript 1"},
        {"id": "sample-002", "latency_ms": 200.0, "text": "transcript 2"},
    ]
    assert info.model_id == "Qwen/Qwen3-ASR-0.6B-hf"
    assert info.revision == revision
    assert info.decoding == {
        "device": "cuda",
        "dtype": "bfloat16",
        "language": "English",
        "max_new_tokens": 256,
        "trust_remote_code": False,
    }
    assert info.performance == {
        "local_rtfx": pytest.approx(0.0022222222),
        "median_latency_ms": pytest.approx(150.0),
        "peak_vram_bytes": 123_456,
        "timing_scope": "decode_only_excludes_model_load",
    }


def test_qwen_adapter_rejects_unregistered_model_before_backend_load(
    tmp_path: Path,
) -> None:
    references, audio_dir = _dataset(tmp_path)

    with pytest.raises(ModelRegistryError, match="missing license metadata"):
        run_qwen3_asr(
            audio_dir,
            references,
            tmp_path / "predictions.jsonl",
            model_id="untrusted/model",
            backend=SimpleNamespace(),
        )
