import json
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy
from scipy.signal import resample_poly

from deafbench.benchmark.models import qwen3_asr
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


@pytest.mark.parametrize(
    ("model_id", "revision", "adapter_name"),
    [
        (
            "Qwen/Qwen3-ASR-0.6B-hf",
            "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
            "qwen3-asr-0.6b",
        ),
        (
            "Qwen/Qwen3-ASR-1.7B-hf",
            "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
            "qwen3-asr-1.7b",
        ),
    ],
)
def test_qwen_adapter_pins_safe_runtime_and_writes_sorted_predictions(
    tmp_path: Path,
    model_id: str,
    revision: str,
    adapter_name: str,
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
        model_id=model_id,
        backend=backend,
    )

    expected_load = (
        model_id,
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
    assert info.name == adapter_name
    assert info.model_id == model_id
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


@pytest.mark.parametrize(
    ("license_metadata", "message"),
    [
        (
            SimpleNamespace(
                remote_code_required=True,
                supported_runtimes=("transformers",),
                revision="a" * 40,
            ),
            "rejects remote-code model",
        ),
        (
            SimpleNamespace(
                remote_code_required=False,
                supported_runtimes=("onnxruntime",),
                revision="b" * 40,
            ),
            "requires a registered Transformers runtime",
        ),
    ],
)
def test_qwen_adapter_rejects_unsafe_registered_runtime(
    monkeypatch: pytest.MonkeyPatch,
    license_metadata: SimpleNamespace,
    message: str,
) -> None:
    monkeypatch.setattr(
        qwen3_asr,
        "get_model_license",
        lambda _model_id: license_metadata,
    )

    with pytest.raises(ModelRegistryError, match=message):
        qwen3_asr._licensed_revision("registered/model")


@pytest.mark.parametrize("decoded", ["transcript", [], [1], ["one", "two"]])
def test_qwen_adapter_rejects_malformed_transcription(decoded: object) -> None:
    processor = SimpleNamespace(decode=lambda *_args, **_kwargs: decoded)

    with pytest.raises(ValueError, match="Invalid Qwen3-ASR transcription output"):
        qwen3_asr._decode_transcription(processor, "generated-ids")


def test_qwen_audio_reader_preserves_matching_sample_rate(tmp_path: Path) -> None:
    audio = tmp_path / "matching-rate.wav"
    _write_wav(audio)
    unexpected_resample = Mock(side_effect=AssertionError("resampler called"))

    samples, duration = qwen3_asr._read_pcm16_mono(
        audio,
        48_000,
        numpy,
        unexpected_resample,
    )

    unexpected_resample.assert_not_called()
    assert samples.shape == (16,)
    assert samples.dtype == numpy.float32
    assert duration == pytest.approx(16 / 48_000)


def test_qwen_audio_reader_rejects_non_mono_pcm16(tmp_path: Path) -> None:
    audio = tmp_path / "stereo.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\x00\x00" * 32)

    with pytest.raises(ValueError, match="requires mono PCM16 WAV"):
        qwen3_asr._read_pcm16_mono(audio, 16_000, numpy, resample_poly)
