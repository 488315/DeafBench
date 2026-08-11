import builtins
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy
import pytest
from scipy.signal import resample_poly

from deafbench.benchmark.models import granite_speech as granite_adapter
from deafbench.benchmark.models.granite_speech import run_granite_speech


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\x00\x00" * 48_000)


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


class _FakeTensor:
    def __init__(self, value: object = None) -> None:
        self.value = value
        self.unsqueezed: list[int] = []

    def unsqueeze(self, dimension: int) -> "_FakeTensor":
        self.unsqueezed.append(dimension)
        return self


class _FakeInputs(dict[str, object]):
    def __init__(self) -> None:
        super().__init__(input_ids=SimpleNamespace(shape=(1, 3)))
        self.moves: list[str] = []

    def to(self, device: str) -> "_FakeInputs":
        self.moves.append(device)
        return self


class _FakeOutput:
    def __getitem__(self, key: object) -> str:
        assert key == (slice(None), slice(3, None))
        return "generated-ids"


class _FakeTokenizer:
    def __init__(self) -> None:
        self.chats: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self,
        chat: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.chats.append(chat)
        return "rendered-prompt"

    def batch_decode(self, tokens: str, **options: object) -> list[str]:
        assert tokens == "generated-ids"
        assert options == {
            "add_special_tokens": False,
            "skip_special_tokens": True,
        }
        return [f"transcript {len(self.chats)}"]


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.inputs: list[_FakeInputs] = []

    def __call__(self, *args: object, **options: object) -> _FakeInputs:
        inputs = _FakeInputs()
        self.calls.append((args, options))
        self.inputs.append(inputs)
        return inputs


class _FakeModel:
    def __init__(self) -> None:
        self.moved_to: list[str] = []
        self.evaluated = False
        self.generations: list[dict[str, object]] = []

    def to(self, device: str) -> None:
        self.moved_to.append(device)

    def eval(self) -> None:
        self.evaluated = True

    def generate(self, **inputs: object) -> _FakeOutput:
        self.generations.append(inputs)
        return _FakeOutput()


def test_granite_adapter_pins_runtime_and_applies_keyword_prompt(
    tmp_path: Path,
) -> None:
    references, audio_dir = _dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    processor = _FakeProcessor()
    model = _FakeModel()
    processor_loads: list[tuple[str, dict[str, object]]] = []
    model_loads: list[tuple[str, dict[str, object]]] = []
    waveforms: list[_FakeTensor] = []
    peak_resets: list[bool] = []

    class ProcessorFactory:
        @staticmethod
        def from_pretrained(model_id: str, **options: object) -> _FakeProcessor:
            processor_loads.append((model_id, options))
            return processor

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_id: str, **options: object) -> _FakeModel:
            model_loads.append((model_id, options))
            return model

    def from_numpy(samples: object) -> _FakeTensor:
        waveform = _FakeTensor(samples)
        waveforms.append(waveform)
        return waveform

    clock_values = iter((1.0, 1.1, 2.0, 2.2))
    backend = SimpleNamespace(
        AutoModelForSpeechSeq2Seq=ModelFactory,
        AutoProcessor=ProcessorFactory,
        clock=lambda: next(clock_values),
        numpy=numpy,
        resample_poly=resample_poly,
        torch=SimpleNamespace(
            bfloat16="bfloat16",
            float32="float32",
            from_numpy=from_numpy,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                max_memory_allocated=lambda: 456_789,
                reset_peak_memory_stats=lambda: peak_resets.append(True),
            ),
            device=lambda name: name,
            inference_mode=nullcontext,
        ),
    )

    info = run_granite_speech(
        audio_dir,
        references,
        output,
        keywords=("Dr. Martinez", "8:30 PM"),
        backend=backend,
    )

    pinned_options = {
        "revision": "de575db64086f84fdc79da4932d1076e965bc546",
        "trust_remote_code": False,
    }
    assert processor_loads == [(granite_adapter.MODEL_ID, pinned_options)]
    assert model_loads == [
        (granite_adapter.MODEL_ID, {"torch_dtype": "bfloat16", **pinned_options})
    ]
    assert processor.tokenizer.chats == [
        [
            {
                "role": "user",
                "content": (
                    "<|audio|>transcribe the speech to text. Keywords: "
                    "Dr. Martinez, 8:30 PM"
                ),
            }
        ]
    ]
    assert all(waveform.unsqueezed == [0] for waveform in waveforms)
    assert all(call[0][0] == "rendered-prompt" for call in processor.calls)
    assert all(call[0][1] in waveforms for call in processor.calls)
    assert all(call[1] == {"device": "cuda", "return_tensors": "pt"} for call in processor.calls)
    assert all(inputs.moves == ["cuda"] for inputs in processor.inputs)
    assert model.moved_to == ["cuda"]
    assert model.evaluated is True
    assert peak_resets == [True]
    assert all(
        generation["max_new_tokens"] == 200
        and generation["do_sample"] is False
        and generation["num_beams"] == 1
        for generation in model.generations
    )
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"id": "sample-001", "latency_ms": 100.0, "text": "transcript 1"},
        {"id": "sample-002", "latency_ms": 200.0, "text": "transcript 1"},
    ]
    assert info.decoding == {
        "device": "cuda",
        "dtype": "bfloat16",
        "keyword_biasing": True,
        "max_new_tokens": 200,
        "num_beams": 1,
        "trust_remote_code": False,
    }
    assert info.performance == {
        "local_rtfx": pytest.approx(6.6666666667),
        "median_latency_ms": 150.0,
        "peak_vram_bytes": 456_789,
        "timing_scope": "decode_only_excludes_model_load",
    }


@pytest.mark.parametrize("keyword", ["", "comma,value", "line\nbreak"])
def test_granite_adapter_rejects_unsafe_keywords(
    tmp_path: Path,
    keyword: str,
) -> None:
    references, audio_dir = _dataset(tmp_path)

    with pytest.raises(ValueError, match="plain nonempty text"):
        run_granite_speech(
            audio_dir,
            references,
            tmp_path / "predictions.jsonl",
            keywords=(keyword,),
            backend=SimpleNamespace(),
        )


def test_granite_adapter_reports_missing_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_backend(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers":
            raise ModuleNotFoundError(name="transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_backend)

    with pytest.raises(RuntimeError, match=r"deafbench\[granite-asr\]"):
        granite_adapter._load_backend()
