from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from deafbench.benchmark.models import _ark_asr_worker as worker


class _Inputs(dict):
    input_ids = SimpleNamespace(shape=(1, 4))

    def to(self, device):
        assert device == "cuda"
        return self


class _Audios:
    def to(self, *, dtype):
        assert dtype == "float16"
        return "fp16-audio"


class _Outputs:
    def __getitem__(self, key):
        assert key == (slice(None), slice(4, None))
        return ["generated-tokens"]


class _Processor:
    def apply_chat_template(self, conversation, **options):
        assert conversation[0]["content"][0] == {
            "type": "audio",
            "path": conversation[0]["content"][0]["path"],
        }
        assert conversation[0]["content"][1]["text"] == "Please transcribe this audio."
        assert options["audio_max_length"] == 30 * 16_000
        return _Inputs(audios=_Audios())


class _Tokenizer:
    eos_token_id = [9]
    pad_token_id = 8
    all_special_ids = [7, 8, 9]

    def get_added_vocab(self):
        return {"<audio>": 10, "ordinary": 11}

    def batch_decode(self, outputs, *, skip_special_tokens):
        assert outputs == ["generated-tokens"]
        assert skip_special_tokens is True
        return ["recognized speech"]


class _Model:
    def to(self, device):
        assert device == "cuda"
        return self

    def eval(self):
        return None

    def generate(self, **inputs):
        assert inputs["audios"] == "fp16-audio"
        assert inputs["do_sample"] is False
        assert inputs["bad_words_ids"] == [[7], [8], [10]]
        return _Outputs()


def _backend():
    load_calls: list[tuple[str, str, dict[str, object]]] = []

    class AutoProcessor:
        @staticmethod
        def from_pretrained(path, **options):
            load_calls.append(("processor", path, options))
            return _Processor()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **options):
            load_calls.append(("tokenizer", path, options))
            return _Tokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **options):
            load_calls.append(("model", path, options))
            return _Model()

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def max_memory_allocated():
            return 1234

        @staticmethod
        def synchronize():
            return None

    clock_values = iter([10.0, 10.25])
    backend = worker._Backend(
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoProcessor=AutoProcessor,
        AutoTokenizer=AutoTokenizer,
        clock=lambda: next(clock_values),
        soundfile=SimpleNamespace(
            info=lambda path: SimpleNamespace(
                channels=1,
                duration=1.0,
                samplerate=16_000,
            )
        ),
        torch=SimpleNamespace(
            float16="float16",
            cuda=Cuda,
            device=lambda value: value,
            inference_mode=nullcontext,
        ),
    )
    return backend, load_calls


def test_worker_verifies_source_and_uses_official_generation_contract(
    monkeypatch,
    tmp_path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"fixture")
    order: list[str] = []
    audit = SimpleNamespace(revision="a" * 40)
    monkeypatch.setattr(worker, "load_remote_code_audit", lambda model_id: audit)
    monkeypatch.setattr(
        worker,
        "verify_audited_files",
        lambda received, root: order.append("verify"),
    )
    backend, load_calls = _backend()

    result = worker.run_request(
        {
            "model_id": worker.MODEL_ID,
            "revision": "a" * 40,
            "snapshot_root": str(snapshot.resolve()),
            "wav_paths": [str(wav.resolve())],
        },
        backend,
    )

    assert order == ["verify"]
    assert [call[0] for call in load_calls] == ["processor", "tokenizer", "model"]
    assert all(call[2]["local_files_only"] is True for call in load_calls)
    assert all(call[2]["trust_remote_code"] is True for call in load_calls)
    assert result["records"] == [
        {"id": "sample", "latency_ms": 250.0, "text": "recognized speech"}
    ]
    assert result["performance"] == {
        "local_rtfx": 4.0,
        "median_latency_ms": 250.0,
        "peak_vram_bytes": 1234,
        "timing_scope": "decode_only_excludes_model_load",
    }


def test_worker_rejects_revision_before_backend_load(monkeypatch, tmp_path) -> None:
    audit = SimpleNamespace(revision="a" * 40)
    monkeypatch.setattr(worker, "load_remote_code_audit", lambda model_id: audit)

    with pytest.raises(ValueError, match="revision differs"):
        worker.run_request(
            {
                "model_id": worker.MODEL_ID,
                "revision": "b" * 40,
                "snapshot_root": str(tmp_path.resolve()),
                "wav_paths": [],
            }
        )


def test_worker_chunks_audio_longer_than_official_limit(
    monkeypatch,
    tmp_path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"fixture")
    audit = SimpleNamespace(revision="a" * 40)
    monkeypatch.setattr(worker, "load_remote_code_audit", lambda model_id: audit)
    monkeypatch.setattr(worker, "verify_audited_files", lambda audit, root: None)
    backend, _ = _backend()
    writes: list[tuple[str, int]] = []
    backend.soundfile.info = lambda path: SimpleNamespace(
        channels=1,
        duration=30.01,
        frames=480_160,
        samplerate=16_000,
    )
    backend.soundfile.read = lambda path, **options: (
        [0] * options["frames"],
        16_000,
    )
    backend.soundfile.write = lambda path, audio, sample_rate: writes.append(
        (path, len(audio))
    )
    backend = replace(backend, clock=iter([10.0, 10.5]).__next__)

    result = worker.run_request(
        {
            "model_id": worker.MODEL_ID,
            "revision": "a" * 40,
            "snapshot_root": str(snapshot.resolve()),
            "wav_paths": [str(wav.resolve())],
        },
        backend,
    )

    assert [frames for _, frames in writes] == [480_000, 160]
    assert result["records"] == [
        {
            "id": "sample",
            "latency_ms": 500.0,
            "text": "recognized speech recognized speech",
        }
    ]
    assert result["decoding"]["long_audio_strategy"] == (
        "contiguous_30_second_chunks_without_overlap"
    )
