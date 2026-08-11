from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from deafbench.benchmark.models import _granite_nar_worker as worker


class _Waveform:
    ndim = 2
    shape = (1, 16_000)

    def squeeze(self, dimension):
        assert dimension == 0
        return self


class _Processor:
    def __call__(self, waveform, *, device):
        assert isinstance(waveform, _Waveform)
        assert device == "cuda"
        return {"input_features": "features"}

    def batch_decode(self, predictions):
        assert predictions == ["tokens"]
        return ["recognized speech"]


class _Model:
    def to(self, device):
        assert device == "cuda"

    def eval(self):
        return None

    def transcribe(self, **inputs):
        assert inputs == {"input_features": "features"}
        return SimpleNamespace(preds=["tokens"])


def _backend():
    load_calls: list[tuple[str, str, dict[str, object]]] = []

    class AutoProcessor:
        @staticmethod
        def from_pretrained(path, **options):
            load_calls.append(("processor", path, options))
            return _Processor()

    class AutoModel:
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

    clock_values = iter([10.0, 10.25])
    backend = worker._Backend(
        AutoModel=AutoModel,
        AutoProcessor=AutoProcessor,
        clock=lambda: next(clock_values),
        torch=SimpleNamespace(
            bfloat16="bfloat16",
            cuda=Cuda,
            device=lambda value: value,
            inference_mode=nullcontext,
        ),
        torchaudio=SimpleNamespace(
            functional=SimpleNamespace(resample=lambda *args: None),
            load=lambda path: (_Waveform(), 16_000),
        ),
    )
    return backend, load_calls


def test_worker_verifies_source_before_loading_model(monkeypatch, tmp_path) -> None:
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
    assert [call[0] for call in load_calls] == ["processor", "model"]
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


def test_worker_requires_cuda(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"fixture")
    audit = SimpleNamespace(revision="a" * 40)
    monkeypatch.setattr(worker, "load_remote_code_audit", lambda model_id: audit)
    monkeypatch.setattr(worker, "verify_audited_files", lambda audit, root: None)
    backend, _ = _backend()
    backend.torch.cuda.is_available = lambda: False

    with pytest.raises(RuntimeError, match="requires CUDA"):
        worker.run_request(
            {
                "model_id": worker.MODEL_ID,
                "revision": "a" * 40,
                "snapshot_root": str(snapshot.resolve()),
                "wav_paths": [str(wav.resolve())],
            },
            backend,
        )
