from types import SimpleNamespace

import pytest

from deafbench.benchmark.models import _ark_asr_onnx_worker as worker


class _Runtime:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def transcribe(self, **options):
        self.calls.append(options)
        return "recognized speech"


def _backend(runtime, loaded):
    clock_values = iter([10.0, 10.25])

    def load_runtime(snapshot_root, runtime_root):
        loaded.append((snapshot_root, runtime_root))
        return runtime

    return worker._Backend(
        clock=lambda: next(clock_values),
        load_runtime=load_runtime,
        soundfile=SimpleNamespace(
            info=lambda path: SimpleNamespace(
                channels=1,
                duration=1.0,
                samplerate=16_000,
            )
        ),
    )


def test_worker_verifies_source_and_uses_official_contract(
    monkeypatch,
    tmp_path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"fixture")
    order: list[str] = []
    loaded: list[object] = []
    audit = SimpleNamespace(revision="a" * 40)
    monkeypatch.setattr(worker, "load_remote_code_audit", lambda model_id: audit)
    monkeypatch.setattr(
        worker,
        "verify_audited_files",
        lambda received, root: order.append("verify"),
    )
    runtime = _Runtime()

    result = worker.run_request(
        {
            "model_id": worker.MODEL_ID,
            "revision": "a" * 40,
            "snapshot_root": str(snapshot.resolve()),
            "wav_paths": [str(wav.resolve())],
        },
        _backend(runtime, loaded),
    )

    assert order == ["verify"]
    assert loaded[0][0] == snapshot.resolve()
    assert runtime.calls == [
        {
            "audio_path": str(wav.resolve()),
            "max_new_tokens": 256,
            "max_audio_seconds": 30,
            "precision": "int8",
            "asr_block_token_id_from": 151_670,
        }
    ]
    assert result["records"] == [
        {"id": "sample", "latency_ms": 250.0, "text": "recognized speech"}
    ]
    assert result["performance"] == {
        "local_rtfx": 4.0,
        "median_latency_ms": 250.0,
        "peak_vram_bytes": 0,
        "timing_scope": "decode_only_excludes_model_load",
    }
    assert result["decoding"]["execution_provider"] == "CPUExecutionProvider"


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


def test_official_runtime_forces_cpu_provider(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "llm_kv_fp32_qwen_native.json").write_text("{}")
    (snapshot / "infer_ark_audio_onnx.py").write_text("# audited")
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    session_providers: list[list[str]] = []

    class SessionOptions:
        graph_optimization_level = None

    class InferenceSession:
        def __init__(self, path, *, sess_options, providers: list[str]):
            assert sess_options.graph_optimization_level == "all"
            self.path = path
            session_providers.append(list(providers))

    ort = SimpleNamespace(
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=InferenceSession,
        SessionOptions=SessionOptions,
    )

    class ArkAsrOnnxRuntime:
        def __init__(self, root):
            assert root == runtime_root
            module.load_session(root / "model" / "embedding.onnx")

    module = SimpleNamespace(ArkAsrOnnxRuntime=ArkAsrOnnxRuntime, ort=ort)
    monkeypatch.setattr(worker, "_import_official_module", lambda path: module)

    worker._load_official_runtime(snapshot, runtime_root)

    assert session_providers == [["CPUExecutionProvider"]]
    assert (runtime_root / "model" / "infer_ark_audio_onnx.py").exists()
    assert (runtime_root / "build" / "llm_kv_fp32_qwen_native.json").exists()
