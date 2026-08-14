from types import SimpleNamespace
import builtins
import json
from pathlib import Path

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


def test_backend_reports_missing_onnxruntime(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(RuntimeError, match=r"deafbench\[ark-onnx-asr\]"):
        worker._load_backend()


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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "requires model_id"),
        ({"model_id": "wrong/model", "revision": "a"}, "unexpected"),
        (
            {
                "model_id": worker.MODEL_ID,
                "revision": "a" * 40,
                "snapshot_root": "relative",
            },
            "snapshot must be absolute",
        ),
    ],
)
def test_worker_rejects_invalid_request_fields(
    monkeypatch,
    payload,
    message,
) -> None:
    monkeypatch.setattr(
        worker,
        "load_remote_code_audit",
        lambda model_id: SimpleNamespace(revision="a" * 40),
    )

    with pytest.raises(ValueError, match=message):
        worker.run_request(payload)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ("sample.wav", "requires WAV paths"),
        ([1], "invalid WAV path"),
        (["relative.wav"], "unsafe WAV path"),
        ([], "unique WAV paths"),
    ],
)
def test_wav_path_validation_fails_closed(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        worker._validated_wav_paths({"wav_paths": values})


def test_wav_path_validation_rejects_duplicates(tmp_path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="unique WAV paths"):
        worker._validated_wav_paths({"wav_paths": [str(wav), str(wav)]})


def test_layout_requires_runtime_files_and_metadata(tmp_path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    with pytest.raises(ValueError, match="no runtime files"):
        worker._stage_official_layout(snapshot, runtime)

    (snapshot / "model.onnx").write_bytes(b"model")
    second_runtime = tmp_path / "second-runtime"
    second_runtime.mkdir()
    with pytest.raises(ValueError, match="omits model metadata"):
        worker._stage_official_layout(snapshot, second_runtime)


def test_layout_links_resolved_hugging_face_assets(
    monkeypatch,
    tmp_path,
) -> None:
    blob = tmp_path / "blobs" / "runtime_manifest.json"
    blob.parent.mkdir()
    blob.write_text('{"schema_version":1}', encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = snapshot / "runtime_manifest.json"
    source.symlink_to(Path("../blobs/runtime_manifest.json"))
    received: list[Path] = []

    def link(resolved_source: Path, destination: Path) -> None:
        received.append(resolved_source)
        destination.write_bytes(resolved_source.read_bytes())

    monkeypatch.setattr(worker.os, "link", link)
    worker._link_file(source, tmp_path / "staged.json")

    assert received == [blob.resolve()]


def test_official_module_loader_executes_pinned_script(tmp_path) -> None:
    script = tmp_path / "official.py"
    script.write_text("value = 42\n", encoding="utf-8")

    module = worker._import_official_module(script)

    assert module.value == 42


def test_worker_rejects_nonpositive_timing(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"fixture")
    audit = SimpleNamespace(revision="a" * 40)
    monkeypatch.setattr(worker, "load_remote_code_audit", lambda model_id: audit)
    monkeypatch.setattr(worker, "verify_audited_files", lambda audit, root: None)
    runtime = _Runtime()
    backend = _backend(runtime, [])
    backend = worker._Backend(
        clock=iter([10.0, 10.0]).__next__,
        load_runtime=backend.load_runtime,
        soundfile=backend.soundfile,
    )

    with pytest.raises(ValueError, match="timing must be positive"):
        worker.run_request(
            {
                "model_id": worker.MODEL_ID,
                "revision": "a" * 40,
                "snapshot_root": str(snapshot.resolve()),
                "wav_paths": [str(wav.resolve())],
            },
            backend,
        )


def test_main_emits_marked_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        worker,
        "json",
        SimpleNamespace(
            load=lambda stream: {"request": True},
            dumps=json.dumps,
        ),
    )
    monkeypatch.setattr(worker, "run_request", lambda payload: {"ok": True})

    assert worker.main() == 0
    assert capsys.readouterr().out == (
        worker.RESULT_MARKER + '{"ok":true}' + "\n"
    )


def test_main_rejects_nonobject_request(monkeypatch) -> None:
    monkeypatch.setattr(
        worker,
        "json",
        SimpleNamespace(load=lambda stream: [], dumps=json.dumps),
    )

    with pytest.raises(ValueError, match="must be an object"):
        worker.main()
