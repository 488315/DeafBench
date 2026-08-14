"""Child-process inference worker for audited ARK-ASR ONNX source."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from deafbench.benchmark.models._audio_chunks import contiguous_audio_chunks
from deafbench.benchmark.models._isolated import RESULT_MARKER
from deafbench.remote_code_audit import load_remote_code_audit, verify_audited_files


MODEL_ID = "AutoArk-AI/ark-asr-0.6b-int8-onnx"
MAX_AUDIO_SECONDS = 30.0
MAX_NEW_TOKENS = 256
PRECISION = "int8"
ASR_BLOCK_TOKEN_ID_FROM = 151_670


@dataclass(frozen=True)
class _Backend:
    clock: Callable[[], float]
    load_runtime: Callable[[Path, Path], Any]
    soundfile: Any


def _load_backend() -> _Backend:
    try:
        import onnxruntime  # noqa: F401
        import soundfile
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ARK-ASR ONNX is not installed. Run: "
            'python -m pip install "deafbench[ark-onnx-asr]"'
        ) from exc
    return _Backend(perf_counter, _load_official_runtime, soundfile)


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ARK-ASR ONNX request requires {field}")
    return value


def _validated_wav_paths(payload: Mapping[str, Any]) -> tuple[Path, ...]:
    values = payload.get("wav_paths")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("ARK-ASR ONNX request requires WAV paths")
    paths: list[Path] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("ARK-ASR ONNX request contains an invalid WAV path")
        path = Path(value)
        if not path.is_absolute() or path.suffix.lower() != ".wav":
            raise ValueError("ARK-ASR ONNX request contains an unsafe WAV path")
        paths.append(path.resolve(strict=True))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("ARK-ASR ONNX request requires unique WAV paths")
    return tuple(paths)


def _link_file(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    try:
        os.link(source, destination)
    except OSError:
        destination.symlink_to(source)


def _stage_official_layout(snapshot_root: Path, runtime_root: Path) -> None:
    model_dir = runtime_root / "model"
    build_dir = runtime_root / "build"
    model_dir.mkdir()
    build_dir.mkdir()
    files = tuple(path for path in snapshot_root.iterdir() if path.is_file())
    if not files:
        raise ValueError("ARK-ASR ONNX snapshot contains no runtime files")
    for source in files:
        _link_file(source, model_dir / source.name)
    metadata = snapshot_root / "llm_kv_fp32_qwen_native.json"
    if not metadata.is_file():
        raise ValueError("ARK-ASR ONNX snapshot omits model metadata")
    _link_file(metadata, build_dir / metadata.name)


def _import_official_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_deafbench_audited_ark_asr_onnx",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load audited ARK-ASR ONNX source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_official_runtime(snapshot_root: Path, runtime_root: Path) -> Any:
    _stage_official_layout(snapshot_root, runtime_root)
    module = _import_official_module(snapshot_root / "infer_ark_audio_onnx.py")

    def load_cpu_session(model_path: Path) -> Any:
        options = module.ort.SessionOptions()
        options.graph_optimization_level = (
            module.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        return module.ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    module.load_session = load_cpu_session
    return module.ArkAsrOnnxRuntime(runtime_root)


def _transcribe_files(
    runtime: Any,
    wav_paths: Sequence[Path],
    backend: _Backend,
) -> tuple[list[dict[str, object]], list[float], float]:
    records: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    total_audio_seconds = 0.0
    for wav_path in wav_paths:
        with contiguous_audio_chunks(
            wav_path,
            backend.soundfile,
            max_audio_seconds=MAX_AUDIO_SECONDS,
            runtime_name="ARK-ASR ONNX",
        ) as (audio_seconds, chunk_paths):
            started = backend.clock()
            chunk_texts = [
                runtime.transcribe(
                    audio_path=str(chunk_path),
                    max_new_tokens=MAX_NEW_TOKENS,
                    max_audio_seconds=int(MAX_AUDIO_SECONDS),
                    precision=PRECISION,
                    asr_block_token_id_from=ASR_BLOCK_TOKEN_ID_FROM,
                ).strip()
                for chunk_path in chunk_paths
            ]
            latency_ms = round((backend.clock() - started) * 1_000.0, 6)
        latencies_ms.append(latency_ms)
        total_audio_seconds += audio_seconds
        records.append(
            {
                "id": wav_path.stem,
                "latency_ms": latency_ms,
                "text": " ".join(text for text in chunk_texts if text),
            }
        )
    return records, latencies_ms, total_audio_seconds


def run_request(
    payload: Mapping[str, Any],
    backend: _Backend | None = None,
) -> dict[str, Any]:
    """Verify audited source and transcribe local WAV files on CPU."""
    model_id = _required_text(payload, "model_id")
    revision = _required_text(payload, "revision")
    if model_id != MODEL_ID:
        raise ValueError(f"unexpected ARK-ASR ONNX model: {model_id}")
    audit = load_remote_code_audit(model_id)
    if revision != audit.revision:
        raise ValueError("ARK-ASR ONNX request revision differs from audit")
    snapshot_root = Path(_required_text(payload, "snapshot_root"))
    if not snapshot_root.is_absolute():
        raise ValueError("ARK-ASR ONNX snapshot must be absolute")
    snapshot_root = snapshot_root.resolve(strict=True)
    verify_audited_files(audit, snapshot_root)
    wav_paths = _validated_wav_paths(payload)
    runtime_backend = _load_backend() if backend is None else backend

    with TemporaryDirectory(prefix="deafbench-ark-onnx-") as directory:
        runtime = runtime_backend.load_runtime(snapshot_root, Path(directory))
        records, latencies_ms, total_audio_seconds = _transcribe_files(
            runtime,
            wav_paths,
            runtime_backend,
        )

    total_latency_seconds = sum(latencies_ms) / 1_000.0
    if total_latency_seconds <= 0:
        raise ValueError("ARK-ASR ONNX timing must be positive")
    return {
        "decoding": {
            "asr_block_token_id_from": ASR_BLOCK_TOKEN_ID_FROM,
            "audio_max_seconds": MAX_AUDIO_SECONDS,
            "execution_provider": "CPUExecutionProvider",
            "long_audio_strategy": "contiguous_30_second_chunks_without_overlap",
            "max_new_tokens": MAX_NEW_TOKENS,
            "precision": PRECISION,
            "trust_remote_code": True,
        },
        "performance": {
            "local_rtfx": total_audio_seconds / total_latency_seconds,
            "median_latency_ms": median(latencies_ms),
            "peak_vram_bytes": 0,
            "timing_scope": "decode_only_excludes_model_load",
        },
        "records": records,
    }


def main() -> int:
    """Read one JSON request from stdin and emit one machine-readable result."""
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("ARK-ASR ONNX worker request must be an object")
    result = run_request(payload)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
