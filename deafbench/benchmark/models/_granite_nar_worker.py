"""Child-process inference worker for audited Granite Speech NAR source."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from typing import Any

from deafbench.benchmark.models._isolated import RESULT_MARKER
from deafbench.remote_code_audit import (
    load_remote_code_audit,
    verify_audited_files,
)


MODEL_ID = "ibm-granite/granite-speech-4.1-2b-nar"
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class _Backend:
    AutoModel: Any
    AutoProcessor: Any
    clock: Any
    torch: Any
    torchaudio: Any


def _load_backend() -> _Backend:
    try:
        import torch
        import torchaudio
        from transformers import AutoModel, AutoProcessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Granite NAR is not installed. Run: "
            'python -m pip install "deafbench[granite-nar-asr]"'
        ) from exc
    return _Backend(AutoModel, AutoProcessor, perf_counter, torch, torchaudio)


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Granite NAR request requires {field}")
    return value


def _validated_wav_paths(payload: Mapping[str, Any]) -> tuple[Path, ...]:
    values = payload.get("wav_paths")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Granite NAR request requires WAV paths")
    paths: list[Path] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("Granite NAR request contains an invalid WAV path")
        path = Path(value)
        if not path.is_absolute() or path.suffix.lower() != ".wav":
            raise ValueError("Granite NAR request contains an unsafe WAV path")
        paths.append(path.resolve(strict=True))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Granite NAR request requires unique WAV paths")
    return tuple(paths)


def run_request(
    payload: Mapping[str, Any],
    backend: _Backend | None = None,
) -> dict[str, Any]:
    """Verify audited source and transcribe the requested local WAV files."""
    model_id = _required_text(payload, "model_id")
    revision = _required_text(payload, "revision")
    if model_id != MODEL_ID:
        raise ValueError(f"unexpected Granite NAR model: {model_id}")
    audit = load_remote_code_audit(model_id)
    if revision != audit.revision:
        raise ValueError("Granite NAR request revision differs from audit")
    snapshot_root = Path(_required_text(payload, "snapshot_root"))
    if not snapshot_root.is_absolute():
        raise ValueError("Granite NAR snapshot must be absolute")
    snapshot_root = snapshot_root.resolve(strict=True)
    verify_audited_files(audit, snapshot_root)
    wav_paths = _validated_wav_paths(payload)

    runtime = _load_backend() if backend is None else backend
    if not runtime.torch.cuda.is_available():
        raise RuntimeError("Granite NAR requires CUDA with FlashAttention 2")
    runtime.torch.cuda.reset_peak_memory_stats()
    device = runtime.torch.device("cuda")
    load_options = {
        "local_files_only": True,
        "trust_remote_code": True,
    }
    processor = runtime.AutoProcessor.from_pretrained(
        str(snapshot_root),
        **load_options,
    )
    model = runtime.AutoModel.from_pretrained(
        str(snapshot_root),
        attn_implementation="flash_attention_2",
        torch_dtype=runtime.torch.bfloat16,
        **load_options,
    )
    model.to(device)
    model.eval()

    records: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    total_audio_seconds = 0.0
    with runtime.torch.inference_mode():
        for wav_path in wav_paths:
            started = runtime.clock()
            waveform, source_rate = runtime.torchaudio.load(str(wav_path))
            if waveform.ndim != 2 or waveform.shape[0] != 1:
                raise ValueError(f"Granite NAR requires mono audio: {wav_path}")
            audio_seconds = waveform.shape[-1] / source_rate
            if source_rate != SAMPLE_RATE:
                waveform = runtime.torchaudio.functional.resample(
                    waveform,
                    source_rate,
                    SAMPLE_RATE,
                )
            inputs = processor(waveform.squeeze(0), device=str(device))
            output = model.transcribe(**inputs)
            decoded = processor.batch_decode(output.preds)
            if (
                not isinstance(decoded, list)
                or len(decoded) != 1
                or not isinstance(decoded[0], str)
            ):
                raise ValueError("Granite NAR returned an invalid transcription")
            latency_ms = round((runtime.clock() - started) * 1_000.0, 6)
            latencies_ms.append(latency_ms)
            total_audio_seconds += audio_seconds
            records.append(
                {"id": wav_path.stem, "latency_ms": latency_ms, "text": decoded[0]}
            )

    total_latency_seconds = sum(latencies_ms) / 1_000.0
    if total_latency_seconds <= 0:
        raise ValueError("Granite NAR timing must be positive")
    return {
        "decoding": {
            "attn_implementation": "flash_attention_2",
            "device": str(device),
            "dtype": str(runtime.torch.bfloat16),
            "trust_remote_code": True,
        },
        "performance": {
            "local_rtfx": total_audio_seconds / total_latency_seconds,
            "median_latency_ms": median(latencies_ms),
            "peak_vram_bytes": runtime.torch.cuda.max_memory_allocated(),
            "timing_scope": "decode_only_excludes_model_load",
        },
        "records": records,
    }


def main() -> int:
    """Read one JSON request from stdin and emit one machine-readable result."""
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("Granite NAR worker request must be an object")
    result = run_request(payload)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
