"""Pinned NeMo adapter for NVIDIA Parakeet TDT 0.6B v2."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from deafbench.benchmark.models import ModelRunInfo, _validated_wavs
from deafbench.benchmark.workspace import atomic_write_jsonl
from deafbench.model_registry import ModelRegistryError, get_model_license


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
MODEL_NAME = "parakeet-tdt-0.6b-v2"
ARCHIVE_NAME = "parakeet-tdt-0.6b-v2.nemo"


@dataclass(frozen=True)
class _NeMoBackend:
    ASRModel: Any
    clock: Any
    hf_hub_download: Any
    torch: Any


def _load_backend() -> _NeMoBackend:
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from nemo.collections.asr.models import ASRModel
    except ModuleNotFoundError as exc:
        if exc.name not in {"huggingface_hub", "nemo", "torch"}:
            raise
        raise RuntimeError(
            "Parakeet ASR is not installed. Run: "
            'python -m pip install "deafbench[parakeet-asr]"'
        ) from exc
    return _NeMoBackend(ASRModel, perf_counter, hf_hub_download, torch)


def _licensed_revision() -> str:
    model_license = get_model_license(MODEL_ID)
    if model_license.remote_code_required:
        raise ModelRegistryError("Parakeet adapter rejects remote code")
    if not any(
        runtime.startswith("nemo_toolkit")
        for runtime in model_license.supported_runtimes
    ):
        raise ModelRegistryError("Parakeet adapter requires a registered NeMo runtime")
    return model_license.revision


def _duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _transcription_text(result: object) -> str:
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError("Invalid Parakeet transcription output")
    hypothesis = result[0]
    text = (
        hypothesis if isinstance(hypothesis, str) else getattr(hypothesis, "text", None)
    )
    if not isinstance(text, str):
        raise ValueError("Invalid Parakeet transcription output")
    return text


def run_parakeet(
    audio_dir: Path,
    references: Path,
    output: Path,
    backend: Any | None = None,
) -> ModelRunInfo:
    """Transcribe a complete WAV set from a revision-pinned NeMo archive."""
    revision = _licensed_revision()
    wav_paths = _validated_wavs(audio_dir, references)
    runtime = _load_backend() if backend is None else backend
    use_cuda = runtime.torch.cuda.is_available()
    if use_cuda:
        runtime.torch.cuda.reset_peak_memory_stats()
    device = runtime.torch.device("cuda" if use_cuda else "cpu")
    archive = runtime.hf_hub_download(
        repo_id=MODEL_ID,
        filename=ARCHIVE_NAME,
        revision=revision,
    )
    model = runtime.ASRModel.restore_from(
        restore_path=archive,
        map_location=device,
    )
    model.to(device)
    model.eval()

    records: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    total_audio_seconds = 0.0
    with runtime.torch.inference_mode():
        for wav_path in wav_paths:
            started = runtime.clock()
            result = model.transcribe(
                [str(wav_path)],
                batch_size=1,
                timestamps=True,
            )
            latency_ms = round((runtime.clock() - started) * 1_000.0, 6)
            latencies_ms.append(latency_ms)
            total_audio_seconds += _duration_seconds(wav_path)
            records.append(
                {
                    "id": wav_path.stem,
                    "latency_ms": latency_ms,
                    "text": _transcription_text(result),
                }
            )

    total_latency_seconds = sum(latencies_ms) / 1_000.0
    if not math.isfinite(total_latency_seconds) or total_latency_seconds <= 0:
        raise ValueError("Parakeet timing must be positive and finite")
    atomic_write_jsonl(output, records)
    return ModelRunInfo(
        name=MODEL_NAME,
        model_id=MODEL_ID,
        revision=revision,
        decoding={
            "archive": ARCHIVE_NAME,
            "batch_size": 1,
            "device": str(device),
            "timestamps": True,
        },
        performance={
            "local_rtfx": total_audio_seconds / total_latency_seconds,
            "median_latency_ms": median(latencies_ms),
            "peak_vram_bytes": (
                runtime.torch.cuda.max_memory_allocated() if use_cuda else None
            ),
            "timing_scope": "decode_only_excludes_model_load",
        },
    )
