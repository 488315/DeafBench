"""Pinned parent-process adapter for audited ARK-ASR ONNX inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from deafbench.benchmark.models import ModelRunInfo, _validated_wavs
from deafbench.benchmark.models._isolated import invoke_isolated_worker
from deafbench.benchmark.models._isolated_result import (
    required_mapping,
    validated_records,
)
from deafbench.benchmark.workspace import atomic_write_jsonl
from deafbench.model_registry import ModelRegistryError, get_model_license
from deafbench.remote_code_audit import load_remote_code_audit, verify_audited_files


MODEL_ID = "AutoArk-AI/ark-asr-0.6b-int8-onnx"
MODEL_NAME = "ark-asr-0.6b-int8-onnx"
WORKER_MODULE = "deafbench.benchmark.models._ark_asr_onnx_worker"
SnapshotDownload = Callable[..., str]
WorkerInvoker = Callable[[str, Mapping[str, Any]], dict[str, Any]]


def _load_snapshot_downloader() -> SnapshotDownload:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ARK-ASR ONNX is not installed. Run: "
            'python -m pip install "deafbench[ark-onnx-asr]"'
        ) from exc
    return snapshot_download


def _licensed_revision() -> str:
    model_license = get_model_license(MODEL_ID)
    if not model_license.remote_code_required:
        raise ModelRegistryError("ARK-ASR ONNX must remain a remote-code model")
    audit = load_remote_code_audit(MODEL_ID)
    if audit.revision != model_license.revision:
        raise ModelRegistryError("ARK-ASR ONNX audit revision differs from registry")
    return model_license.revision


def _validated_records(
    payload: object,
    expected_ids: Sequence[str],
) -> list[dict[str, object]]:
    return validated_records(payload, expected_ids, worker_name="ARK-ASR ONNX")


def _required_mapping(payload: Mapping[str, Any], field: str) -> Mapping[str, object]:
    return required_mapping(payload, field, worker_name="ARK-ASR ONNX")


def run_ark_asr_onnx(
    audio_dir: Path,
    references: Path,
    output: Path,
    *,
    snapshot_root: Path | None = None,
    snapshot_download: SnapshotDownload | None = None,
    worker_invoker: WorkerInvoker = invoke_isolated_worker,
) -> ModelRunInfo:
    """Transcribe a complete WAV set through the audited ONNX worker."""
    revision = _licensed_revision()
    wav_paths = _validated_wavs(audio_dir, references)
    if snapshot_root is None:
        downloader = snapshot_download or _load_snapshot_downloader()
        snapshot_root = Path(downloader(repo_id=MODEL_ID, revision=revision))
    snapshot_root = snapshot_root.resolve(strict=True)
    audit = load_remote_code_audit(MODEL_ID)
    verify_audited_files(audit, snapshot_root)

    result = worker_invoker(
        WORKER_MODULE,
        {
            "model_id": MODEL_ID,
            "revision": revision,
            "snapshot_root": str(snapshot_root),
            "wav_paths": [str(path.resolve()) for path in wav_paths],
        },
    )
    records = _validated_records(
        result.get("records"),
        [path.stem for path in wav_paths],
    )
    decoding = _required_mapping(result, "decoding")
    performance = _required_mapping(result, "performance")
    atomic_write_jsonl(output, records)
    return ModelRunInfo(
        name=MODEL_NAME,
        model_id=MODEL_ID,
        revision=revision,
        decoding=dict(decoding),
        performance=dict(performance),
    )
