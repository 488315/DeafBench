"""Customer-run evaluation that keeps sample evidence inside one local case."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable, Mapping

from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.workspace import (
    atomic_write_text,
    inspect_audio_set,
    load_reference_records,
)
from deafbench.metrics import evaluate_dataset
from deafbench.model_registry import get_model_license
from deafbench.parser import align_records, parse_jsonl
from deafbench.pilot.authorization import load_authorization
from deafbench.pilot.workspace import discover_git_worktrees, validate_case_root
from deafbench.result_manifest import canonical_result_bytes


ModelRunner = Callable[[Path, Path, Path], ModelRunInfo]
PILOT_MODEL_RUNNERS = (
    "Qwen/Qwen3-ASR-1.7B-hf",
    "nvidia/parakeet-tdt-0.6b-v2",
    "ibm-granite/granite-speech-4.1-2b",
)
_MODEL_SLUGS = {
    "Qwen/Qwen3-ASR-1.7B-hf": "qwen3-asr-1.7b",
    "nvidia/parakeet-tdt-0.6b-v2": "parakeet-tdt-0.6b-v2",
    "ibm-granite/granite-speech-4.1-2b": "granite-speech-4.1-2b",
}


@dataclass(frozen=True)
class CustomerAuditResult:
    """Local validated evidence produced by the three-model pilot run."""

    result_paths: tuple[Path, ...]
    sample_count: int
    prediction_paths: tuple[Path, ...] = ()


def _default_runners(
    on_sample_complete: Callable[[str, Path], None] | None = None,
) -> Mapping[str, ModelRunner]:
    from deafbench.benchmark.models.granite_speech import run_granite_speech
    from deafbench.benchmark.models.parakeet import run_parakeet
    from deafbench.benchmark.models.qwen3_asr import run_qwen3_asr_1_7b

    if on_sample_complete is None:
        return {
            PILOT_MODEL_RUNNERS[0]: run_qwen3_asr_1_7b,
            PILOT_MODEL_RUNNERS[1]: run_parakeet,
            PILOT_MODEL_RUNNERS[2]: run_granite_speech,
        }

    def qwen(audio: Path, references: Path, output: Path) -> ModelRunInfo:
        return run_qwen3_asr_1_7b(
            audio,
            references,
            output,
            progress=lambda path: on_sample_complete(PILOT_MODEL_RUNNERS[0], path),
        )

    def parakeet(audio: Path, references: Path, output: Path) -> ModelRunInfo:
        return run_parakeet(
            audio,
            references,
            output,
            progress=lambda path: on_sample_complete(PILOT_MODEL_RUNNERS[1], path),
        )

    def granite(audio: Path, references: Path, output: Path) -> ModelRunInfo:
        return run_granite_speech(
            audio,
            references,
            output,
            progress=lambda path: on_sample_complete(PILOT_MODEL_RUNNERS[2], path),
        )

    return {
        PILOT_MODEL_RUNNERS[0]: qwen,
        PILOT_MODEL_RUNNERS[1]: parakeet,
        PILOT_MODEL_RUNNERS[2]: granite,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus_fingerprint(references: Path, audio_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(_sha256(references)))
    for audio in sorted(audio_dir.glob("*.wav"), key=lambda path: path.name):
        digest.update(audio.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(audio)))
    return digest.hexdigest()


def _evaluator_fingerprint() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in ("critical_entities.py", "metrics.py"):
        digest.update((package_root / name).read_bytes())
    return digest.hexdigest()


def _performance(info: ModelRunInfo) -> dict[str, Real]:
    performance = info.performance
    required = ("local_rtfx", "median_latency_ms", "peak_vram_bytes")
    if performance is None or any(
        not isinstance(performance.get(field), Real) for field in required
    ):
        raise ValueError("model run did not report complete local performance")
    return {field: performance[field] for field in required}


def _critical_failures(
    metrics: Mapping[str, object], references: tuple[Mapping[str, object], ...]
) -> list[dict[str, str]]:
    by_id = {str(reference["id"]): reference for reference in references}
    failures: list[dict[str, str]] = []
    for failure in metrics["critical_failures"]:
        sample_id = str(failure["id"])
        term = str(failure["expected"])
        types = by_id[sample_id].get("critical_types", {})
        failures.append(
            {
                "id": sample_id,
                "term": term,
                "entity_type": str(types.get(term, "UNCLASSIFIED")),
            }
        )
    return failures


def _result_manifest(
    *,
    info: ModelRunInfo,
    metrics: Mapping[str, object],
    references: tuple[Mapping[str, object], ...],
    corpus_fingerprint: str,
) -> dict[str, object]:
    registry = get_model_license(info.model_id)
    if info.revision != registry.revision:
        raise ValueError("model runner revision differs from the license registry")
    performance = _performance(info)
    return {
        "schema_version": 1,
        "status": "customer_audit_complete",
        "model": {"model_id": info.model_id, "revision": info.revision},
        "license_classification": registry.intended_lane,
        "evaluator_revision": _evaluator_fingerprint(),
        "decoding": dict(info.decoding or {}),
        "corpora": [
            {
                "name": "customer-authorized-audio",
                "manifest_sha256": corpus_fingerprint,
                "frozen": True,
            }
        ],
        "evaluations": [
            {
                "lane": "customer-audit",
                "scope": "complete",
                "sample_count": len(references),
                "metrics": {
                    "wer_percent": metrics["wer"],
                    "strict_lexical_recall_percent": metrics[
                        "strict_critical_recall"
                    ],
                    "canonical_semantic_recall_percent": metrics[
                        "canonical_critical_recall"
                    ],
                    "substitutions": metrics["substitutions"],
                    "insertions": metrics["insertions"],
                    "deletions": metrics["deletions"],
                    **performance,
                },
                "critical_failures": _critical_failures(metrics, references),
            }
        ],
        "claim_boundary": (
            "Customer-executed and environment-dependent local audit; not a "
            "certification or Hugging Face leaderboard result."
        ),
        "verification": {
            "status": "recorded_local_observation",
            "sample_artifacts_in_repository": False,
            "independently_recomputable_from_checkout": False,
        },
    }


def run_customer_audit(
    *,
    repo_root: Path,
    case_root: Path,
    runners: Mapping[str, ModelRunner] | None = None,
    authorization_path: Path | None = None,
    expected_case_id: str | None = None,
    references_path: Path | None = None,
    audio_dir: Path | None = None,
    work_dir: Path | None = None,
    on_model_start: Callable[[str, int], None] | None = None,
    on_sample_complete: Callable[[str, Path], None] | None = None,
    on_model_complete: Callable[[str], None] | None = None,
) -> CustomerAuditResult:
    """Evaluate the authorized local case and retain sample data inside it."""
    try:
        worktrees = discover_git_worktrees(repo_root)
    except (OSError, subprocess.CalledProcessError):
        worktrees = ()
    root = validate_case_root(case_root, worktrees=worktrees)
    authorization = load_authorization(
        authorization_path or root / "authorization.json",
        expected_case_id=expected_case_id or root.name,
    )
    if not set(PILOT_MODEL_RUNNERS).issubset(authorization.permitted_models):
        raise ValueError("authorization does not permit every pilot model")
    references_path = Path(references_path or root / "input" / "references.jsonl")
    audio_dir = Path(audio_dir or root / "input" / "audio")
    status = inspect_audio_set(references_path, audio_dir)
    if not status.complete:
        raise ValueError("customer audit requires a complete valid audio set")
    references = load_reference_records(references_path)
    initial_fingerprint = _corpus_fingerprint(references_path, audio_dir)
    resolved_work_dir = Path(work_dir or root / "work")
    results_dir = resolved_work_dir / "results"
    if results_dir.exists():
        raise FileExistsError("customer audit results already exist")
    results_dir.mkdir(parents=True)

    selected = dict(
        _default_runners(on_sample_complete) if runners is None else runners
    )
    if set(selected) != set(PILOT_MODEL_RUNNERS):
        raise ValueError("customer audit requires the exact three-model pilot set")
    result_paths: list[Path] = []
    prediction_paths: list[Path] = []
    for model_id in PILOT_MODEL_RUNNERS:
        slug = _MODEL_SLUGS[model_id]
        predictions = resolved_work_dir / slug / "predictions.jsonl"
        if on_model_start is not None:
            on_model_start(model_id, len(references))
        info = selected[model_id](audio_dir, references_path, predictions)
        if _corpus_fingerprint(references_path, audio_dir) != initial_fingerprint:
            raise ValueError("customer audit input changed during evaluation")
        metrics = evaluate_dataset(
            align_records(list(references), parse_jsonl(str(predictions)))
        )
        manifest = _result_manifest(
            info=info,
            metrics=metrics,
            references=references,
            corpus_fingerprint=initial_fingerprint,
        )
        result_path = results_dir / f"{slug}.json"
        atomic_write_text(result_path, canonical_result_bytes(manifest).decode())
        result_paths.append(result_path)
        prediction_paths.append(predictions)
        if on_model_complete is not None:
            on_model_complete(model_id)
    return CustomerAuditResult(
        tuple(result_paths),
        len(references),
        tuple(prediction_paths),
    )
