"""Customer-run evaluation that keeps sample evidence inside one local case."""

from __future__ import annotations

import hashlib
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


def _default_runners() -> Mapping[str, ModelRunner]:
    from deafbench.benchmark.models.granite_speech import run_granite_speech
    from deafbench.benchmark.models.parakeet import run_parakeet
    from deafbench.benchmark.models.qwen3_asr import run_qwen3_asr_1_7b

    return {
        PILOT_MODEL_RUNNERS[0]: run_qwen3_asr_1_7b,
        PILOT_MODEL_RUNNERS[1]: run_parakeet,
        PILOT_MODEL_RUNNERS[2]: run_granite_speech,
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
    }


def run_customer_audit(
    *,
    repo_root: Path,
    case_root: Path,
    runners: Mapping[str, ModelRunner] | None = None,
) -> CustomerAuditResult:
    """Evaluate the authorized local case and retain sample data inside it."""
    root = validate_case_root(
        case_root,
        worktrees=discover_git_worktrees(repo_root),
    )
    references_path = root / "input" / "references.jsonl"
    audio_dir = root / "input" / "audio"
    status = inspect_audio_set(references_path, audio_dir)
    if not status.complete:
        raise ValueError("customer audit requires a complete valid audio set")
    references = load_reference_records(references_path)
    initial_fingerprint = _corpus_fingerprint(references_path, audio_dir)
    results_dir = root / "work" / "results"
    if results_dir.exists():
        raise FileExistsError("customer audit results already exist")
    results_dir.mkdir(parents=True)

    selected = dict(_default_runners() if runners is None else runners)
    if set(selected) != set(PILOT_MODEL_RUNNERS):
        raise ValueError("customer audit requires the exact three-model pilot set")
    result_paths: list[Path] = []
    for model_id in PILOT_MODEL_RUNNERS:
        slug = _MODEL_SLUGS[model_id]
        predictions = root / "work" / slug / "predictions.jsonl"
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
    return CustomerAuditResult(tuple(result_paths), len(references))
