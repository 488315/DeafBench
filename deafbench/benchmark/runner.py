"""Transactional orchestration for complete installed benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from deafbench import __version__
from deafbench.benchmark.models import MODEL_NAMES, ModelRunInfo
from deafbench.benchmark.workspace import (
    AudioSource,
    ResolvedAudioSource,
    RunPaths,
    atomic_write_json,
    atomic_write_text,
    inspect_audio_set,
    load_reference_records,
    resolve_audio_source,
    resolve_run_paths,
)
from deafbench.metrics import evaluate_dataset
from deafbench.parser import align_records, parse_jsonl
from deafbench.recorder.workspace import ensure_dataset_workspace
from deafbench.report import generate_markdown_report

if TYPE_CHECKING:
    from deafbench.benchmark.synthetic import SpeechGenerator, TTSInfo


ModelName = Literal[
    "whisper",
    "whisper-at",
    "faster-whisper",
    "distil-whisper",
    "qwen3-asr-0.6b",
    "qwen3-asr-1.7b",
    "parakeet-tdt-0.6b-v2",
]
SyntheticFactory = Callable[[], tuple["SpeechGenerator", "TTSInfo"]]
SyntheticGenerator = Callable[..., Path]
ModelRunner = Callable[[Path, Path, Path], ModelRunInfo]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inputs that identify one reproducible benchmark run."""

    repo_root: Path
    dataset: str
    model: ModelName
    audio_source: AudioSource = "auto"
    scene_profile: str = "default-v1"
    seed: int = 42


@dataclass(frozen=True)
class BenchmarkResult:
    """Promoted artifact paths and metrics from one completed run."""

    resolved_source: ResolvedAudioSource
    predictions: Path
    report: Path
    metadata: Path
    metrics: dict[str, Any]


def _validate_config(config: BenchmarkConfig) -> None:
    if config.model not in MODEL_NAMES:
        raise ValueError(f"Unsupported benchmark model: {config.model}")
    if config.audio_source not in ("auto", "human", "synthetic"):
        raise ValueError(f"Unsupported audio source: {config.audio_source}")


def _load_cached_tts(audio_dir: Path) -> TTSInfo:
    from deafbench.benchmark.synthetic import TTSInfo

    manifest = audio_dir / "manifest.jsonl"
    with manifest.open("r", encoding="utf-8") as handle:
        first_line = next((line for line in handle if line.strip()), "")
    record = json.loads(first_line)
    tts = record["tts"]
    return TTSInfo(cast(str, tts["engine"]), cast(str, tts["version"]))


def _load_validated_v2_tts(paths: RunPaths) -> TTSInfo | None:
    """Recognize an admitted v2 corpus without invoking a mutable TTS cache."""
    from deafbench.benchmark.synthetic import TTSInfo

    generation_path = paths.dataset_dir / "generation-manifest.jsonl"
    quality_path = paths.dataset_dir / "quality-report.json"
    if not generation_path.is_file() and not quality_path.is_file():
        return None
    if not generation_path.is_file() or not quality_path.is_file():
        raise ValueError("Synthetic-v2 admission evidence is incomplete")

    records = [
        json.loads(line)
        for line in generation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_ids = {
        cast(str, record["id"])
        for record in load_reference_records(paths.references)
    }
    actual_ids = {record.get("id") for record in records}
    if actual_ids != expected_ids or len(records) != len(actual_ids):
        raise ValueError("Synthetic-v2 generation manifest is incomplete")
    for record in records:
        sample_id = cast(str, record["id"])
        audio_path = paths.synthetic_audio / f"{sample_id}.wav"
        expected_hash = record.get("audio_sha256")
        if (
            not isinstance(expected_hash, str)
            or not audio_path.is_file()
            or hashlib.sha256(audio_path.read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError(f"Synthetic-v2 audio hash mismatch: {sample_id}")

    replacement_ids = {
        cast(str, record["id"])
        for record in records
        if record.get("replacement_reason") is not None
    }
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    samples = quality.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Synthetic-v2 quality report is invalid")
    accepted_ids = {
        sample.get("id")
        for sample in samples
        if isinstance(sample, dict) and sample.get("status") == "accepted"
    }
    if len(samples) != len(accepted_ids) or accepted_ids != replacement_ids:
        raise ValueError("Synthetic-v2 contains unaccepted replacements")

    revision = hashlib.sha256(generation_path.read_bytes()).hexdigest()
    return TTSInfo("validated-synthetic-v2", f"manifest-sha256:{revision}")


def _prepare_synthetic(
    config: BenchmarkConfig,
    paths: RunPaths,
    synthetic_factory: SyntheticFactory,
    synthetic_generator: SyntheticGenerator,
) -> TTSInfo:
    from deafbench.benchmark.synthetic import synthetic_set_is_current

    validated_v2 = _load_validated_v2_tts(paths)
    if validated_v2 is not None:
        return validated_v2

    if synthetic_set_is_current(
        paths.synthetic_audio,
        paths.references,
        config.scene_profile,
        config.seed,
    ):
        return _load_cached_tts(paths.synthetic_audio)

    speech_generator, tts_info = synthetic_factory()
    synthetic_generator(
        paths.references,
        paths.synthetic_audio,
        speech_generator,
        tts_info,
        scene_profile=config.scene_profile,
        seed=config.seed,
    )
    return tts_info


def _require_complete_audio(
    references: Path,
    audio_dir: Path,
    source: ResolvedAudioSource,
) -> None:
    status = inspect_audio_set(references, audio_dir)
    if not status.complete:
        raise ValueError(
            f"Selected {source} audio set is incomplete: "
            f"missing={list(status.missing)}; extra={list(status.extra)}; "
            f"invalid={list(status.invalid)}"
        )


def _default_model_runner(model: ModelName) -> ModelRunner:
    if model == "whisper":
        from deafbench.benchmark.models.whisper import run_whisper

        return run_whisper
    if model == "whisper-at":
        from deafbench.benchmark.models.whisper_at import run_whisper_at

        return run_whisper_at
    if model == "faster-whisper":
        from deafbench.benchmark.models.faster_whisper import run_faster_whisper

        return run_faster_whisper
    if model == "qwen3-asr-0.6b":
        from deafbench.benchmark.models.qwen3_asr import run_qwen3_asr

        return run_qwen3_asr
    if model == "qwen3-asr-1.7b":
        from deafbench.benchmark.models.qwen3_asr import run_qwen3_asr_1_7b

        return run_qwen3_asr_1_7b
    if model == "parakeet-tdt-0.6b-v2":
        from deafbench.benchmark.models.parakeet import run_parakeet

        return run_parakeet
    from deafbench.benchmark.models.distil_whisper import run_distil_whisper

    return run_distil_whisper


def _evaluate(
    references: Path,
    predictions: Path,
) -> dict[str, Any]:
    reference_records = parse_jsonl(str(references))
    prediction_records = parse_jsonl(str(predictions))
    expected_ids = {cast(str, record["id"]) for record in reference_records}
    actual_ids: set[str] = set()
    for record in prediction_records:
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Invalid prediction ID: expected a non-empty string")
        if sample_id in actual_ids:
            raise ValueError(f"Duplicate prediction ID: {sample_id}")
        actual_ids.add(sample_id)
    if actual_ids != expected_ids:
        raise ValueError(
            "Prediction IDs do not match references: "
            f"missing={sorted(expected_ids - actual_ids)}; "
            f"extra={sorted(actual_ids - expected_ids)}"
        )
    aligned = align_records(reference_records, prediction_records)
    return evaluate_dataset(aligned)


def _metadata(
    config: BenchmarkConfig,
    paths: RunPaths,
    source: ResolvedAudioSource,
    audio_dir: Path,
    model_info: ModelRunInfo,
    sample_count: int,
    tts_info: TTSInfo | None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "dataset": config.dataset,
        "model": config.model,
        "model_id": model_info.model_id,
        "audio_source": source,
        "references": str(paths.references),
        "audio": str(audio_dir),
        "predictions": str(paths.predictions),
        "report": str(paths.report),
        "samples": sample_count,
        "benchmark_version": __version__,
    }
    if model_info.revision is not None:
        value["model_revision"] = model_info.revision
    if model_info.decoding is not None:
        value["decoding"] = dict(model_info.decoding)
    if model_info.performance is not None:
        value["performance"] = dict(model_info.performance)
    if source == "synthetic":
        if tts_info is None:
            raise ValueError("Synthetic run is missing TTS provenance")
        value.update(
            {
                "scene_profile": config.scene_profile,
                "seed": config.seed,
                "tts": {
                    "engine": tts_info.engine,
                    "version": tts_info.version,
                },
            }
        )
    return value


def _promote_directory(staging: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}-backup")
    if backup.exists():
        if destination.exists():
            shutil.rmtree(backup)
        else:
            os.replace(backup, destination)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError:
                pass


def _build_run_bundle(
    config: BenchmarkConfig,
    paths: RunPaths,
    source: ResolvedAudioSource,
    audio_dir: Path,
    tts_info: TTSInfo | None,
    model_runner: ModelRunner,
) -> dict[str, Any]:
    paths.run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{paths.run_dir.name}-run-",
            dir=paths.run_dir.parent,
        )
    )
    promoted = False
    try:
        staging_predictions = staging / "predictions.jsonl"
        model_info = model_runner(
            audio_dir,
            paths.references,
            staging_predictions,
        )
        metrics = _evaluate(paths.references, staging_predictions)
        report = generate_markdown_report(
            metrics,
            str(paths.references),
            str(paths.predictions),
        )
        atomic_write_text(staging / "report.md", report)
        metadata = _metadata(
            config,
            paths,
            source,
            audio_dir,
            model_info,
            cast(int, metrics["samples"]),
            tts_info,
        )
        atomic_write_json(staging / "run.json", metadata)
        _promote_directory(staging, paths.run_dir)
        promoted = True
        return metrics
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def run_benchmark(
    config: BenchmarkConfig,
    synthetic_factory: SyntheticFactory | None = None,
    synthetic_generator: SyntheticGenerator | None = None,
    whisper_runner: ModelRunner | None = None,
    whisper_at_runner: ModelRunner | None = None,
) -> BenchmarkResult:
    """Run one validated benchmark and atomically promote its artifacts."""
    _validate_config(config)
    ensure_dataset_workspace(config.repo_root, config.dataset)
    initial_paths = resolve_run_paths(
        config.repo_root,
        config.dataset,
        config.model,
        "human",
    )
    load_reference_records(initial_paths.references)
    human_status = inspect_audio_set(
        initial_paths.references,
        initial_paths.human_audio,
    )
    source = resolve_audio_source(config.audio_source, human_status)
    paths = resolve_run_paths(
        config.repo_root,
        config.dataset,
        config.model,
        source,
    )

    tts_info: TTSInfo | None = None
    if source == "human":
        audio_dir = paths.human_audio
    else:
        audio_dir = paths.synthetic_audio
        from deafbench.benchmark.synthetic import (
            create_whisperspeech_generator,
            generate_synthetic_set,
        )

        tts_info = _prepare_synthetic(
            config,
            paths,
            synthetic_factory or create_whisperspeech_generator,
            synthetic_generator or generate_synthetic_set,
        )
    _require_complete_audio(paths.references, audio_dir, source)

    runner = {
        "whisper": whisper_runner,
        "whisper-at": whisper_at_runner,
    }.get(config.model)
    metrics = _build_run_bundle(
        config,
        paths,
        source,
        audio_dir,
        tts_info,
        runner or _default_model_runner(config.model),
    )
    return BenchmarkResult(
        source,
        paths.predictions,
        paths.report,
        paths.metadata,
        metrics,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deafbench benchmark",
        description="Run a complete DeafBench model benchmark.",
    )
    parser.add_argument("dataset")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--audio-source",
        choices=("auto", "human", "synthetic"),
        default="auto",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--scene-profile", default="default-v1")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the installed command-line benchmark workflow."""
    args = _parser().parse_args(argv)
    result = run_benchmark(
        BenchmarkConfig(
            repo_root=args.repo_root,
            dataset=args.dataset,
            model=args.model,
            audio_source=args.audio_source,
            scene_profile=args.scene_profile,
            seed=args.seed,
        )
    )
    from deafbench.cli import format_terminal_output

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Audio source: {result.resolved_source}")
    print(format_terminal_output(result.metrics))
    print(f"Predictions: {result.predictions}")
    print(f"Report: {result.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
