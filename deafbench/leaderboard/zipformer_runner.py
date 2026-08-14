"""Run the reviewed Zipformer Space against pinned leaderboard inputs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from deafbench.dependency_security import load_dependency_dispositions

from .official import open_asr_evaluator
from .policy import verify_evaluation_policy
from .zipformer import PinnedZipformerContract


ZIPFORMER_RUNNER_REVISION = "64c698c42932a54bc7a40a7f172d03c8c4838fe6"
ICEFALL_REVISION = "3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1"
_PINNED_IMPORT_ROOTS = (
    "run_eval",
    "normalizer",
    "beam_search",
    "train",
    "icefall",
)


def _require_fresh_pinned_imports() -> None:
    """Reject module-cache state that could bypass verified source paths."""
    loaded = [name for name in _PINNED_IMPORT_ROOTS if name in sys.modules]
    if loaded:
        raise RuntimeError(
            "pinned Zipformer modules are already loaded: " + ", ".join(loaded)
        )


def _runner_argv(
    args: argparse.Namespace,
    contract: PinnedZipformerContract,
) -> list[str]:
    """Translate DeafBench arguments into the reviewed Space CLI contract."""
    if args.device < 0:
        raise ValueError("the pinned Zipformer baseline requires a CUDA device")
    contract.validate_dataset(args.dataset, args.split)
    runner_argv = [
        "--model_id",
        contract.model_id,
        "--dataset_path",
        contract.dataset_id,
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--device",
        str(args.device),
        "--batch_size",
        str(args.batch_size),
        "--warmup_steps",
        str(args.warmup_steps),
    ]
    if args.max_eval_samples is not None:
        runner_argv.extend(["--max_eval_samples", str(args.max_eval_samples)])
    if not args.streaming:
        runner_argv.append("--no-streaming")
    return runner_argv


def _require_source(
    checkout: Path,
    revision: str,
    required_file: str,
    label: str,
) -> None:
    if not (checkout / required_file).is_file():
        raise RuntimeError(f"{label} is missing {required_file}: {checkout}")
    try:
        actual = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot verify {label}: {checkout}") from exc
    if actual != revision:
        raise RuntimeError(f"{label} revision mismatch: expected {revision}, got {actual}")
    if status:
        raise RuntimeError(f"{label} source is modified: {checkout}")
    try:
        ignored_content = subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot verify ignored {label} source: {checkout}") from exc
    if ignored_content:
        raise RuntimeError(f"{label} has ignored checkout content: {checkout}")


def run(args: argparse.Namespace) -> dict[str, float]:
    """Execute one pinned dataset/split using the official Space runner."""
    load_dependency_dispositions()
    runner_repo = Path(args.runner_repo).resolve()
    evaluator_repo = Path(args.official_repo).resolve()
    icefall_repo = Path(args.icefall_repo).resolve()
    output_dir = Path(args.output_dir).resolve()

    verify_evaluation_policy(args.evaluation_policy)
    _require_source(
        runner_repo,
        ZIPFORMER_RUNNER_REVISION,
        "run_eval.py",
        "Zipformer runner",
    )
    open_asr_evaluator(evaluator_repo).validate()
    _require_source(
        icefall_repo,
        ICEFALL_REVISION,
        "egs/librispeech/ASR/zipformer/train.py",
        "Icefall",
    )
    _require_fresh_pinned_imports()

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = os.getcwd()
    previous_path = list(sys.path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path[:0] = [
            str(runner_repo),
            str(evaluator_repo),
            str(icefall_repo),
            str(icefall_repo / "egs/librispeech/ASR/zipformer"),
        ]
        os.chdir(output_dir)

        from datasets import load_dataset
        from huggingface_hub import snapshot_download
        import torch

        official_runner = importlib.import_module("run_eval")
        contract = PinnedZipformerContract()
        official_runner.data_utils.load_data = (
            lambda runner_args: contract.load_dataset(load_dataset, runner_args)
        )
        official_runner.snapshot_download = (
            lambda model_id, **kwargs: contract.snapshot_model(
                snapshot_download,
                model_id,
                **kwargs,
            )
        )
        runner_argv = _runner_argv(args, contract)

        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        official_runner.main(official_runner.get_parser().parse_args(runner_argv))
        summary = {
            "wall_seconds": round(time.time() - started, 3),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }
        print("DEAFBENCH_ZIPFORMER_RUN=" + json.dumps(summary, sort_keys=True))
        return summary
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_dont_write_bytecode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-repo", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--icefall-repo", required=True)
    parser.add_argument("--evaluation-policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--streaming", action="store_true")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
