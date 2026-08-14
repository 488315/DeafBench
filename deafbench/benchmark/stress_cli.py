"""Installed command for the local accessibility stress lane."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, cast

from deafbench.benchmark.models import MODEL_NAMES
from deafbench.benchmark.runner import ModelName, _default_model_runner
from deafbench.benchmark.stress_contract import load_stress_cases
from deafbench.benchmark.stress_evaluation import ModelRunner, run_stress_evaluation
from deafbench.benchmark.stress_runner import (
    IMPLEMENTED_STRESSORS,
    prepare_stress_audio,
)


def _implemented_case_ids(references: Path) -> tuple[str, ...]:
    return tuple(
        cast(str, case["id"])
        for case in load_stress_cases(references)
        if len(case["stressors"]) == 2
        and case["stressors"][1]["kind"] in IMPLEMENTED_STRESSORS
    )


def run_stress_benchmark(
    references: Path,
    clean_audio: Path,
    destination: Path,
    model: ModelName,
    *,
    case_ids: Sequence[str] | None = None,
    implemented_only: bool = False,
    seed: int = 42,
    model_runner: ModelRunner | None = None,
) -> Mapping[str, Any]:
    """Prepare, execute, score, and atomically promote one local stress run."""
    if destination.exists():
        raise ValueError("Stress run destination already exists")
    if implemented_only and case_ids:
        raise ValueError("Choose either implemented-only or explicit case IDs")
    selected = _implemented_case_ids(references) if implemented_only else case_ids
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-run-", dir=destination.parent)
    )
    promoted = False
    try:
        prepared = staging / "prepared"
        prepare_stress_audio(
            references,
            clean_audio,
            prepared,
            case_ids=selected,
            seed=seed,
        )
        result = run_stress_evaluation(
            prepared,
            references,
            staging / "evaluation",
            model_runner or _default_model_runner(model),
        )
        os.replace(staging, destination)
        promoted = True
        return result
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deafbench stress",
        description="Run the local accessibility stress benchmark.",
    )
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--clean-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--implemented-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the installed accessibility stress workflow."""
    args = _parser().parse_args(argv)
    try:
        result = run_stress_benchmark(
            args.references,
            args.clean_audio,
            args.output,
            cast(ModelName, args.model),
            case_ids=args.case_id,
            implemented_only=args.implemented_only,
            seed=args.seed,
        )
    except ValueError as exc:
        raise SystemExit(f"Stress benchmark failed: {exc}") from exc
    print(f"Stress samples: {result['sample_count']}")
    print(f"Result: {args.output / 'evaluation' / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
