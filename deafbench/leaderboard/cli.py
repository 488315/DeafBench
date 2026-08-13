"""Command-line surface for official leaderboard scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .official import OfficialEvaluatorError, open_asr_evaluator


def _add_evaluator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("results", help="Directory containing official result JSONLs")
    parser.add_argument("--official-repo", required=True, help="Pinned evaluator checkout")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id")
    parser.add_argument("--output", help="Write JSON output to this path")


def add_leaderboard_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "leaderboard",
        help="Use a pinned official leaderboard evaluator.",
    )
    actions = parser.add_subparsers(dest="leaderboard_action", required=True)
    score = actions.add_parser(
        "score",
        help="Score official JSONL manifests with the pinned upstream scorer.",
    )
    _add_evaluator_arguments(score)
    analyze = actions.add_parser(
        "analyze",
        help="Rank utterance errors with pinned normalization and alignment.",
    )
    _add_evaluator_arguments(analyze)
    analyze.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum error rows retained per dataset (default: 20)",
    )


def run_leaderboard(parsed: argparse.Namespace) -> int:
    try:
        evaluator = open_asr_evaluator(parsed.official_repo)
        if parsed.leaderboard_action == "analyze":
            result = evaluator.analyze(
                parsed.results,
                parsed.model_id,
                limit=parsed.limit,
            )
        else:
            result = evaluator.score(parsed.results, parsed.model_id)
    except OfficialEvaluatorError as exc:
        print(f"Leaderboard evaluation failed: {exc}", file=sys.stderr)
        return 1

    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if parsed.output:
        try:
            Path(parsed.output).write_text(serialized, encoding="utf-8")
        except OSError as exc:
            print(f"Could not write leaderboard score: {exc}", file=sys.stderr)
            return 1
    else:
        print(serialized, end="")
    return 0
