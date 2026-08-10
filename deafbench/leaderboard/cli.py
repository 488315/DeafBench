"""Command-line surface for official leaderboard scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .official import OfficialEvaluatorError, open_asr_evaluator


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
    score.add_argument("results", help="Directory containing official result JSONLs")
    score.add_argument("--official-repo", required=True, help="Pinned evaluator checkout")
    score.add_argument("--model-id", required=True, help="Hugging Face model id")
    score.add_argument("--output", help="Write the score as JSON")


def run_leaderboard(parsed: argparse.Namespace) -> int:
    try:
        result = open_asr_evaluator(parsed.official_repo).score(
            parsed.results,
            parsed.model_id,
        )
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
