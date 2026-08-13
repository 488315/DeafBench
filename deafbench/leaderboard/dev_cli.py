"""Command line interface for the pinned public development corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence

from .dev_corpus import DevCorpusError, materialize_dev_corpus


def main(
    argv: Sequence[str] | None = None,
    materializer: Callable[[Path, Path, Path], Mapping[str, object]] = (
        materialize_dev_corpus
    ),
) -> int:
    """Materialize a versioned cohort from its repository contract."""
    parser = argparse.ArgumentParser(prog="deafbench dev-corpus")
    actions = parser.add_subparsers(dest="action", required=True)
    materialize = actions.add_parser("materialize")
    materialize.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    corpus = args.repo_root / "benchmarks" / "real-speech-dev-v1"
    try:
        result = materializer(
            corpus / "manifest.json",
            corpus / "references.jsonl",
            corpus / "audio",
        )
    except (DevCorpusError, OSError) as exc:
        print(f"dev corpus error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
