"""Customer-run zero-custody pilot commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from deafbench.pilot.export import CustomerExportResult, create_customer_export
from deafbench.pilot.rehearsal import RehearsalResult, run_synthetic_rehearsal
from deafbench.pilot.zero_custody import load_execution_attestation


def _result(value: object) -> str:
    return json.dumps(value.__dict__, sort_keys=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    rehearsal_runner: Callable[..., RehearsalResult] = run_synthetic_rehearsal,
    exporter: Callable[..., CustomerExportResult] = create_customer_export,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m deafbench.pilot.cli")
    actions = parser.add_subparsers(dest="action", required=True)

    rehearse = actions.add_parser("rehearse")
    rehearse.add_argument("--repo-root", type=Path, required=True)
    rehearse.add_argument("--output-dir", type=Path, required=True)
    rehearse.add_argument("--signing-key", type=Path, required=True)

    export = actions.add_parser("export")
    export.add_argument("--repo-root", type=Path, required=True)
    export.add_argument("--attestation", type=Path, required=True)
    export.add_argument("--result", type=Path, action="append", required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--signing-key", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.action == "rehearse":
        result = rehearsal_runner(
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            signing_key=args.signing_key,
        )
    else:
        load_execution_attestation(args.attestation)
        result = exporter(
            repo_root=args.repo_root,
            result_paths=args.result,
            output_dir=args.output_dir,
            signing_key=args.signing_key,
        )
    print(_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
