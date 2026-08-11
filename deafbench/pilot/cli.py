"""Operator CLI for synthetic pilot release rehearsals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from deafbench.pilot.rehearsal import RehearsalResult, run_synthetic_rehearsal
from deafbench.pilot.storage import probe_bitlocker, restrict_acl_to_current_account


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., RehearsalResult] = run_synthetic_rehearsal,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m deafbench.pilot.cli")
    actions = parser.add_subparsers(dest="action", required=True)
    rehearse = actions.add_parser("rehearse")
    rehearse.add_argument("--repo-root", type=Path, required=True)
    rehearse.add_argument("--case-base", type=Path, required=True)
    rehearse.add_argument("--records-root", type=Path, required=True)
    rehearse.add_argument("--operator", required=True)
    args = parser.parse_args(argv)
    result = runner(
        repo_root=args.repo_root,
        case_base=args.case_base,
        records_root=args.records_root,
        operator=args.operator,
        protection_probe=probe_bitlocker,
        acl_restrictor=restrict_acl_to_current_account,
    )
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
