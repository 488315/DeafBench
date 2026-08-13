"""Reject customer artifacts from the Git index."""

from __future__ import annotations

import argparse
from pathlib import Path

from deafbench.pilot.source_control import scan_staged, scan_tracked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--tracked", action="store_true")
    parser.add_argument("--base")
    args = parser.parse_args()
    if args.base and not args.tracked:
        parser.error("--base requires --tracked")
    findings = (
        scan_tracked(args.repo_root, base=args.base)
        if args.tracked
        else scan_staged(args.repo_root)
    )
    for finding in findings:
        print(f"BLOCKED {finding.path}: {finding.reason}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
