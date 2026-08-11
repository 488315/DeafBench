"""Installed entry point for the fail-closed customer-artifact hook."""

from __future__ import annotations

import argparse
from pathlib import Path

from deafbench.pilot.source_control import scan_staged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    findings = scan_staged(args.repo_root)
    for finding in findings:
        print(f"BLOCKED {finding.path}: {finding.reason}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
