"""Shared fake Open ASR evaluator checkout builder for leaderboard tests."""

from __future__ import annotations

from pathlib import Path
import subprocess


def commit_fake_official_evaluator(
    tmp_path: Path,
    *,
    score_stub: str,
    normalizer_stub: str = "normalizer = lambda text: text.lower()\n",
) -> tuple[Path, str]:
    """Create and commit a fake evaluator checkout with caller-provided behavior."""
    checkout = tmp_path / "official"
    normalizer = checkout / "normalizer"
    normalizer.mkdir(parents=True)
    for name in ("__init__.py", "normalizer.py"):
        (normalizer / name).write_text("", encoding="utf-8")
    (normalizer / "data_utils.py").write_text(normalizer_stub, encoding="utf-8")
    (normalizer / "eval_utils.py").write_text(score_stub, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=DeafBench Tests",
            "-c",
            "user.email=tests@deafbench.invalid",
            "commit",
            "-qm",
            "test evaluator",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return checkout, revision
