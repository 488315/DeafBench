import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.smoke


def test_module_cli_compare_smoke():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deafbench",
            "compare",
            "examples/references.jsonl",
            "examples/model-b.jsonl",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DeafBench v0.1" in result.stdout
    assert "Samples: 3" in result.stdout


def test_benchmark_help_does_not_require_checkout_or_heavy_dependencies(
    tmp_path: Path,
):
    subprocess_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COV_CORE") and key != "COVERAGE_PROCESS_START"
    }
    result = subprocess.run(
        [sys.executable, "-m", "deafbench", "benchmark", "--help"],
        cwd=tmp_path,
        env=subprocess_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--audio-source" in result.stdout
