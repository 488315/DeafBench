import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


def test_benchmark_extra_installs_whisperspeech_runtime_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert "webdataset>=1.0.2,<2.0.0" in metadata["project"][
        "optional-dependencies"
    ]["benchmark"]


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("DEAFBENCH_RUN_PACKAGING_INTEGRATION") != "1",
    reason="requires an isolated installation of DeafBench[benchmark]",
)
def test_benchmark_install_imports_whisperspeech_pipeline() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import deafbench; import webdataset; "
                "from whisperspeech.pipeline import Pipeline; "
                "print(deafbench.__version__ if "
                "hasattr(deafbench, '__version__') else 'deafbench-ok')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
