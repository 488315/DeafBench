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


def test_qwen_asr_extra_is_isolated_from_base_installation() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = metadata["project"]["dependencies"]
    qwen_dependencies = metadata["project"]["optional-dependencies"]["qwen-asr"]

    assert qwen_dependencies == ["transformers[torch]>=5.13.0,<6.0.0"]
    assert all("transformers" not in dependency for dependency in dependencies)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("DEAFBENCH_RUN_PACKAGING_INTEGRATION") != "1",
    reason="requires an isolated installation of DeafBench[benchmark]",
)
def test_benchmark_install_imports_whisperspeech_pipeline(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = tmp_path / "benchmark-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(environment)],
        check=True,
        capture_output=True,
        text=True,
    )
    environment_python = environment / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    installed = subprocess.run(
        [
            str(environment_python),
            "-m",
            "pip",
            "install",
            f"{project_root}[benchmark]",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr

    completed = subprocess.run(
        [
            str(environment_python),
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
