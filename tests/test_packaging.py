import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


_PACKAGING_TIMEOUT_SECONDS = 300


def test_wheel_discovery_excludes_nonruntime_trees() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert metadata["tool"]["setuptools"]["include-package-data"] is False
    discovery = metadata["tool"]["setuptools"]["packages"]["find"]
    assert discovery["namespaces"] is False
    excluded = set(discovery["exclude"])
    assert {
        "build",
        "build.*",
        "experiments",
        "experiments.*",
        "tests",
        "tests.*",
    } <= excluded


def test_benchmark_extra_installs_whisperspeech_runtime_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert (
        "webdataset>=1.0.2,<2.0.0"
        in metadata["project"]["optional-dependencies"]["benchmark"]
    )
    assert (
        "WhisperSpeech>=0.8.9"
        in metadata["project"]["optional-dependencies"]["benchmark"]
    )


def test_test_extra_installs_collection_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    test_dependencies = metadata["project"]["optional-dependencies"]["test"]
    assert "cryptography>=48.0,<49.0" in test_dependencies
    assert "kaldialign==0.12.0" in test_dependencies
    assert "scipy>=1.15,<2.0" in test_dependencies
    assert "soundfile>=0.13,<1.0" in test_dependencies


def test_real_speech_dev_extra_is_isolated_from_base_installation() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = metadata["project"]["dependencies"]
    dev_dependencies = metadata["project"]["optional-dependencies"][
        "real-speech-dev"
    ]

    assert dev_dependencies == [
        "datasets==3.6.0; python_version < '3.14'",
        "numpy>=1.26",
        "scipy>=1.15,<2.0",
        "soundfile>=0.13,<1.0",
    ]
    assert all("datasets" not in dependency for dependency in dependencies)


def test_coverage_tracks_isolated_evaluator_subprocesses() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert metadata["tool"]["coverage"]["run"]["patch"] == ["subprocess"]
    assert metadata["tool"]["coverage"]["report"]["omit"] == [
        "deafbench/recorder/app.py"
    ]


def test_qwen_asr_extra_is_isolated_from_base_installation() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = metadata["project"]["dependencies"]
    qwen_dependencies = metadata["project"]["optional-dependencies"]["qwen-asr"]

    assert qwen_dependencies == [
        "scipy>=1.15,<2.0",
        "transformers[torch]>=5.13.0,<6.0.0",
    ]
    assert all("transformers" not in dependency for dependency in dependencies)


def test_parakeet_asr_extra_is_isolated_from_base_installation() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = metadata["project"]["dependencies"]
    parakeet_dependencies = metadata["project"]["optional-dependencies"]["parakeet-asr"]

    assert parakeet_dependencies == [
        "nemo_toolkit[asr]>=2.4,<3",
        "numba>=0.61,<0.62",
    ]
    assert all("nemo_toolkit" not in dependency for dependency in dependencies)


def test_granite_asr_extra_is_isolated_from_base_installation() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = metadata["project"]["dependencies"]
    granite_dependencies = metadata["project"]["optional-dependencies"]["granite-asr"]

    assert granite_dependencies == [
        "scipy>=1.15,<2.0",
        "torchaudio>=2.8,<3.0",
        "transformers[torch]>=5.13.0,<6.0.0",
    ]
    assert all("transformers" not in dependency for dependency in dependencies)


def test_granite_nar_extra_is_isolated_from_base_installation() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = metadata["project"]["dependencies"]
    granite_dependencies = metadata["project"]["optional-dependencies"][
        "granite-nar-asr"
    ]

    assert granite_dependencies == [
        "accelerate>=1.10,<2.0",
        "flash-attn==2.8.3; platform_system == 'Linux'",
        "soundfile>=0.13,<1.0",
        "torch==2.9.1",
        "torchcodec==0.9.1",
        "torchaudio==2.9.1",
        "transformers>=5.5.3,<6.0.0",
    ]
    assert all("transformers" not in dependency for dependency in dependencies)


def test_ark_asr_extra_is_isolated_from_base_installation() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = metadata["project"]["dependencies"]
    ark_dependencies = metadata["project"]["optional-dependencies"]["ark-asr"]

    assert ark_dependencies == [
        "librosa>=0.11,<1.0",
        "soundfile>=0.13,<1.0",
        "transformers[torch]>=4.57.6,<6.0.0",
    ]
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
        timeout=_PACKAGING_TIMEOUT_SECONDS,
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
        timeout=_PACKAGING_TIMEOUT_SECONDS,
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
        timeout=_PACKAGING_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
