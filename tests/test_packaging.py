import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


_PACKAGING_TIMEOUT_SECONDS = 300


def test_build_backend_requires_patched_setuptools() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    requirements = [
        Requirement(requirement)
        for requirement in metadata["build-system"]["requires"]
    ]
    setuptools = next(
        requirement
        for requirement in requirements
        if requirement.name.casefold() == "setuptools"
    )

    assert Version("82.0.1") not in setuptools.specifier
    assert Version("83.0.0") in setuptools.specifier


def test_project_uses_spdx_license_metadata() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert metadata["project"]["license"] == "Apache-2.0"
    assert metadata["project"]["license-files"] == ["LICENSE"]


def test_whisper_at_docs_do_not_downgrade_setuptools() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert 'pip install "setuptools<81"' not in readme
    assert "python -m deafbench.whisper_at_compat" in readme
    assert "setuptools 83 or newer" in readme


def test_whisper_at_ci_exercises_exact_setuptools_floor() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = root.joinpath(".github/workflows/ci.yml").read_text(encoding="utf-8")
    constraint = root.joinpath(
        ".github/build-constraints/whisper-at-setuptools83.txt"
    ).read_text(encoding="utf-8")
    job_start = workflow.index("  whisper-at:\n")
    job_end = workflow.index("\n  ark-transformers:\n", job_start)
    whisper_at_job = workflow[job_start:job_end]

    floor_start = whisper_at_job.index(
        "      - name: Verify setuptools 83 isolated build\n"
    )
    normal_start = whisper_at_job.index(
        "      - name: Install pinned Whisper-AT source normally\n"
    )
    floor_step = whisper_at_job[floor_start:normal_start]
    normal_step = whisper_at_job[normal_start:]

    assert constraint.strip() == "setuptools==83.0.0"
    assert (
        "PIP_BUILD_CONSTRAINT: >-\n"
        "            ${{ github.workspace }}\\.github\\build-constraints\\"
        "whisper-at-setuptools83.txt"
    ) in floor_step
    assert "run: python -m deafbench.whisper_at_compat" in floor_step
    assert "run: python -m deafbench.whisper_at_compat" in normal_step


def test_ark_ci_exercises_secure_transformers_floor() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    ).read_text(encoding="utf-8")
    job_start = workflow.index("  ark-transformers:\n")
    job_end = workflow.index("\n  package:\n", job_start)
    ark_job = workflow[job_start:job_end]

    assert 'python -m pip install "transformers[torch]==5.5.0"' in ark_job
    assert 'python -m pip install ".[ark-onnx-asr,test]"' in ark_job
    assert "assert transformers.__version__ == '5.5.0'" in ark_job
    assert "load_native(); load_onnx()" in ark_job
    assert "tests/test_ark_asr_adapter.py" in ark_job
    assert "tests/test_ark_asr_worker.py" in ark_job
    assert "tests/test_ark_asr_onnx_adapter.py" in ark_job
    assert "tests/test_ark_asr_onnx_worker.py" in ark_job
    assert "python -m pip check" in ark_job


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
    assert "ruff>=0.12,<0.13" in test_dependencies
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
        "transformers[torch]>=5.5.0,<6.0.0",
    ]
    assert all("transformers" not in dependency for dependency in dependencies)


def test_ark_onnx_asr_extra_uses_secure_transformers_floor() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = metadata["project"]["dependencies"]
    ark_dependencies = metadata["project"]["optional-dependencies"][
        "ark-onnx-asr"
    ]

    assert ark_dependencies == [
        "librosa>=0.11,<1.0",
        "onnxruntime>=1.23,<2.0",
        "soundfile>=0.13,<1.0",
        "transformers[torch]>=5.5.0,<6.0.0",
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
