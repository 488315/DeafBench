from pathlib import Path
import tomllib


def test_benchmark_extra_installs_whisperspeech_runtime_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert "webdataset>=1.0.2,<2.0.0" in metadata["project"][
        "optional-dependencies"
    ]["benchmark"]
