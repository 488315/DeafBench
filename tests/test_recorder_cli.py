import importlib

import pytest


pytestmark = pytest.mark.functional


def _recorder_app():
    return importlib.import_module("deafbench.recorder.app")


def test_packaged_recorder_bootstraps_core_reference(tmp_path):
    app = _recorder_app()

    references, audio_dir = app.ensure_dataset_workspace(tmp_path, "core-v1")

    assert references == tmp_path / "benchmarks" / "core-v1" / "references.jsonl"
    assert audio_dir == tmp_path / "benchmarks" / "core-v1" / "audio"
    assert '"id":"core-001"' in references.read_text(encoding="utf-8")


def test_packaged_recorder_preserves_existing_reference(tmp_path):
    app = _recorder_app()
    dataset_dir = tmp_path / "benchmarks" / "core-v1"
    dataset_dir.mkdir(parents=True)
    references = dataset_dir / "references.jsonl"
    references.write_text('{"id":"custom","text":"Keep me"}\n', encoding="utf-8")

    resolved_references, _ = app.ensure_dataset_workspace(tmp_path, "core-v1")

    assert resolved_references == references
    assert references.read_text(encoding="utf-8") == '{"id":"custom","text":"Keep me"}\n'


def test_packaged_recorder_rejects_missing_custom_reference(tmp_path):
    app = _recorder_app()

    with pytest.raises(FileNotFoundError, match="custom-v1"):
        app.ensure_dataset_workspace(tmp_path, "custom-v1")
