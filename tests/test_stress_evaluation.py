import json

import pytest

from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.stress_evaluation import run_stress_evaluation


def _write_inputs(tmp_path):
    references = tmp_path / "references.jsonl"
    reference = {
        "id": "stress-001",
        "text": "Meet Priya Shah at 8:30 PM",
        "critical": ["Priya Shah", "8:30 PM"],
        "critical_types": {
            "Priya Shah": "PROPER_NAME",
            "8:30 PM": "TIME",
        },
        "risk_categories": {
            "Priya Shah": "PROPER_NAME",
            "8:30 PM": "TIME",
        },
        "sounds": [],
    }
    references.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    prepared = tmp_path / "prepared"
    (prepared / "clean").mkdir(parents=True)
    (prepared / "stressed").mkdir()
    (prepared / "preparation-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lane": "accessibility-stress-v1",
                "sample_count": 1,
                "samples": [{"id": "stress-001"}],
            }
        ),
        encoding="utf-8",
    )
    return references, prepared


def _runner(audio, references, predictions):
    text = (
        "Meet Priya Shah at 8:30 PM"
        if audio.name == "clean"
        else "Meet Priya at PM"
    )
    predictions.write_text(
        json.dumps({"id": "stress-001", "text": text}) + "\n",
        encoding="utf-8",
    )
    return ModelRunInfo(
        "test-model",
        "example/test-model",
        "revision-1",
        {"beam_size": 1},
        {"local_rtfx": 2.0 if audio.name == "clean" else 1.5},
    )


def test_run_stress_evaluation_scores_and_promotes_paired_result(tmp_path):
    references, prepared = _write_inputs(tmp_path)
    destination = tmp_path / "result"

    result = run_stress_evaluation(
        prepared, references, destination, _runner
    )

    assert result["result_kind"] == "local_stress_observation"
    assert result["model"] == {
        "name": "test-model",
        "model_id": "example/test-model",
        "revision": "revision-1",
        "decoding": {"beam_size": 1},
    }
    assert result["summary"]["clean"]["wer"] == 0.0
    assert result["summary"]["stressed"]["deletions"] == 2
    assert result["performance"]["clean"]["local_rtfx"] == 2.0
    assert json.loads(
        (destination / "result.json").read_text(encoding="utf-8")
    ) == result


def test_run_stress_evaluation_rejects_model_configuration_drift(tmp_path):
    references, prepared = _write_inputs(tmp_path)
    calls = 0

    def drifting_runner(audio, references_path, predictions):
        nonlocal calls
        calls += 1
        _runner(audio, references_path, predictions)
        return ModelRunInfo("test-model", f"model-{calls}")

    with pytest.raises(ValueError, match="different model configurations"):
        run_stress_evaluation(
            prepared,
            references,
            tmp_path / "result",
            drifting_runner,
        )

    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "manifest",
    [
        "not-json",
        json.dumps({"lane": "wrong", "sample_count": 0, "samples": []}),
        json.dumps(
            {
                "lane": "accessibility-stress-v1",
                "sample_count": 1,
                "samples": [{"id": "unknown"}],
            }
        ),
    ],
)
def test_run_stress_evaluation_rejects_invalid_preparation(tmp_path, manifest):
    references, prepared = _write_inputs(tmp_path)
    (prepared / "preparation-manifest.json").write_text(
        manifest, encoding="utf-8"
    )

    with pytest.raises(ValueError, match="manifest|unknown samples"):
        run_stress_evaluation(
            prepared, references, tmp_path / "result", _runner
        )
