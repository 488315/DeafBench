import json

import pytest

from tools.recorder.core import load_prompts, output_path


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sample_id", ["../outside", "..\\outside", "C:outside"])
def test_load_prompts_rejects_unsafe_sample_ids(tmp_path, sample_id):
    references = tmp_path / "references.jsonl"
    references.write_text(
        json.dumps({"id": sample_id, "text": "Unsafe sample"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid id"):
        load_prompts(references)


@pytest.mark.parametrize("sample_id", ["../outside", "..\\outside", "C:outside"])
def test_output_path_rejects_unsafe_sample_ids(tmp_path, sample_id):
    with pytest.raises(ValueError, match="Invalid sample ID"):
        output_path(tmp_path / "audio", sample_id)
