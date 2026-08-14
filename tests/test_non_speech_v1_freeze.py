import json
import subprocess
from pathlib import Path

from deafbench.benchmark.freeze import verify_frozen_corpus


_ROOT = Path(__file__).parents[1]
_MANIFEST = _ROOT / "benchmarks" / "non-speech-v1" / "freeze-manifest.json"


def test_non_speech_v1_freeze_covers_every_tracked_corpus_artifact() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    required = set(manifest["artifacts"]["required"])
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "--", "benchmarks/non-speech-v1"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    tracked.remove("benchmarks/non-speech-v1/freeze-manifest.json")

    assert manifest["corpus"] == "non-speech-v1"
    assert manifest["status"] == "frozen"
    assert required == tracked


def test_checked_out_non_speech_v1_matches_its_freeze_manifest() -> None:
    result = verify_frozen_corpus(_MANIFEST, _ROOT)

    assert result.verified_required == 15
    assert result.verified_optional == 0
    assert result.missing_optional == ()
