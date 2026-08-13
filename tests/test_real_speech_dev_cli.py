import json

from deafbench.leaderboard.dev_cli import main


def test_materialize_command_uses_versioned_contract(tmp_path, capsys):
    calls = []

    def materializer(manifest, references, destination):
        calls.append((manifest, references, destination))
        return {"sample_count": 100, "references_sha256": "a" * 64}

    assert main(["materialize", "--repo-root", str(tmp_path)], materializer) == 0

    corpus = tmp_path / "benchmarks" / "real-speech-dev-v1"
    assert calls == [
        (
            corpus / "manifest.json",
            corpus / "references.jsonl",
            corpus / "audio",
        )
    ]
    assert json.loads(capsys.readouterr().out) == {
        "references_sha256": "a" * 64,
        "sample_count": 100,
    }
