import os
import subprocess
from pathlib import Path


def test_pre_commit_hook_blocks_staged_audio(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    source = Path(__file__).parents[1]
    (repo / "sample.wav").write_bytes(b"synthetic")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)

    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test Operator"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.hooksPath", str(source / ".githooks")],
        check=True,
    )
    completed = subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "must fail"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(source)},
    )

    assert completed.returncode == 1
    assert "BLOCKED sample.wav" in completed.stderr
