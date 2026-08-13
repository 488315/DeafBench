import subprocess
from pathlib import Path

from deafbench.pilot.source_control import scan_staged


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Test Operator")
    (path / "safe.txt").write_text("safe", encoding="utf-8")
    _git(path, "add", "safe.txt")
    _git(path, "commit", "-qm", "initial")
    return path


def test_scanner_rejects_customer_artifact_categories(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    for name in ("sample.wav", "transcript.json", "prediction.json", "customer-report.md"):
        (repo / name).write_text("content", encoding="utf-8")
        _git(repo, "add", name)

    reasons = {finding.reason for finding in scan_staged(repo)}

    assert reasons == {"customer artifact path"}


def test_scanner_rejects_case_identifiers(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    marker = "case-" + "1" * 32
    (repo / "notes.txt").write_text(marker, encoding="utf-8")
    _git(repo, "add", "notes.txt")

    findings = scan_staged(repo)

    assert findings[0].reason == "opaque customer case identifier"


def test_scanner_rejects_likely_secrets(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    secret_assignment = "api" + "_key=do-not-store"
    (repo / "notes.txt").write_text(secret_assignment, encoding="utf-8")
    _git(repo, "add", "notes.txt")

    assert scan_staged(repo)[0].reason == "possible secret"


def test_scanner_rejects_private_signing_key_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "pilot-signing-key.pem").write_text("private material", encoding="utf-8")
    _git(repo, "add", "pilot-signing-key.pem")

    assert scan_staged(repo)[0].reason == "private key artifact"


def test_scanner_rejects_renamed_private_key_content(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    header = "-----BEGIN " + "PRIVATE KEY-----"
    footer = "-----END " + "PRIVATE KEY-----"
    (repo / "notes.txt").write_text(
        f"{header}\nprivate material\n{footer}\n",
        encoding="utf-8",
    )
    _git(repo, "add", "notes.txt")

    assert scan_staged(repo)[0].reason == "private key artifact"


def test_scanner_uses_nested_repository_index(tmp_path: Path) -> None:
    outer = _repo(tmp_path / "outer")
    nested = _repo(outer / "nested")
    (nested / "case-artifact.txt").write_text("x", encoding="utf-8")
    _git(nested, "add", "case-artifact.txt")

    assert scan_staged(outer) == ()
    assert scan_staged(nested)[0].path == "case-artifact.txt"


def test_scanner_uses_alternate_worktree_index(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    worktree = tmp_path / "alternate"
    _git(repo, "worktree", "add", "-q", "-b", "alternate", str(worktree))
    (worktree / "sample.flac").write_bytes(b"not audio")
    _git(worktree, "add", "sample.flac")

    assert scan_staged(repo) == ()
    assert scan_staged(worktree)[0].path == "sample.flac"
