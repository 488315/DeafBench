from pathlib import Path
import re

from packaging.version import Version


_LOCK = (
    Path(__file__).parents[1]
    / "experiments"
    / "open-asr"
    / "requirements.lock.txt"
)


def _locked_version(package: str, lock_text: str) -> str:
    match = re.search(
        rf"(?m)^{re.escape(package)}(?:==| @ [^\n]*/{re.escape(package)}-)"
        r"(\d+\.\d+\.\d+)",
        lock_text,
    )
    assert match is not None, f"{package} must be pinned"
    return match.group(1).split("+", maxsplit=1)[0]


def test_open_asr_torch_runtime_uses_patched_matching_abi() -> None:
    lock_text = _LOCK.read_text(encoding="utf-8")
    torch_version = _locked_version("torch", lock_text)
    torchaudio_version = _locked_version("torchaudio", lock_text)

    assert Version(torch_version) >= Version("2.6.0")
    assert torchaudio_version == torch_version
    assert f".torch{torch_version}-cp312-" in lock_text
    assert "--extra-index-url" not in lock_text
    assert lock_text.count("download-r2.pytorch.org/whl/cu124/") == 2
    assert lock_text.count("#sha256=") == 3
