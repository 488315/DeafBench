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


def _direct_url_line(package: str, lock_text: str) -> str:
    match = re.search(
        rf"(?m)^{re.escape(package)} @ (?P<url>[^\n]+)$",
        lock_text,
    )
    assert match is not None, f"{package} must use a direct URL"
    return match.group("url")


def test_open_asr_torch_runtime_uses_patched_matching_abi() -> None:
    lock_text = _LOCK.read_text(encoding="utf-8")
    torch_version = _locked_version("torch", lock_text)
    torchaudio_version = _locked_version("torchaudio", lock_text)
    torch_url = _direct_url_line("torch", lock_text)
    torchaudio_url = _direct_url_line("torchaudio", lock_text)
    k2_url = _direct_url_line("k2", lock_text)

    assert Version(torch_version) >= Version("2.6.0")
    assert torchaudio_version == torch_version
    assert "--extra-index-url" not in lock_text
    assert (
        torch_url
        == "https://download-r2.pytorch.org/whl/cu124/"
        "torch-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
        "#sha256=a393b506844035c0dac2f30ea8478c343b8e95a429f06f3b3cadfc7f53adb597"
    )
    assert (
        torchaudio_url
        == "https://download-r2.pytorch.org/whl/cu124/"
        "torchaudio-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
        "#sha256=3e5ffa69606171c74f3e2b969785ead50b782ca657e746aaee1ee7cc88dcfc08"
    )
    assert (
        k2_url
        == "https://huggingface.co/csukuangfj/k2/resolve/"
        "da4df24bb5f00061097f24d8a5caab841fa3c7fd/ubuntu-cuda/"
        "k2-1.24.4.dev20250130+cuda12.4.torch2.6.0-cp312-cp312-"
        "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        "#sha256=e9d703f0599b56dfccba1f659fa172c38e5599808b5ca1b766ab8a724a3d0c21"
    )
