from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest


class _FakeWaveform:
    def unsqueeze(self, dimension: int) -> "_FakeWaveform":
        assert dimension == 0
        return self

    def to(self, device: str) -> "_FakeWaveform":
        assert device == "cpu"
        return self


class _FakeSamples:
    def mean(self, *, axis: int) -> str:
        assert axis == 1
        return "mono"


@pytest.fixture
def install_fake_torchaudio_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Install the common Torch audio modules around a bundle-specific fake."""

    def install(
        *, bundle: object, pipeline_name: str, on_soundfile_read=None
    ) -> None:
        torch = SimpleNamespace(
            device=lambda value: value,
            from_numpy=lambda mono: _FakeWaveform(),
            hub=SimpleNamespace(get_dir=lambda: str(tmp_path)),
            inference_mode=nullcontext,
        )
        torchaudio = SimpleNamespace(
            __version__="2.9.1",
            functional=SimpleNamespace(
                resample=lambda waveform, source, target: waveform
            ),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
        monkeypatch.setitem(
            sys.modules,
            "torchaudio.pipelines",
            SimpleNamespace(**{pipeline_name: bundle}),
        )
        def read(source, **kwargs):
            if on_soundfile_read is not None:
                on_soundfile_read(source)
            return _FakeSamples(), 48_000

        monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(read=read))

    return install
