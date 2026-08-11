from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest

from deafbench.benchmark.independent_asr import (
    Wav2Vec2IndependentASR,
    collapse_ctc_labels,
)


def test_ctc_decoder_collapses_repeats_blanks_and_word_boundaries():
    labels = ("-", "|", "A", "B")

    text = collapse_ctc_labels(
        (0, 1, 2, 2, 0, 3, 1, 1, 2),
        labels,
        blank=0,
    )

    assert text == "ab a"


def test_ctc_decoder_rejects_unknown_label_index():
    with pytest.raises(ValueError, match="label index"):
        collapse_ctc_labels((9,), ("-", "A"), blank=0)


class _Waveform:
    def unsqueeze(self, dimension):
        assert dimension == 0
        return self

    def to(self, device):
        assert device == "cpu"
        return self


class _Samples:
    def mean(self, *, axis):
        assert axis == 1
        return "mono"


class _Indices:
    def cpu(self):
        return self

    def tolist(self):
        return [0, 1, 2, 2, 0, 3]


class _EmissionRow:
    def argmax(self, *, dim):
        assert dim == -1
        return _Indices()


class _Emission:
    def __getitem__(self, index):
        assert index == 0
        return _EmissionRow()


class _Model:
    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        return self

    def __call__(self, waveform):
        assert isinstance(waveform, _Waveform)
        return (_Emission(), None)


def _install_fake_asr_runtime(monkeypatch, tmp_path, *, artifact=True):
    checkpoint = tmp_path / "checkpoints" / "wav2vec2.pt"
    checkpoint.parent.mkdir()
    if artifact:
        checkpoint.write_bytes(b"pinned-model")

    bundle = SimpleNamespace(
        _path="https://download.example/wav2vec2.pt",
        sample_rate=16_000,
        get_model=lambda: _Model(),
        get_labels=lambda: ("-", "|", "A", "B"),
    )
    torch = SimpleNamespace(
        device=lambda value: value,
        from_numpy=lambda mono: _Waveform(),
        hub=SimpleNamespace(get_dir=lambda: str(tmp_path)),
        inference_mode=nullcontext,
    )
    torchaudio = SimpleNamespace(
        __version__="2.9.1",
        functional=SimpleNamespace(resample=lambda waveform, source, target: waveform),
    )
    soundfile = SimpleNamespace(read=lambda *args, **kwargs: (_Samples(), 48_000))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setitem(
        sys.modules,
        "torchaudio.pipelines",
        SimpleNamespace(WAV2VEC2_ASR_BASE_960H=bundle),
    )
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)


def test_independent_asr_records_artifact_and_transcribes(monkeypatch, tmp_path):
    _install_fake_asr_runtime(monkeypatch, tmp_path)
    recognizer = Wav2Vec2IndependentASR()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")

    assert recognizer.transcribe(audio) == "ab"
    assert recognizer.adapter_revision.startswith(
        "torchaudio=2.9.1;model_sha256="
    )


def test_independent_asr_rejects_missing_model_artifact(monkeypatch, tmp_path):
    _install_fake_asr_runtime(monkeypatch, tmp_path, artifact=False)

    with pytest.raises(RuntimeError, match="artifact is missing"):
        Wav2Vec2IndependentASR()
