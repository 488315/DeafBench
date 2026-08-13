from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest

from deafbench.benchmark.forced_alignment import (
    MMSForcedAligner,
    coverage_from_word_scores,
)
from deafbench.benchmark.spoken_reference import prepare_spoken_reference


def test_alignment_coverage_counts_characters_at_predeclared_score_floor():
    words = ("meet", "eight", "thirty")
    scores = (
        (0.9, 0.9, 0.9, 0.9),
        (0.9, 0.9, 0.1, 0.9, 0.9),
        (0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
    )

    total, entities = coverage_from_word_scores(
        words,
        scores,
        {"8:30": (1, 3)},
        score_threshold=0.25,
    )

    assert total == pytest.approx(14 / 15)
    assert entities == {"8:30": pytest.approx(10 / 11)}


def test_alignment_coverage_rejects_shape_or_range_drift():
    with pytest.raises(ValueError, match="word count"):
        coverage_from_word_scores(("one",), (), {}, score_threshold=0.25)
    with pytest.raises(ValueError, match="character count"):
        coverage_from_word_scores(
            ("one",), ((0.9,),), {}, score_threshold=0.25
        )
    with pytest.raises(ValueError, match="entity word range"):
        coverage_from_word_scores(
            ("one",), ((0.9, 0.9, 0.9),), {"bad": (0, 2)}, score_threshold=0.25
        )


def test_alignment_coverage_rejects_invalid_score_floor():
    with pytest.raises(ValueError, match="threshold"):
        coverage_from_word_scores((), (), {}, score_threshold=1.1)


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


class _Model:
    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        return self

    def __call__(self, waveform):
        assert isinstance(waveform, _Waveform)
        return (["emission"], None)

    def state_dict(self):
        tensor = SimpleNamespace(
            detach=lambda: tensor,
            cpu=lambda: tensor,
            contiguous=lambda: tensor,
            dtype="float32",
            shape=(1,),
            numpy=lambda: SimpleNamespace(tobytes=lambda: b"pinned-model"),
        )
        return {"weight": tensor}


def _install_fake_alignment_runtime(monkeypatch, tmp_path):
    bundle = SimpleNamespace(
        _path="https://download.example/mms.pt",
        sample_rate=16_000,
        get_model=lambda **kwargs: _Model(),
        get_tokenizer=lambda: lambda words: list(words),
        get_aligner=lambda: lambda emission, tokens: [
            [SimpleNamespace(score=0.9) for _ in word] for word in tokens
        ],
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
        SimpleNamespace(MMS_FA=bundle),
    )
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    return bundle


def test_mms_aligner_records_artifact_and_aligns_reference(monkeypatch, tmp_path):
    _install_fake_alignment_runtime(monkeypatch, tmp_path)
    aligner = MMSForcedAligner()
    prepared = prepare_spoken_reference(
        "Meet at eight thirty",
        {"eight thirty": "TIME"},
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")

    evidence = aligner.align(audio, prepared, score_threshold=0.25)

    assert evidence.token_coverage == 1.0
    assert evidence.critical_entity_coverage == {"eight thirty": 1.0}
    assert evidence.adapter == "torchaudio-MMS_FA"
    assert evidence.adapter_revision.startswith("torchaudio=2.9.1;model_sha256=")


def test_mms_aligner_rejects_empty_model_state(monkeypatch, tmp_path):
    bundle = _install_fake_alignment_runtime(monkeypatch, tmp_path)
    bundle.get_model = lambda **kwargs: SimpleNamespace(
        to=lambda device: SimpleNamespace(
            eval=lambda: SimpleNamespace(state_dict=lambda: {})
        )
    )

    with pytest.raises(RuntimeError, match="model state is empty"):
        MMSForcedAligner()
