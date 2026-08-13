import hashlib
from io import BytesIO
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


class _Model:
    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        return self

    def __call__(self, waveform):
        assert waveform.to("cpu") is waveform
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


def _install_fake_alignment_runtime(
    install_fake_torchaudio_runtime, *, decoded_bytes=None
):
    bundle = SimpleNamespace(
        _path="https://download.example/mms.pt",
        sample_rate=16_000,
        get_model=lambda **kwargs: _Model(),
        get_tokenizer=lambda: lambda words: list(words),
        get_aligner=lambda: lambda emission, tokens: [
            [SimpleNamespace(score=0.9) for _ in word] for word in tokens
        ],
    )
    def inspect_decoded_audio(source):
        assert isinstance(source, BytesIO)
        if decoded_bytes is not None:
            decoded_bytes.append(source.read())

    install_fake_torchaudio_runtime(
        bundle=bundle,
        pipeline_name="MMS_FA",
        on_soundfile_read=inspect_decoded_audio,
    )
    return bundle


def test_mms_aligner_records_artifact_and_aligns_reference(
    install_fake_torchaudio_runtime, tmp_path
):
    decoded_bytes = []
    _install_fake_alignment_runtime(
        install_fake_torchaudio_runtime, decoded_bytes=decoded_bytes
    )
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
    assert decoded_bytes == [b"fixture"]
    assert evidence.audio_sha256 == hashlib.sha256(decoded_bytes[0]).hexdigest()


def test_mms_aligner_rejects_empty_model_state(install_fake_torchaudio_runtime):
    bundle = _install_fake_alignment_runtime(install_fake_torchaudio_runtime)
    bundle.get_model = lambda **kwargs: SimpleNamespace(
        to=lambda device: SimpleNamespace(
            eval=lambda: SimpleNamespace(state_dict=lambda: {})
        )
    )

    with pytest.raises(RuntimeError, match="model state is empty"):
        MMSForcedAligner()
