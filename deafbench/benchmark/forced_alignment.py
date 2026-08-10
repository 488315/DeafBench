"""MMS reference-conditioned forced alignment for synthetic admission."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .quality import AlignmentEvidence
from .spoken_reference import SpokenReference


def coverage_from_word_scores(
    words: Sequence[str],
    word_scores: Sequence[Sequence[float]],
    entity_word_ranges: Mapping[str, tuple[int, int]],
    *,
    score_threshold: float,
) -> tuple[float, dict[str, float]]:
    """Calculate covered-character fractions from forced-alignment scores."""
    if not 0 <= score_threshold <= 1:
        raise ValueError("score threshold must be between zero and one")
    if len(words) != len(word_scores):
        raise ValueError("alignment word count does not match transcript")
    for word, scores in zip(words, word_scores, strict=True):
        if len(word) != len(scores):
            raise ValueError("alignment character count does not match transcript")

    def fraction(start: int, end: int) -> float:
        selected = [
            score
            for scores in word_scores[start:end]
            for score in scores
        ]
        if not selected:
            return 0.0
        return sum(score >= score_threshold for score in selected) / len(selected)

    for start, end in entity_word_ranges.values():
        if not 0 <= start < end <= len(words):
            raise ValueError("invalid entity word range")
    return fraction(0, len(words)), {
        term: fraction(start, end)
        for term, (start, end) in entity_word_ranges.items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MMSForcedAligner:
    """Load torchaudio MMS_FA once and emit corpus-admission evidence."""

    def __init__(self, *, device: str = "cpu") -> None:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA

        self._torch = torch
        self._torchaudio = torchaudio
        self._bundle = MMS_FA
        self._device = torch.device(device)
        self._model = MMS_FA.get_model(with_star=False).to(self._device).eval()
        self._tokenizer = MMS_FA.get_tokenizer()
        self._aligner = MMS_FA.get_aligner()

        filename = Path(urlparse(MMS_FA._path).path).name
        artifact = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not artifact.is_file():
            raise RuntimeError("MMS forced-alignment model artifact is missing")
        self.adapter_revision = (
            f"torchaudio={torchaudio.__version__};"
            f"model_sha256={_sha256(artifact)}"
        )

    def _word_scores(self, audio_path: Path, words: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        import soundfile as sf

        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        waveform = self._torch.from_numpy(mono).unsqueeze(0)
        if sample_rate != self._bundle.sample_rate:
            waveform = self._torchaudio.functional.resample(
                waveform,
                sample_rate,
                self._bundle.sample_rate,
            )
        with self._torch.inference_mode():
            emission, _ = self._model(waveform.to(self._device))
        spans: list[list[Any]] = self._aligner(
            emission[0],
            self._tokenizer(list(words)),
        )
        return tuple(
            tuple(float(span.score) for span in word_spans)
            for word_spans in spans
        )

    def align(
        self,
        audio_path: Path,
        prepared: SpokenReference,
        *,
        score_threshold: float,
    ) -> AlignmentEvidence:
        """Align one exact prepared reference and summarize coverage."""
        word_scores = self._word_scores(Path(audio_path), prepared.words)
        token_coverage, entity_coverage = coverage_from_word_scores(
            prepared.words,
            word_scores,
            prepared.entity_word_ranges,
            score_threshold=score_threshold,
        )
        return AlignmentEvidence(
            reference_sha256=prepared.reference_sha256,
            token_coverage=token_coverage,
            critical_entity_coverage=entity_coverage,
            coverage_score_threshold=score_threshold,
            adapter="torchaudio-MMS_FA",
            adapter_revision=self.adapter_revision,
        )
