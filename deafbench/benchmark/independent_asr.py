"""Independent Wav2Vec2 transcription evidence for synthetic admission."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse


def collapse_ctc_labels(
    indices: Sequence[int],
    labels: Sequence[str],
    *,
    blank: int,
) -> str:
    """Greedily decode CTC labels without a language model."""
    collapsed: list[int] = []
    previous: int | None = None
    for index in indices:
        if not 0 <= index < len(labels):
            raise ValueError(f"CTC label index is out of range: {index}")
        if index != previous:
            collapsed.append(index)
        previous = index
    text = "".join(labels[index] for index in collapsed if index != blank)
    return " ".join(text.replace("|", " ").casefold().split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Wav2Vec2IndependentASR:
    """Greedy LibriSpeech ASR used only as supporting validator evidence."""

    def __init__(self, *, device: str = "cpu") -> None:
        import torch
        import torchaudio
        from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

        self._torch = torch
        self._torchaudio = torchaudio
        self._bundle = WAV2VEC2_ASR_BASE_960H
        self._device = torch.device(device)
        self._model = WAV2VEC2_ASR_BASE_960H.get_model().to(self._device).eval()
        self._labels = WAV2VEC2_ASR_BASE_960H.get_labels()

        filename = Path(urlparse(WAV2VEC2_ASR_BASE_960H._path).path).name
        artifact = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not artifact.is_file():
            raise RuntimeError("independent ASR model artifact is missing")
        self.adapter_revision = (
            f"torchaudio={torchaudio.__version__};"
            f"model_sha256={_sha256(artifact)}"
        )

    def transcribe(self, audio_path: Path) -> str:
        """Return a greedy transcript with no tuned decoder or language model."""
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
        indices = emission[0].argmax(dim=-1).cpu().tolist()
        return collapse_ctc_labels(indices, self._labels, blank=0)
