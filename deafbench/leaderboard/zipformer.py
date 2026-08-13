"""Pinned data and checkpoint boundary for the official Zipformer runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


_OFFICIAL_DATASET_SPLITS = frozenset(
    {
        ("ami_cleaned", "test"),
        ("earnings22", "test"),
        ("gigaspeech_cleaned", "test"),
        ("librispeech", "test.clean"),
        ("librispeech", "test.other"),
        ("spgispeech", "test"),
        ("voxpopuli_cleaned_aa", "test"),
    }
)
@dataclass(frozen=True)
class PinnedZipformerContract:
    """Resolve the public baseline without mutable Hugging Face revisions."""

    dataset_id: str = "hf-audio/open-asr-leaderboard"
    dataset_revision: str = "b6bdcd0beb34f8975dc659796176d88f43aff502"
    model_id: str = "soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M"
    model_revision: str = "d410fb15a71cbf87ec5e0a860356563deb9d8f01"

    def validate_dataset(self, dataset: str, split: str) -> None:
        """Reject dataset selections outside the reviewed seven-set contract."""
        if (dataset, split) not in _OFFICIAL_DATASET_SPLITS:
            raise ValueError(
                "unsupported official dataset/split: " f"{dataset}/{split}"
            )

    def load_dataset(self, loader: Callable[..., Any], args: Any) -> Any:
        """Load one official split anonymously at the reviewed revision."""
        if args.dataset_path != self.dataset_id:
            raise ValueError(f"unsupported leaderboard dataset: {args.dataset_path}")
        self.validate_dataset(args.dataset, args.split)
        return loader(
            self.dataset_id,
            args.dataset,
            revision=self.dataset_revision,
            split=args.split,
            streaming=args.streaming,
            token=False,
        )

    def snapshot_model(
        self,
        downloader: Callable[..., Any],
        model_id: str,
        **kwargs: Any,
    ) -> Any:
        """Download only the reviewed baseline checkpoint revision."""
        if model_id != self.model_id:
            raise ValueError(f"unsupported Zipformer model: {model_id}")
        return downloader(model_id, revision=self.model_revision, **kwargs)
