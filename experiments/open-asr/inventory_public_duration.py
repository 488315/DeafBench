"""Inventory pinned public leaderboard duration without downloading audio."""

from __future__ import annotations

import duckdb
from huggingface_hub import HfApi


DATASET_ID = "hf-audio/open-asr-leaderboard"
DATASET_REVISION = "b6bdcd0beb34f8975dc659796176d88f43aff502"
PUBLIC_SPLITS = (
    ("ami_cleaned", "test"),
    ("earnings22", "test"),
    ("gigaspeech_cleaned", "test"),
    ("librispeech", "test.clean"),
    ("librispeech", "test.other"),
    ("spgispeech", "test"),
    ("voxpopuli_cleaned_aa", "test"),
)


def main() -> None:
    files = HfApi().list_repo_files(
        DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    connection = duckdb.connect()
    total_seconds = 0.0
    for name, split in PUBLIC_SPLITS:
        prefix = f"{name}/{split}-"
        parquet_urls = [
            "https://huggingface.co/datasets/"
            f"{DATASET_ID}/resolve/{DATASET_REVISION}/{path}"
            for path in files
            if path.startswith(prefix) and path.endswith(".parquet")
        ]
        if not parquet_urls:
            raise RuntimeError(f"no Parquet shards found for {name}/{split}")
        samples, seconds = connection.execute(
            "SELECT count(*), sum(audio_length_s) FROM read_parquet($files)",
            {"files": parquet_urls},
        ).fetchone()
        total_seconds += seconds
        print(f"{name}/{split}: samples={samples} hours={seconds / 3600:.3f}")

    print(f"total_hours={total_seconds / 3600:.3f}")
    print(f"estimated_gpu_hours_at_rtfx_20={total_seconds / 3600 / 20:.3f}")


if __name__ == "__main__":
    main()
