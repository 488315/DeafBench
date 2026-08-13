"""Fail-closed access to a pinned official Open ASR evaluator checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


OPEN_ASR_EVALUATOR_REVISION = "9585fc39bff55697a2ec1c5f13921b18812bfde8"
_RESULT_MARKER = "DEAFBENCH_OFFICIAL_RESULT="
_REQUIRED_FILES = (
    "normalizer/__init__.py",
    "normalizer/data_utils.py",
    "normalizer/eval_utils.py",
    "normalizer/normalizer.py",
)


class OfficialEvaluatorError(RuntimeError):
    """Raised when the external evaluator cannot be trusted or executed."""


class OfficialEvaluator:
    """Run normalization and scoring from one exact external Git revision."""

    def __init__(self, checkout: Path | str, *, expected_revision: str):
        self.checkout = Path(checkout).resolve()
        self.expected_revision = expected_revision

    def validate(self) -> None:
        """Reject missing, incomplete, or differently pinned checkouts."""
        if not self.checkout.is_dir():
            raise OfficialEvaluatorError(
                f"official evaluator checkout does not exist: {self.checkout}"
            )

        missing = [
            relative for relative in _REQUIRED_FILES
            if not (self.checkout / relative).is_file()
        ]
        if missing:
            raise OfficialEvaluatorError(
                "official evaluator checkout is incomplete: " + ", ".join(missing)
            )

        try:
            completed = subprocess.run(
                ["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise OfficialEvaluatorError(
                f"cannot verify official evaluator revision: {self.checkout}"
            ) from exc

        actual_revision = completed.stdout.strip()
        if actual_revision != self.expected_revision:
            raise OfficialEvaluatorError(
                "official evaluator revision mismatch: "
                f"expected {self.expected_revision}, got {actual_revision or 'unknown'}"
            )

        try:
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.checkout),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise OfficialEvaluatorError(
                f"cannot verify official evaluator source: {self.checkout}"
            ) from exc
        if status.stdout.strip():
            raise OfficialEvaluatorError(
                "official evaluator source is modified; use a clean pinned checkout"
            )

    def normalize(self, texts: Iterable[str]) -> list[str]:
        """Normalize text with the pinned upstream English normalizer."""
        payload = self._invoke_worker("normalize", {"texts": list(texts)})
        normalized = payload.get("normalized")
        if not isinstance(normalized, list) or not all(
            isinstance(text, str) for text in normalized
        ):
            raise OfficialEvaluatorError("official normalizer returned invalid data")
        return normalized

    def score(self, results_dir: Path | str, model_id: str) -> dict[str, Any]:
        """Score manifests by calling the pinned upstream score_results function."""
        results_path = Path(results_dir).resolve()
        if not results_path.is_dir():
            raise OfficialEvaluatorError(
                f"results directory does not exist: {results_path}"
            )
        if not model_id.strip():
            raise OfficialEvaluatorError("model_id must not be empty")

        return self._invoke_worker(
            "score",
            {"results_dir": str(results_path), "model_id": model_id},
        )

    def analyze(
        self,
        results_dir: Path | str,
        model_id: str,
        *,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Rank utterance errors using pinned normalization and alignment."""
        results_path = Path(results_dir).resolve()
        if not results_path.is_dir():
            raise OfficialEvaluatorError(
                f"results directory does not exist: {results_path}"
            )
        if not model_id.strip():
            raise OfficialEvaluatorError("model_id must not be empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise OfficialEvaluatorError("analysis limit must be positive")

        return self._invoke_worker(
            "analyze",
            {
                "results_dir": str(results_path),
                "model_id": model_id,
                "limit": limit,
            },
        )

    def _invoke_worker(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.validate()
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "deafbench.leaderboard._official_worker",
                    "--checkout",
                    str(self.checkout),
                    action,
                ],
                input=json.dumps(payload),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                check=True,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise OfficialEvaluatorError("could not launch official evaluator") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or "unknown error"
            raise OfficialEvaluatorError(
                f"official evaluator failed: {detail}"
            ) from exc

        marker_lines = [
            line.removeprefix(_RESULT_MARKER)
            for line in completed.stdout.splitlines()
            if line.startswith(_RESULT_MARKER)
        ]
        if len(marker_lines) != 1:
            raise OfficialEvaluatorError(
                "official evaluator did not return exactly one result marker"
            )
        try:
            result = json.loads(marker_lines[0])
        except json.JSONDecodeError as exc:
            raise OfficialEvaluatorError(
                "official evaluator returned malformed JSON"
            ) from exc
        if not isinstance(result, dict):
            raise OfficialEvaluatorError("official evaluator returned invalid data")
        return result


def open_asr_evaluator(checkout: Path | str) -> OfficialEvaluator:
    """Create an evaluator locked to DeafBench's reviewed Open ASR revision."""
    return OfficialEvaluator(
        checkout,
        expected_revision=OPEN_ASR_EVALUATOR_REVISION,
    )
