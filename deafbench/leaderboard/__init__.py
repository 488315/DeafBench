"""Bridges to externally pinned benchmark authorities."""

from .official import (
    OPEN_ASR_EVALUATOR_REVISION,
    OfficialEvaluator,
    OfficialEvaluatorError,
    open_asr_evaluator,
)

__all__ = [
    "OPEN_ASR_EVALUATOR_REVISION",
    "OfficialEvaluator",
    "OfficialEvaluatorError",
    "open_asr_evaluator",
]
