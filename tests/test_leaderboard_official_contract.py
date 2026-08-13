import os
from pathlib import Path

import pytest

from deafbench.leaderboard.official import open_asr_evaluator


OFFICIAL_CHECKOUT = os.environ.get("DEAFBENCH_OPEN_ASR_REPO")


@pytest.mark.skipif(
    not OFFICIAL_CHECKOUT,
    reason="set DEAFBENCH_OPEN_ASR_REPO to the pinned official checkout",
)
def test_pinned_official_normalization_contract():
    evaluator = open_asr_evaluator(Path(OFFICIAL_CHECKOUT))

    assert evaluator.normalize(
        [
            "Dr. Smith earned $1,250.50 [noise]",
            "I'm gonna use the wi-fi at the café.",
            "Uh, twenty three percent.",
        ]
    ) == [
        "doctor smith earned $1250.50",
        "i am going to use the wifi at the cafe",
        "23%",
    ]
