import pytest

from deafbench.benchmark.windows_sapi import WindowsSapiGenerator


@pytest.mark.parametrize("rate", (-11, 11))
def test_sapi_rejects_rate_outside_platform_contract(rate: int):
    with pytest.raises(ValueError, match="rate"):
        WindowsSapiGenerator(rate=rate)


def test_sapi_rejects_sample_speech_rate_outside_platform_contract():
    with pytest.raises(ValueError, match="sample speech rates"):
        WindowsSapiGenerator(sample_speech_rates={"core-011": 11})
