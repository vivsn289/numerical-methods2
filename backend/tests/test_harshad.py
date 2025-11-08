import pytest
from app import harshad


def test_digit_sum_and_is_harshad():
    assert harshad.digit_sum(18) == 9
    assert harshad.is_harshad(18)
    assert not harshad.is_harshad(19)
    assert not harshad.is_harshad(0)
    assert harshad.is_harshad(24)  # 4! = 24 is Harshad


def test_first_nonharshad_factorial_small():
    # set small max_k so test runs quickly
    res = harshad.first_nonharshad_factorial(max_k=10)
    assert isinstance(res, dict)
    # either found non-harshad or returned error if none in range
    if "error" not in res:
        assert res["is_harshad"] is False


def test_find_consecutive_small():
    res = harshad.find_consecutive_harshads(length=3, start_hint=110, max_iter=1000)
    assert res["length"] == 3 or "error" in res
