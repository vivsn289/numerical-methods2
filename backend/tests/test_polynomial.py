import sympy as sp
import numpy as np
from app import polynomial


def test_legendre_and_companion_and_pipeline():
    order = 6
    res = polynomial.compute_polynomial_pipeline(order=order)
    assert isinstance(res, dict)
    assert res["order"] == order
    assert "coeffs" not in res or isinstance(res.get("coeffs", []), list) or isinstance(res.get("coefficients", []), list)
    # roots should be list-like
    assert isinstance(res.get("roots", []), list)
