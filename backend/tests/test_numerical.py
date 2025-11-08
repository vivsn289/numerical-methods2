import numpy as np
import pytest
from app import numerical


def test_lu_decomposition_basic():
    A = np.array([[4.0, 3.0], [6.0, 3.0]])
    L, U = numerical.lu_decomposition(A)
    # L @ U should reconstruct A
    assert np.allclose(L @ U, A, atol=1e-8)


def test_triangular_solvers():
    L = np.array([[1.0, 0.0], [2.0, 1.0]])
    b = np.array([3.0, 7.0])
    y = numerical.forward_substitution(L, b)
    assert np.allclose(L @ y, b)

    U = np.array([[2.0, 1.0], [0.0, 3.0]])
    x = numerical.back_substitution(U, np.array([4.0, 6.0]))
    assert np.allclose(U @ x, np.array([4.0, 6.0]))


def test_newton_raphson_root():
    # solve x^2 - 4 = 0
    f = lambda x: x * x - 4
    df = lambda x: 2 * x
    root, info = numerical.newton_raphson(f, df, 2.0)
    assert abs(root - 2.0) < 1e-8
