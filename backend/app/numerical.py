"""
Core numerical methods — LU decomposition, Newton-Raphson.
"""

import numpy as np
import scipy.linalg as la
from typing import Callable, Tuple, Dict

def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for i in range(n):
        if abs(U[i, i]) < 1e-14:
            raise ValueError("Matrix is singular or near singular.")
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, :] -= L[j, i] * U[i, :]
    return L, U


def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y


def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y)
    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x


def newton_raphson(func: Callable[[float], float], dfunc: Callable[[float], float],
                   x0: float, tol: float = 1e-12, max_iter: int = 1000) -> Tuple[float, Dict]:
    x = x0
    trace = []
    for k in range(max_iter):
        fx = func(x)
        dfx = dfunc(x)
        if abs(dfx) < 1e-14:
            break
        x_new = x - fx / dfx
        trace.append({"k": k, "x": x, "f(x)": fx})
        if abs(x_new - x) < tol:
            return x_new, {"iterations": k, "trace": trace}
        x = x_new
    return x, {"iterations": max_iter, "trace": trace}
