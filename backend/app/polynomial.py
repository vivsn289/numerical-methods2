"""
Polynomial computations robust for large n.
"""

import numpy as np
import sympy as sp
from typing import List, Dict, Tuple, Optional, Callable
import scipy.linalg as la

ProgressCB = Optional[Callable[[int, Optional[str]], None]]
x = sp.symbols("x")


def legendre_coeffs_iterative(n: int, progress_callback: ProgressCB = None) -> List[float]:
    """Compute coefficients iteratively with recurrence."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [1.0, 0.0]
    P0 = [1.0]
    P1 = [1.0, 0.0]
    for k in range(1, n):
        if progress_callback:
            progress_callback(int(50 * k / n), f"recurrence step {k}")
        Pn = np.polynomial.polynomial.polymul([((2*k+1)/(k+1)), 0], P1)
        Pn = np.polynomial.polynomial.polyadd(Pn, np.polynomial.polynomial.polymul([-k/(k+1)], P0))
        P0, P1 = P1, Pn
    coeffs = list(P1[::-1])
    return coeffs


def companion_matrix(coeffs: List[float]) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    n = len(coeffs) - 1
    if n <= 0:
        return np.zeros((0, 0))
    a = coeffs / coeffs[0]
    C = np.zeros((n, n))
    C[0, :] = -a[1:]
    for i in range(1, n):
        C[i, i - 1] = 1.0
    return C


def roots_via_companion(C: np.ndarray) -> np.ndarray:
    if C.size == 0:
        return np.array([])
    vals = la.eigvals(C)
    return np.real_if_close(vals)


def safe_lu_solve(A, b) -> Tuple[np.ndarray, Dict]:
    try:
        lu, piv = la.lu_factor(A)
        x = la.lu_solve((lu, piv), b)
        return x, {"method": "lu", "shape": A.shape}
    except la.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0], {"method": "lstsq"}


def compute_polynomial_pipeline(order: int, progress_callback: ProgressCB = None) -> Dict[str, any]:
    order = int(order)
    if order > 256:
        raise ValueError("Polynomial order too large (>256) for numerical stability.")
    coeffs = legendre_coeffs_iterative(order, progress_callback)
    if progress_callback:
        progress_callback(60, "building companion matrix")
    C = companion_matrix(coeffs)
    roots = roots_via_companion(C)
    n = C.shape[0]
    b = np.arange(1, n + 1, dtype=float)
    if progress_callback:
        progress_callback(75, "solving Ax=b")
    xsol, info = safe_lu_solve(C, b)
    return {
        "order": order,
        "coefficients": coeffs,
        "roots": roots.tolist(),
        "Ax_b_solution": xsol.tolist(),
        "lu_info": info
    }
