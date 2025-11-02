"""
Numerical Methods Core Implementations

Implements LU decomposition, Newton-Raphson, and supporting linear algebra routines.
"""

import numpy as np
import warnings
from typing import Tuple, Dict, List


def lu_decomposition_doolittle(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LU decomposition using Doolittle algorithm with partial pivoting.
    
    Decomposes A into PA = LU where:
    - P is a permutation matrix
    - L is lower triangular with ones on diagonal
    - U is upper triangular
    
    Algorithm:
        1. For each column, find pivot (largest absolute value)
        2. Swap rows if needed (partial pivoting)
        3. Eliminate elements below pivot
        4. Store multipliers in L, results in U
    
    Args:
        A: Square matrix of shape (n, n)
        
    Returns:
        (L, U, P): Lower triangular, upper triangular, and permutation matrices
        
    Complexity: O(n^3) operations, O(n^2) space
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    
    # Initialize
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)
    
    for k in range(n - 1):
        # Partial pivoting: find row with largest absolute value in column k
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        
        # Check for singular matrix (tolerance adjusted)
        if abs(U[pivot_row, k]) < 1e-12:
            # Try to continue if possible, warn user
            warnings.warn(f"Matrix may be singular or nearly singular at column {k}")
            # Don't raise error immediately - some matrices can still be solved
            if abs(U[pivot_row, k]) < 1e-15:
                raise ValueError(f"Matrix is singular at column {k}")
        
        # Swap rows in U, L, and P if needed
        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        
        # Gaussian elimination
        for i in range(k + 1, n):
            if abs(U[k, k]) > 1e-15:  # Avoid division by zero
                multiplier = U[i, k] / U[k, k]
                L[i, k] = multiplier
                U[i, k:] -= multiplier * U[k, k:]
    
    return L, U, P


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Lx = b where L is lower triangular.
    
    Algorithm: Forward substitution
        x[0] = b[0] / L[0,0]
        x[i] = (b[i] - sum(L[i,j]*x[j] for j<i)) / L[i,i]
    
    Args:
        L: Lower triangular matrix (n, n)
        b: Right-hand side vector (n,)
        
    Returns:
        Solution vector x
        
    Complexity: O(n^2)
    """
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n):
        sum_val = np.dot(L[i, :i], x[:i])
        if abs(L[i, i]) < 1e-15:
            raise ValueError(f"Division by zero at row {i}")
        x[i] = (b[i] - sum_val) / L[i, i]
    
    return x


def back_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ux = b where U is upper triangular.
    
    Algorithm: Back substitution
        x[n-1] = b[n-1] / U[n-1,n-1]
        x[i] = (b[i] - sum(U[i,j]*x[j] for j>i)) / U[i,i]
    
    Args:
        U: Upper triangular matrix (n, n)
        b: Right-hand side vector (n,)
        
    Returns:
        Solution vector x
        
    Complexity: O(n^2)
    """
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        sum_val = np.dot(U[i, i + 1:], x[i + 1:])
        if abs(U[i, i]) < 1e-15:
            raise ValueError(f"Division by zero at row {i}")
        x[i] = (b[i] - sum_val) / U[i, i]
    
    return x


def horner_eval(coeffs: List[float], x: float) -> float:
    """
    Evaluate polynomial at x using Horner's method.
    
    For p(x) = a_n*x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0,
    Horner's method computes: (...((a_n*x + a_{n-1})*x + a_{n-2})*x + ... + a_0)
    
    Args:
        coeffs: Polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
        x: Point at which to evaluate
        
    Returns:
        p(x)
        
    Complexity: O(n) with n = degree
    """
    result = 0.0
    for coeff in coeffs:
        result = result * x + coeff
    return result


def horner_eval_derivative(coeffs: List[float], x: float) -> Tuple[float, float]:
    """
    Evaluate polynomial and its derivative at x using Horner's method.
    
    Simultaneously computes p(x) and p'(x) in a single pass.
    
    Args:
        coeffs: Polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
        x: Point at which to evaluate
        
    Returns:
        (p(x), p'(x))
        
    Complexity: O(n)
    """
    n = len(coeffs) - 1
    p = coeffs[0]
    dp = 0.0
    
    for i in range(1, len(coeffs)):
        dp = dp * x + p
        p = p * x + coeffs[i]
    
    return p, dp


def newton_raphson_polynomial(
    coeffs: List[float],
    x0: float,
    max_iter: int = 100,
    tol: float = 1e-10
) -> Dict:
    """
    Find a root of polynomial using Newton-Raphson method.
    
    Iteration: x_{k+1} = x_k - f(x_k) / f'(x_k)
    
    Converges quadratically near a simple root.
    
    Args:
        coeffs: Polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
        x0: Initial guess
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Dictionary with root, iterations, convergence trace
        
    Complexity: O(k*n) where k is iterations, n is polynomial degree
    """
    x = x0
    trace = []
    
    for iteration in range(max_iter):
        # Evaluate polynomial and derivative
        p_x, dp_x = horner_eval_derivative(coeffs, x)
        
        # Record iteration
        trace.append({
            "iteration": iteration,
            "x": float(x),
            "f_x": float(p_x),
            "df_x": float(dp_x)
        })
        
        # Check convergence
        if abs(p_x) < tol:
            return {
                "root": float(x),
                "converged": True,
                "iterations": iteration + 1,
                "final_error": abs(p_x),
                "trace": trace
            }
        
        # Check if derivative is too small
        if abs(dp_x) < 1e-14:
            return {
                "root": float(x),
                "converged": False,
                "iterations": iteration + 1,
                "final_error": abs(p_x),
                "trace": trace,
                "message": "Derivative too small, Newton-Raphson may fail"
            }
        
        # Newton-Raphson update
        x_new = x - p_x / dp_x
        
        # Check for stagnation
        if abs(x_new - x) < tol:
            return {
                "root": float(x_new),
                "converged": True,
                "iterations": iteration + 1,
                "final_error": abs(p_x),
                "trace": trace
            }
        
        x = x_new
    
    # Max iterations reached
    p_x, _ = horner_eval_derivative(coeffs, x)
    return {
        "root": float(x),
        "converged": False,
        "iterations": max_iter,
        "final_error": abs(p_x),
        "trace": trace,
        "message": "Maximum iterations reached"
    }
