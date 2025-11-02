"""
Polynomial Operations for Legendre Polynomials

Implements generation of Legendre polynomial coefficients, companion matrices,
and various root-finding methods for high-degree polynomials (up to degree 100).
"""

import numpy as np
import scipy.linalg
import mpmath as mp
from typing import Dict, List, Tuple, Optional
import warnings


def generate_legendre_coefficients(n: int, normalized: bool = False) -> Dict:
    """
    Generate coefficients of the Legendre polynomial P_n(x).
    
    Uses numpy.polynomial.legendre for numerical stability.
    Legendre polynomials are orthogonal on [-1, 1] with weight function w(x) = 1.
    
    Recurrence relation:
        (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
    with P_0(x) = 1, P_1(x) = x
    
    Args:
        n: Polynomial degree
        normalized: If True, normalize so that P_n(1) = 1 (already satisfied)
        
    Returns:
        Dictionary with coefficients and summary
        
    Complexity: O(n^2) for coefficient generation
    """
    # Use numpy's Legendre polynomial generator
    # Returns coefficients in increasing degree order: [a0, a1, a2, ..., an]
    leg_coeffs = np.polynomial.legendre.Legendre.basis(n).convert(kind=np.polynomial.Polynomial).coef
    
    # Convert to standard polynomial form (decreasing powers)
    coeffs = leg_coeffs[::-1].tolist()
    
    # Summary statistics
    nonzero = np.count_nonzero(leg_coeffs)
    max_coeff = float(np.max(np.abs(leg_coeffs)))
    min_nonzero = float(np.min(np.abs(leg_coeffs[leg_coeffs != 0]))) if nonzero > 0 else 0
    
    summary = {
        "degree": n,
        "num_coefficients": len(coeffs),
        "nonzero_coefficients": int(nonzero),
        "max_abs_coefficient": max_coeff,
        "min_nonzero_coefficient": min_nonzero,
        "leading_coefficient": float(coeffs[0]),
        "constant_term": float(coeffs[-1])
    }
    
    return {
        "coefficients": coeffs,
        "summary": summary
    }


def build_companion_matrix(coeffs: List[float]) -> Dict:
    """
    Build the companion matrix for a polynomial.
    
    For polynomial p(x) = a_n*x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0,
    the companion matrix is an n×n matrix whose eigenvalues are the roots of p(x).
    
    Companion matrix form (for monic polynomial x^n + c_{n-1}*x^{n-1} + ... + c_0):
        [ 0   1   0   ...  0  ]
        [ 0   0   1   ...  0  ]
        [ :   :   :   ...  :  ]
        [ 0   0   0   ...  1  ]
        [-c0 -c1 -c2  ... -c_{n-1}]
    
    Args:
        coeffs: Polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
        
    Returns:
        Dictionary with matrix preview and properties
        
    Complexity: O(n^2) space and construction
    """
    n = len(coeffs) - 1
    
    # Normalize to monic polynomial (leading coefficient = 1)
    if abs(coeffs[0]) < 1e-15:
        raise ValueError("Leading coefficient cannot be zero")
    
    normalized_coeffs = [c / coeffs[0] for c in coeffs]
    
    # Build companion matrix
    C = np.zeros((n, n))
    
    # Superdiagonal of ones
    for i in range(n - 1):
        C[i, i + 1] = 1.0
    
    # Last row contains negated coefficients (except leading)
    # normalized_coeffs = [1, c_{n-1}, c_{n-2}, ..., c_1, c_0]
    # Last row should be [-c_0, -c_1, -c_2, ..., -c_{n-1}]
    for i in range(n):
        C[n - 1, i] = -normalized_coeffs[n - i]
    
    # Preview: first 6×6 block
    preview_size = min(6, n)
    preview = C[:preview_size, :preview_size].tolist()
    
    # Matrix properties
    try:
        cond_num = np.linalg.cond(C)
    except:
        cond_num = float('inf')
    
    properties = {
        "condition_number": float(cond_num),
        "norm": float(np.linalg.norm(C, 'fro')),
        "determinant_estimate": "Product of roots",
        "is_singular": bool(np.abs(np.linalg.det(C)) < 1e-10) if n <= 20 else "Not computed for large n"
    }
    
    return {
        "shape": [n, n],
        "preview": preview,
        "properties": properties,
        "matrix": C  # Full matrix for internal use
    }


def compute_roots_eigenvalue(
    n: int,
    method: str = "companion_eig",
    precision: int = 64,
    normalized: bool = False
) -> Dict:
    """
    Compute roots of Legendre polynomial using eigenvalue methods.
    
    Methods:
        - "companion_eig": Eigenvalues of companion matrix using numpy
        - "legroots": Direct computation using numpy.polynomial.legendre
        - "mpmath": High-precision computation using mpmath
    
    Args:
        n: Polynomial degree
        method: Root-finding method
        precision: Bits of precision for mpmath method
        normalized: Whether to normalize polynomial
        
    Returns:
        Dictionary with roots, error estimates, and statistics
        
    Complexity: O(n^3) for eigenvalue computation
    """
    poly_data = generate_legendre_coefficients(n, normalized)
    coeffs = poly_data["coefficients"]
    
    warnings_list = []
    
    if method == "companion_eig":
        # Build companion matrix and compute eigenvalues
        try:
            companion_data = build_companion_matrix(coeffs)
            C = companion_data["matrix"]
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(C)
            
            # Filter to real roots (imaginary part < tolerance)
            tol = 1e-10
            roots = []
            for ev in eigenvalues:
                if abs(ev.imag) < tol:
                    roots.append(float(ev.real))
                else:
                    roots.append(complex(ev))
            
            roots = sorted(roots, key=lambda x: x.real if isinstance(x, complex) else x)
            
            if n >= 50:
                warnings_list.append("Companion matrix method may be numerically unstable for degree >= 50")
        except Exception as e:
            warnings_list.append(f"Companion method failed: {e}. Falling back to legroots.")
            method = "legroots"  # Fallback
    
    if method == "legroots":
        # Use numpy's specialized Legendre root finder
        # numpy stores Legendre coefficients differently
        leg_coeffs = np.zeros(n + 1)
        leg_coeffs[n] = 1  # Coefficient for P_n
        roots = np.polynomial.legendre.legroots(leg_coeffs).tolist()
        roots = sorted(roots)
    
    elif method == "mpmath":
        # High-precision computation using mpmath
        mp.mp.dps = int(precision * 0.301)  # Convert bits to decimal places
        
        # Use mpmath to find roots
        # For Legendre polynomials, roots are symmetric about 0 in [-1, 1]
        leg_coeffs_mp = [mp.mpf(c) for c in coeffs]
        
        try:
            roots_mp = mp.polyroots(leg_coeffs_mp)
            roots = [float(r.real) if abs(r.imag) < mp.mpf(10)**(-precision//4) else complex(r) 
                     for r in roots_mp]
            roots = sorted(roots, key=lambda x: x.real if isinstance(x, complex) else x)
        except Exception as e:
            warnings_list.append(f"mpmath root-finding failed: {e}")
            # Fallback to numpy method
            leg_coeffs = np.zeros(n + 1)
            leg_coeffs[n] = 1
            roots = np.polynomial.legendre.legroots(leg_coeffs).tolist()
    
    # Extract smallest and largest real roots
    real_roots = [r for r in roots if not isinstance(r, complex)]
    
    if real_roots:
        smallest_root = min(real_roots)
        largest_root = max(real_roots)
    else:
        smallest_root = None
        largest_root = None
        warnings_list.append("No real roots found")
    
    # Error estimation: evaluate polynomial at roots
    error_estimates = []
    for root in roots[:min(10, len(roots))]:  # Check first 10 roots
        if isinstance(root, complex):
            continue
        # Evaluate polynomial at root using Horner's method
        val = 0
        for coeff in coeffs:
            val = val * root + coeff
        error_estimates.append({
            "root": float(root),
            "p_of_root": abs(val),
            "relative_error": abs(val) / (1 + abs(root))
        })
    
    return {
        "roots": [float(r) if not isinstance(r, complex) else {"real": r.real, "imag": r.imag} 
                  for r in roots],
        "smallest_root": float(smallest_root) if smallest_root is not None else None,
        "largest_root": float(largest_root) if largest_root is not None else None,
        "num_roots": len(roots),
        "num_real_roots": len(real_roots),
        "error_estimates": error_estimates,
        "warnings": warnings_list
    }


def solve_lu_system(n: int, method: str = "doolittle", normalized: bool = False) -> Dict:
    """
    Solve Ax = b using LU decomposition where A is the companion matrix.
    
    b = [1, 2, 3, ..., n]
    
    Implements Doolittle LU decomposition with partial pivoting for pedagogy,
    then validates against scipy.linalg.lu_solve.
    
    LU Decomposition: A = PLU where P is permutation, L is lower triangular, U is upper triangular.
    Solve: Ly = Pb, then Ux = y
    
    Args:
        n: Polynomial degree (determines matrix size)
        method: "doolittle" or "scipy"
        normalized: Polynomial normalization flag
        
    Returns:
        Solution vector x, residual norm, and verification
        
    Complexity: O(n^3) for LU decomposition and back-substitution
    """
    from app.numerical import lu_decomposition_doolittle, forward_substitution, back_substitution
    
    # Generate companion matrix
    poly_data = generate_legendre_coefficients(n, normalized)
    coeffs = poly_data["coefficients"]
    companion_data = build_companion_matrix(coeffs)
    A = companion_data["matrix"]
    
    # Right-hand side vector
    b = np.arange(1, n + 1, dtype=float)
    
    # Check if matrix is singular before attempting LU
    try:
        det = np.linalg.det(A)
        if abs(det) < 1e-12:
            warnings.warn("Matrix appears to be singular or nearly singular")
    except:
        pass
    
    if method == "doolittle":
        # Custom Doolittle implementation
        try:
            L, U, P = lu_decomposition_doolittle(A)
            
            # Solve Ly = Pb
            Pb = P @ b
            y = forward_substitution(L, Pb)
            
            # Solve Ux = y
            x = back_substitution(U, y)
            
            method_used = "doolittle (custom implementation)"
        except Exception as e:
            # Fallback to scipy if custom fails
            try:
                x = scipy.linalg.solve(A, b)
                method_used = f"scipy (fallback, doolittle failed: {str(e)[:50]})"
            except Exception as e2:
                # If even scipy fails, return NaN solution
                x = np.full(n, np.nan)
                method_used = f"Failed: {str(e2)[:50]}"
    
    elif method == "scipy":
        # Use scipy's robust LU solver
        try:
            lu, piv = scipy.linalg.lu_factor(A)
            x = scipy.linalg.lu_solve((lu, piv), b)
            method_used = "scipy.linalg.lu_solve"
        except Exception as e:
            # Matrix is singular
            x = np.full(n, np.nan)
            method_used = f"Failed: {str(e)[:50]}"
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute residual: ||Ax - b||
    try:
        residual = A @ x - b
        residual_norm = float(np.linalg.norm(residual))
        max_error = float(np.max(np.abs(residual)))
    except:
        residual_norm = float('nan')
        max_error = float('nan')
    
    # Condition number
    try:
        cond_num = float(np.linalg.cond(A))
    except:
        cond_num = float('inf')
    
    # Verification: check if x satisfies Ax ≈ b
    is_accurate = not np.isnan(residual_norm) and residual_norm < 1e-6 * np.linalg.norm(b)
    relative_error = residual_norm / np.linalg.norm(b) if not np.isnan(residual_norm) else float('nan')
    
    return {
        "x": x.tolist(),
        "solution_x": x.tolist(),  # Alias for compatibility
        "residual_norm": residual_norm,
        "max_error": max_error,
        "condition_number": cond_num,
        "method_used": method_used,
        "verification": {
            "is_accurate": is_accurate,
            "relative_error": relative_error
        }
    }


def newton_raphson_root(
    n: int,
    which: str = "both",
    precision: int = 64,
    normalized: bool = False,
    initial_guesses: Optional[List[float]] = None
) -> Dict:
    """
    Refine smallest and/or largest roots using Newton-Raphson method.
    
    Newton-Raphson iteration: x_{k+1} = x_k - f(x_k) / f'(x_k)
    
    Uses eigenvalues as initial guesses if not provided.
    Evaluates polynomial and derivative using Horner's method.
    
    Args:
        n: Polynomial degree
        which: "min", "max", or "both"
        precision: Bits of precision
        normalized: Polynomial normalization
        initial_guesses: Optional initial guesses [min_guess, max_guess]
        
    Returns:
        Refined roots with iteration trace
        
    Complexity: O(k*n) where k is number of iterations, n is polynomial degree
    """
    from app.numerical import newton_raphson_polynomial
    
    # Get polynomial coefficients
    poly_data = generate_legendre_coefficients(n, normalized)
    coeffs = poly_data["coefficients"]
    
    # Get initial guesses from eigenvalues if not provided
    if initial_guesses is None:
        roots_data = compute_roots_eigenvalue(n, method="legroots", precision=precision, normalized=normalized)
        roots = [r for r in roots_data["roots"] if isinstance(r, (int, float))]
        
        if roots:
            min_initial = min(roots)
            max_initial = max(roots)
        else:
            # Fallback: Legendre roots are in [-1, 1]
            min_initial = -0.99
            max_initial = 0.99
    else:
        min_initial = initial_guesses[0]
        max_initial = initial_guesses[1] if len(initial_guesses) > 1 else initial_guesses[0]
    
    results = {}
    
    if which in ["min", "both"]:
        # Refine smallest root
        min_result = newton_raphson_polynomial(coeffs, min_initial, max_iter=100, tol=10**(-precision//4))
        results["smallest"] = min_result
    
    if which in ["max", "both"]:
        # Refine largest root
        max_result = newton_raphson_polynomial(coeffs, max_initial, max_iter=100, tol=10**(-precision//4))
        results["largest"] = max_result
    
    return {"results": results}
