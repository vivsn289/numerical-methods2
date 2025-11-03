"""
Gaussian Quadrature Implementation using Golub-Welsch Algorithm

Implements Gauss-Legendre quadrature by computing nodes (roots) and weights
from the Jacobi matrix eigenvalue problem as described in the 1969 paper by
Golub and Welsch.
"""

import numpy as np
from typing import Dict, Tuple, List
import scipy.linalg


def build_jacobi_matrix(n: int) -> np.ndarray:
    """
    Build the Jacobi matrix for Gauss-Legendre quadrature.
    
    For Legendre polynomials, the Jacobi matrix is symmetric tridiagonal:
    - Diagonal elements are all 0
    - Off-diagonal elements are β_i = i/sqrt(4i² - 1)
    
    The eigenvalues of this matrix give the quadrature nodes,
    and the weights can be computed from the eigenvectors.
    
    Args:
        n: Number of quadrature points (degree n+1 rule)
        
    Returns:
        Symmetric tridiagonal Jacobi matrix of size n×n
    """
    J = np.zeros((n, n))
    
    # Off-diagonal elements
    for i in range(1, n):
        beta = i / np.sqrt(4 * i**2 - 1)
        J[i-1, i] = beta
        J[i, i-1] = beta
    
    return J


def compute_gauss_legendre_golub_welsch(n: int) -> Dict:
    """
    Compute Gauss-Legendre nodes and weights using Golub-Welsch algorithm.
    
    Algorithm:
    1. Construct the Jacobi matrix J
    2. Compute eigenvalues (nodes) and eigenvectors of J
    3. Weights are computed as w_i = 2 * v_i[0]² where v_i is the i-th eigenvector
    
    Reference: G. H. Golub and J. H. Welsch, "Calculation of Gauss Quadrature Rules,"
               Math. Comp. 23 (1969) 221–230
    
    Args:
        n: Number of quadrature points
        
    Returns:
        Dictionary with nodes, weights, and verification info
    """
    # Build Jacobi matrix
    J = build_jacobi_matrix(n)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    
    # Nodes are the eigenvalues (already sorted by eigh)
    nodes = eigenvalues
    
    # Weights from eigenvector norms
    # For Legendre: w_i = 2 * (first component of eigenvector)²
    weights = 2.0 * eigenvectors[0, :]**2
    
    return {
        "nodes": nodes.tolist(),
        "weights": weights.tolist(),
        "num_points": n,
        "method": "golub_welsch",
        "jacobi_matrix": J.tolist()
    }


def lagrange_weights(nodes: np.ndarray) -> np.ndarray:
    """
    Compute Gauss-Legendre weights using Lagrangian interpolation.
    
    For Gauss-Legendre quadrature, weights can also be computed as:
    w_i = ∫_{-1}^{1} L_i(x) dx
    where L_i(x) is the Lagrange basis polynomial.
    
    For Legendre, there's a formula: w_i = 2 / [(1 - x_i²) * [P'_n(x_i)]²]
    
    Args:
        nodes: Quadrature nodes (roots of Legendre polynomial)
        
    Returns:
        Array of weights
    """
    n = len(nodes)
    weights = np.zeros(n)
    
    # Evaluate derivative of Legendre polynomial at nodes
    # Using the property: P'_n(x) = n * [P_{n-1}(x) - x*P_n(x)] / (1 - x²)
    # But P_n(x_i) = 0 at roots, so P'_n(x_i) = n * P_{n-1}(x_i) / (1 - x_i²)
    
    for i, x in enumerate(nodes):
        # Compute P_{n-1}(x) using numpy
        P_prev = np.polynomial.legendre.legval(x, [0]*(n-1) + [1])
        
        # Weight formula
        weights[i] = 2.0 / ((1 - x**2) * P_prev**2)
    
    return weights


def orthogonal_collocation_matrices(nodes: np.ndarray, weights: np.ndarray) -> Dict:
    """
    Compute A and B matrices for orthogonal collocation.
    
    The A matrix represents the derivative operator:
    A_ij = dL_j/dx(x_i) where L_j are Lagrange basis polynomials
    
    The B matrix represents the second derivative operator:
    B_ij = d²L_j/dx²(x_i)
    
    Args:
        nodes: Collocation points (Gauss-Legendre nodes)
        weights: Quadrature weights
        
    Returns:
        Dictionary with A and B matrices
    """
    n = len(nodes)
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    
    # Compute A matrix (first derivative)
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal elements
                sum_val = 0
                for k in range(n):
                    if k != i:
                        sum_val += 1.0 / (nodes[i] - nodes[k])
                A[i, j] = sum_val
            else:
                # Off-diagonal elements
                prod = 1.0
                for k in range(n):
                    if k != j:
                        prod *= (nodes[i] - nodes[k])
                
                prod_j = 1.0
                for k in range(n):
                    if k != j:
                        prod_j *= (nodes[j] - nodes[k])
                
                A[i, j] = prod_j / prod / (nodes[i] - nodes[j])
    
    # Compute B matrix (second derivative)
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal elements for second derivative
                sum1 = 0
                for k in range(n):
                    if k != i:
                        sum2 = 0
                        for m in range(n):
                            if m != i and m != k:
                                sum2 += 1.0 / (nodes[i] - nodes[m])
                        sum1 += sum2 / (nodes[i] - nodes[k])
                B[i, j] = 2 * sum1
            else:
                # Off-diagonal elements
                sum_val = 0
                for k in range(n):
                    if k != j:
                        sum_val += 1.0 / (nodes[i] - nodes[k])
                
                B[i, j] = 2 * A[i, j] * (sum_val - 1.0 / (nodes[i] - nodes[j]))
    
    return {
        "A_matrix": A.tolist(),
        "B_matrix": B.tolist(),
        "A_shape": list(A.shape),
        "B_shape": list(B.shape),
        "A_norm": float(np.linalg.norm(A)),
        "B_norm": float(np.linalg.norm(B))
    }


def compute_up_to_n(max_n: int = 64) -> Dict:
    """
    Compute Gauss-Legendre nodes and weights for n = 1 to max_n.
    
    Args:
        max_n: Maximum number of quadrature points
        
    Returns:
        Dictionary with results for each n
    """
    results = {}
    
    for n in range(1, max_n + 1):
        try:
            result = compute_gauss_legendre_golub_welsch(n)
            results[n] = {
                "nodes": result["nodes"],
                "weights": result["weights"],
                "sum_weights": sum(result["weights"])  # Should be 2 for [-1,1]
            }
        except Exception as e:
            results[n] = {"error": str(e)}
    
    return {
        "results": results,
        "max_n": max_n,
        "summary": f"Computed Gauss-Legendre quadrature for n=1 to {max_n}"
    }


def test_quadrature(nodes: List[float], weights: List[float], test_func=None) -> Dict:
    """
    Test the quadrature rule on a known integral.
    
    Tests ∫_{-1}^{1} f(x) dx using the quadrature rule.
    
    Args:
        nodes: Quadrature nodes
        weights: Quadrature weights
        test_func: Function to integrate (default: lambda x: x**2)
        
    Returns:
        Dictionary with test results
    """
    if test_func is None:
        # Default: integrate x^2 from -1 to 1 = 2/3
        test_func = lambda x: x**2
        exact = 2.0 / 3.0
    
    # Compute quadrature approximation
    approx = sum(w * test_func(x) for x, w in zip(nodes, weights))
    
    error = abs(approx - exact)
    relative_error = error / abs(exact) if exact != 0 else error
    
    return {
        "exact": exact,
        "approximate": approx,
        "absolute_error": error,
        "relative_error": relative_error,
        "num_points": len(nodes)
    }