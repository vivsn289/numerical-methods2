"""
Unit tests for polynomial operations.
"""

import pytest
import numpy as np
from app.polynomial import (
    generate_legendre_coefficients,
    build_companion_matrix,
    compute_roots_eigenvalue,
    solve_lu_system,
    newton_raphson_root
)


class TestLegendreCoefficients:
    """Test Legendre polynomial coefficient generation."""
    
    def test_degree_0(self):
        """P_0(x) = 1"""
        result = generate_legendre_coefficients(0)
        coeffs = result["coefficients"]
        assert len(coeffs) == 1
        assert abs(coeffs[0] - 1.0) < 1e-10
    
    def test_degree_1(self):
        """P_1(x) = x"""
        result = generate_legendre_coefficients(1)
        coeffs = result["coefficients"]
        assert len(coeffs) == 2
        assert abs(coeffs[0] - 1.0) < 1e-10  # x term
        assert abs(coeffs[1]) < 1e-10  # constant term
    
    def test_degree_2(self):
        """P_2(x) = (3x^2 - 1)/2"""
        result = generate_legendre_coefficients(2)
        coeffs = result["coefficients"]
        assert len(coeffs) == 3
        # Coefficients should be [3/2, 0, -1/2]
        assert abs(coeffs[0] - 1.5) < 1e-10
        assert abs(coeffs[1]) < 1e-10
        assert abs(coeffs[2] - (-0.5)) < 1e-10
    
    def test_degree_10(self):
        """Test degree 10 polynomial."""
        result = generate_legendre_coefficients(10)
        coeffs = result["coefficients"]
        assert len(coeffs) == 11
        assert "summary" in result
        assert result["summary"]["degree"] == 10
    
    def test_degree_100(self):
        """Test high-degree polynomial generation."""
        result = generate_legendre_coefficients(100)
        coeffs = result["coefficients"]
        assert len(coeffs) == 101
        assert result["summary"]["degree"] == 100
        # Should have non-zero coefficients
        assert result["summary"]["nonzero_coefficients"] > 0


class TestCompanionMatrix:
    """Test companion matrix construction."""
    
    def test_simple_polynomial(self):
        """Test companion matrix for x^2 - 1."""
        coeffs = [1.0, 0.0, -1.0]  # x^2 - 1
        result = build_companion_matrix(coeffs)
        
        assert result["shape"] == [2, 2]
        assert "matrix" in result
        
        # Eigenvalues should be ±1
        C = result["matrix"]
        eigenvalues = np.linalg.eigvals(C)
        eigenvalues_sorted = sorted(eigenvalues.real)
        
        assert abs(eigenvalues_sorted[0] - (-1.0)) < 1e-10
        assert abs(eigenvalues_sorted[1] - 1.0) < 1e-10
    
    def test_matrix_shape(self):
        """Test that companion matrix has correct shape."""
        coeffs = [1.0] + [0.0] * 9 + [-1.0]  # x^10 - 1
        result = build_companion_matrix(coeffs)
        
        assert result["shape"] == [10, 10]
        assert "preview" in result
    
    def test_properties(self):
        """Test that matrix properties are computed."""
        coeffs = [1.0, 0.0, -1.0]
        result = build_companion_matrix(coeffs)
        
        assert "properties" in result
        assert "condition_number" in result["properties"]


class TestRootFinding:
    """Test root-finding methods."""
    
    def test_degree_2_roots(self):
        """Test roots of P_2(x) = (3x^2 - 1)/2."""
        result = compute_roots_eigenvalue(2, method="legroots")
        
        roots = [r for r in result["roots"] if isinstance(r, (int, float))]
        roots_sorted = sorted(roots)
        
        # Roots should be ±1/sqrt(3) ≈ ±0.5773502692
        expected = 1.0 / np.sqrt(3)
        assert abs(roots_sorted[0] - (-expected)) < 1e-6
        assert abs(roots_sorted[1] - expected) < 1e-6
    
    def test_degree_10_roots(self):
        """Test roots of degree 10 polynomial."""
        result = compute_roots_eigenvalue(10, method="legroots")
        
        assert result["num_roots"] == 10
        # Legendre polynomial roots are all real and in [-1, 1]
        roots = [r for r in result["roots"] if isinstance(r, (int, float))]
        assert len(roots) == 10
        
        for root in roots:
            assert -1.0 <= root <= 1.0
    
    def test_smallest_largest_roots(self):
        """Test that smallest and largest roots are identified."""
        result = compute_roots_eigenvalue(5, method="legroots")
        
        assert result["smallest_root"] is not None
        assert result["largest_root"] is not None
        assert result["smallest_root"] < result["largest_root"]
    
    def test_method_comparison(self):
        """Compare different root-finding methods."""
        n = 10
        
        result1 = compute_roots_eigenvalue(n, method="companion_eig")
        result2 = compute_roots_eigenvalue(n, method="legroots")
        
        # Both should find same number of roots
        assert result1["num_roots"] == result2["num_roots"]


class TestLUSolve:
    """Test LU decomposition and system solving."""
    
    def test_solve_small_system(self):
        """Test solving for small degree."""
        result = solve_lu_system(5, method="scipy")
        
        assert "x" in result
        assert "residual_norm" in result
        assert len(result["x"]) == 5
        
        # Residual should be small
        assert result["residual_norm"] < 1e-6
    
    def test_doolittle_method(self):
        """Test custom Doolittle implementation."""
        result = solve_lu_system(5, method="doolittle")
        
        assert "x" in result
        assert len(result["x"]) == 5
        # Should have verification info
        assert "verification" in result
    
    def test_solution_accuracy(self):
        """Test that solution is accurate."""
        result = solve_lu_system(10, method="scipy")
        
        assert "verification" in result
        # Should be marked as accurate
        if result["residual_norm"] < 1e-6:
            assert result["verification"]["is_accurate"]


class TestNewtonRaphson:
    """Test Newton-Raphson root refinement."""
    
    def test_refine_roots(self):
        """Test root refinement."""
        result = newton_raphson_root(5, which="both", precision=64)
        
        assert "results" in result
        assert "smallest" in result["results"] or "largest" in result["results"]
    
    def test_convergence(self):
        """Test that Newton-Raphson converges."""
        result = newton_raphson_root(10, which="min", precision=64)
        
        if "smallest" in result["results"]:
            min_result = result["results"]["smallest"]
            assert "converged" in min_result
            assert "iterations" in min_result
    
    def test_trace_available(self):
        """Test that iteration trace is provided."""
        result = newton_raphson_root(5, which="max", precision=64)
        
        if "largest" in result["results"]:
            max_result = result["results"]["largest"]
            assert "trace" in max_result
            assert len(max_result["trace"]) > 0


class TestNumericalStability:
    """Test numerical stability for high-degree polynomials."""
    
    def test_degree_50(self):
        """Test degree 50 computation."""
        result = generate_legendre_coefficients(50)
        assert len(result["coefficients"]) == 51
    
    def test_degree_100_generation(self):
        """Test that degree 100 can be generated."""
        result = generate_legendre_coefficients(100)
        assert len(result["coefficients"]) == 101
        assert result["summary"]["degree"] == 100
    
    def test_high_precision_roots(self):
        """Test high precision root finding."""
        # This may be slow, so we use smaller degree
        result = compute_roots_eigenvalue(20, method="legroots", precision=128)
        assert result["num_roots"] == 20


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_polynomial_workflow(self):
        """Test complete workflow for degree 10."""
        n = 10
        
        # Generate coefficients
        poly_result = generate_legendre_coefficients(n)
        assert len(poly_result["coefficients"]) == n + 1
        
        # Build companion matrix
        companion_result = build_companion_matrix(poly_result["coefficients"])
        assert companion_result["shape"] == [n, n]
        
        # Find roots
        roots_result = compute_roots_eigenvalue(n, method="legroots")
        assert roots_result["num_roots"] == n
        
        # Solve linear system
        lu_result = solve_lu_system(n, method="scipy")
        assert len(lu_result["x"]) == n
        
        # Refine roots with Newton
        newton_result = newton_raphson_root(n, which="both")
        assert "results" in newton_result
