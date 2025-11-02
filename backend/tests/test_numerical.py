"""
Unit tests for numerical methods core functions.
"""

import pytest
import numpy as np
from app.numerical import (
    lu_decomposition_doolittle,
    forward_substitution,
    back_substitution,
    horner_eval,
    horner_eval_derivative,
    newton_raphson_polynomial
)


class TestLUDecomposition:
    """Test LU decomposition implementation."""
    
    def test_simple_matrix(self):
        """Test LU decomposition of a simple matrix."""
        A = np.array([
            [2.0, 1.0],
            [1.0, 2.0]
        ])
        
        L, U, P = lu_decomposition_doolittle(A)
        
        # Check dimensions
        assert L.shape == (2, 2)
        assert U.shape == (2, 2)
        assert P.shape == (2, 2)
        
        # Check that L is lower triangular with ones on diagonal
        assert abs(L[0, 0] - 1.0) < 1e-10
        assert abs(L[1, 1] - 1.0) < 1e-10
        assert abs(L[0, 1]) < 1e-10
        
        # Check that U is upper triangular
        assert abs(U[1, 0]) < 1e-10
        
        # Check PA = LU
        reconstructed = L @ U
        original = P @ A
        assert np.allclose(reconstructed, original)
    
    def test_3x3_matrix(self):
        """Test 3×3 matrix decomposition."""
        A = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0]
        ])
        
        L, U, P = lu_decomposition_doolittle(A)
        
        # Verify PA = LU
        assert np.allclose(P @ A, L @ U, atol=1e-10)
    
    def test_random_matrix(self):
        """Test with random matrix."""
        np.random.seed(42)
        A = np.random.rand(5, 5)
        
        L, U, P = lu_decomposition_doolittle(A)
        
        # Verify decomposition
        assert np.allclose(P @ A, L @ U, atol=1e-10)
    
    def test_singular_matrix(self):
        """Test that singular matrix raises error."""
        A = np.array([
            [1.0, 2.0],
            [2.0, 4.0]  # Second row is multiple of first
        ])
        
        with pytest.raises(ValueError, match="singular"):
            lu_decomposition_doolittle(A)


class TestTriangularSolvers:
    """Test forward and back substitution."""
    
    def test_forward_substitution(self):
        """Test forward substitution with lower triangular matrix."""
        L = np.array([
            [2.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [4.0, 2.0, 5.0]
        ])
        b = np.array([2.0, 5.0, 12.0])
        
        x = forward_substitution(L, b)
        
        # Verify Lx = b
        assert np.allclose(L @ x, b)
    
    def test_back_substitution(self):
        """Test back substitution with upper triangular matrix."""
        U = np.array([
            [2.0, 1.0, 3.0],
            [0.0, 3.0, 2.0],
            [0.0, 0.0, 5.0]
        ])
        b = np.array([11.0, 8.0, 5.0])
        
        x = back_substitution(U, b)
        
        # Verify Ux = b
        assert np.allclose(U @ x, b)
    
    def test_identity_matrix(self):
        """Test with identity matrix."""
        I = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        
        x_forward = forward_substitution(I, b)
        x_back = back_substitution(I, b)
        
        # Both should give x = b
        assert np.allclose(x_forward, b)
        assert np.allclose(x_back, b)


class TestHornerEvaluation:
    """Test Horner's method for polynomial evaluation."""
    
    def test_simple_polynomial(self):
        """Test p(x) = x^2 + 2x + 1."""
        coeffs = [1.0, 2.0, 1.0]  # x^2 + 2x + 1
        
        # Evaluate at x = 2: 4 + 4 + 1 = 9
        result = horner_eval(coeffs, 2.0)
        assert abs(result - 9.0) < 1e-10
        
        # Evaluate at x = 0: 1
        result = horner_eval(coeffs, 0.0)
        assert abs(result - 1.0) < 1e-10
    
    def test_linear_polynomial(self):
        """Test p(x) = 3x + 5."""
        coeffs = [3.0, 5.0]
        
        result = horner_eval(coeffs, 2.0)
        assert abs(result - 11.0) < 1e-10
    
    def test_constant_polynomial(self):
        """Test p(x) = 7."""
        coeffs = [7.0]
        
        result = horner_eval(coeffs, 100.0)
        assert abs(result - 7.0) < 1e-10


class TestHornerDerivative:
    """Test Horner's method with derivative."""
    
    def test_quadratic(self):
        """Test p(x) = x^2 + 2x + 1, p'(x) = 2x + 2."""
        coeffs = [1.0, 2.0, 1.0]
        
        p_val, dp_val = horner_eval_derivative(coeffs, 2.0)
        
        # p(2) = 9, p'(2) = 6
        assert abs(p_val - 9.0) < 1e-10
        assert abs(dp_val - 6.0) < 1e-10
    
    def test_cubic(self):
        """Test p(x) = x^3 + 3x^2 + 3x + 1, p'(x) = 3x^2 + 6x + 3."""
        coeffs = [1.0, 3.0, 3.0, 1.0]
        
        p_val, dp_val = horner_eval_derivative(coeffs, 1.0)
        
        # p(1) = 8, p'(1) = 12
        assert abs(p_val - 8.0) < 1e-10
        assert abs(dp_val - 12.0) < 1e-10
    
    def test_at_zero(self):
        """Test evaluation at x = 0."""
        coeffs = [2.0, 3.0, 5.0]  # 2x^2 + 3x + 5
        
        p_val, dp_val = horner_eval_derivative(coeffs, 0.0)
        
        # p(0) = 5, p'(0) = 3
        assert abs(p_val - 5.0) < 1e-10
        assert abs(dp_val - 3.0) < 1e-10


class TestNewtonRaphson:
    """Test Newton-Raphson method."""
    
    def test_simple_root(self):
        """Test finding root of x^2 - 4 (roots at ±2)."""
        coeffs = [1.0, 0.0, -4.0]  # x^2 - 4
        
        result = newton_raphson_polynomial(coeffs, x0=3.0, tol=1e-10)
        
        assert result["converged"]
        assert abs(result["root"] - 2.0) < 1e-8
    
    def test_negative_root(self):
        """Test finding negative root."""
        coeffs = [1.0, 0.0, -4.0]  # x^2 - 4
        
        result = newton_raphson_polynomial(coeffs, x0=-3.0, tol=1e-10)
        
        assert result["converged"]
        assert abs(result["root"] - (-2.0)) < 1e-8
    
    def test_cubic_root(self):
        """Test finding root of x^3 - 8 (root at 2)."""
        coeffs = [1.0, 0.0, 0.0, -8.0]  # x^3 - 8
        
        result = newton_raphson_polynomial(coeffs, x0=3.0, tol=1e-10)
        
        assert result["converged"]
        assert abs(result["root"] - 2.0) < 1e-8
    
    def test_convergence_trace(self):
        """Test that iteration trace is recorded."""
        coeffs = [1.0, 0.0, -4.0]
        
        result = newton_raphson_polynomial(coeffs, x0=3.0)
        
        assert "trace" in result
        assert len(result["trace"]) > 0
        assert result["trace"][0]["iteration"] == 0
    
    def test_max_iterations(self):
        """Test that max iterations limit works."""
        coeffs = [1.0, 0.0, -4.0]
        
        result = newton_raphson_polynomial(coeffs, x0=3.0, max_iter=2)
        
        assert result["iterations"] <= 2


class TestNumericalAccuracy:
    """Test numerical accuracy of methods."""
    
    def test_lu_solve_accuracy(self):
        """Test solving linear system with LU."""
        A = np.array([
            [4.0, 3.0],
            [6.0, 3.0]
        ])
        b = np.array([7.0, 9.0])
        
        L, U, P = lu_decomposition_doolittle(A)
        
        # Solve using substitutions
        y = forward_substitution(L, P @ b)
        x = back_substitution(U, y)
        
        # Verify solution
        assert np.allclose(A @ x, b, atol=1e-10)
    
    def test_polynomial_evaluation_consistency(self):
        """Test that Horner evaluation is consistent."""
        coeffs = [2.0, -3.0, 0.0, 5.0, -1.0]
        x = 1.5
        
        # Evaluate using Horner
        p_horner = horner_eval(coeffs, x)
        
        # Evaluate directly
        p_direct = coeffs[0] * x**4 + coeffs[1] * x**3 + coeffs[2] * x**2 + coeffs[3] * x + coeffs[4]
        
        assert abs(p_horner - p_direct) < 1e-10
