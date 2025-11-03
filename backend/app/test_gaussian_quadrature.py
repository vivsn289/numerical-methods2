"""
Unit tests for Gaussian quadrature implementation.
"""

import pytest
import numpy as np
from app.gaussian_quadrature import (
    build_jacobi_matrix,
    compute_gauss_legendre_golub_welsch,
    lagrange_weights,
    orthogonal_collocation_matrices,
    test_quadrature
)


class TestJacobiMatrix:
    """Test Jacobi matrix construction."""
    
    def test_matrix_shape(self):
        """Test that Jacobi matrix has correct shape."""
        n = 5
        J = build_jacobi_matrix(n)
        assert J.shape == (n, n)
    
    def test_matrix_symmetry(self):
        """Test that Jacobi matrix is symmetric."""
        J = build_jacobi_matrix(10)
        assert np.allclose(J, J.T)
    
    def test_diagonal_zeros(self):
        """Test that diagonal elements are zero for Legendre."""
        J = build_jacobi_matrix(8)
        assert np.allclose(np.diag(J), 0)
    
    def test_tridiagonal(self):
        """Test that matrix is tridiagonal."""
        n = 6
        J = build_jacobi_matrix(n)
        
        # Check that elements far from diagonal are zero
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    assert abs(J[i, j]) < 1e-15


class TestGolubWelsch:
    """Test Golub-Welsch algorithm."""
    
    def test_n2_nodes(self):
        """Test 2-point Gauss-Legendre nodes."""
        result = compute_gauss_legendre_golub_welsch(2)
        nodes = np.array(result["nodes"])
        
        # Nodes should be ±1/sqrt(3)
        expected = 1.0 / np.sqrt(3)
        assert abs(nodes[0] - (-expected)) < 1e-10
        assert abs(nodes[1] - expected) < 1e-10
    
    def test_n2_weights(self):
        """Test 2-point Gauss-Legendre weights."""
        result = compute_gauss_legendre_golub_welsch(2)
        weights = np.array(result["weights"])
        
        # Both weights should be 1.0
        assert abs(weights[0] - 1.0) < 1e-10
        assert abs(weights[1] - 1.0) < 1e-10
    
    def test_weights_sum(self):
        """Test that weights sum to 2 (integral of 1 over [-1,1])."""
        for n in [3, 5, 10, 20]:
            result = compute_gauss_legendre_golub_welsch(n)
            weights = np.array(result["weights"])
            assert abs(np.sum(weights) - 2.0) < 1e-10
    
    def test_nodes_in_interval(self):
        """Test that all nodes are in [-1, 1]."""
        result = compute_gauss_legendre_golub_welsch(15)
        nodes = np.array(result["nodes"])
        
        assert np.all(nodes >= -1.0)
        assert np.all(nodes <= 1.0)
    
    def test_nodes_symmetric(self):
        """Test that nodes are symmetric about 0."""
        result = compute_gauss_legendre_golub_welsch(10)
        nodes = np.array(result["nodes"])
        
        # Nodes should come in pairs (x, -x)
        n = len(nodes)
        for i in range(n // 2):
            assert abs(nodes[i] + nodes[n - 1 - i]) < 1e-10
    
    def test_positive_weights(self):
        """Test that all weights are positive."""
        result = compute_gauss_legendre_golub_welsch(20)
        weights = np.array(result["weights"])
        
        assert np.all(weights > 0)


class TestLagrangeWeights:
    """Test Lagrangian weight computation."""
    
    def test_matches_golub_welsch(self):
        """Test that Lagrange weights match Golub-Welsch."""
        n = 5
        result = compute_gauss_legendre_golub_welsch(n)
        nodes = np.array(result["nodes"])
        weights_gw = np.array(result["weights"])
        
        weights_lag = lagrange_weights(nodes)
        
        # Should be very close
        assert np.allclose(weights_gw, weights_lag, rtol=1e-8)
    
    def test_multiple_n(self):
        """Test Lagrange weights for various n."""
        for n in [3, 8, 16]:
            result = compute_gauss_legendre_golub_welsch(n)
            nodes = np.array(result["nodes"])
            weights_gw = np.array(result["weights"])
            weights_lag = lagrange_weights(nodes)
            
            assert np.allclose(weights_gw, weights_lag, rtol=1e-6)


class TestCollocationMatrices:
    """Test orthogonal collocation matrices."""
    
    def test_matrix_shapes(self):
        """Test that A and B matrices have correct shapes."""
        n = 8
        result = compute_gauss_legendre_golub_welsch(n)
        nodes = np.array(result["nodes"])
        weights = np.array(result["weights"])
        
        matrices = orthogonal_collocation_matrices(nodes, weights)
        
        assert matrices["A_shape"] == [n, n]
        assert matrices["B_shape"] == [n, n]
    
    def test_matrix_exists(self):
        """Test that matrices are computed successfully."""
        n = 5
        result = compute_gauss_legendre_golub_welsch(n)
        nodes = np.array(result["nodes"])
        weights = np.array(result["weights"])
        
        matrices = orthogonal_collocation_matrices(nodes, weights)
        
        A = np.array(matrices["A_matrix"])
        B = np.array(matrices["B_matrix"])
        
        assert A.shape == (n, n)
        assert B.shape == (n, n)
        assert not np.any(np.isnan(A))
        assert not np.any(np.isnan(B))


class TestQuadratureAccuracy:
    """Test quadrature accuracy on known integrals."""
    
    def test_constant_function(self):
        """Test ∫_{-1}^{1} 1 dx = 2."""
        n = 2
        result = compute_gauss_legendre_golub_welsch(n)
        nodes = result["nodes"]
        weights = result["weights"]
        
        integral = sum(w * 1.0 for w in weights)
        assert abs(integral - 2.0) < 1e-12
    
    def test_linear_function(self):
        """Test ∫_{-1}^{1} x dx = 0."""
        n = 2
        result = compute_gauss_legendre_golub_welsch(n)
        nodes = result["nodes"]
        weights = result["weights"]
        
        integral = sum(w * x for w, x in zip(weights, nodes))
        assert abs(integral) < 1e-12
    
    def test_quadratic_function(self):
        """Test ∫_{-1}^{1} x^2 dx = 2/3."""
        test_result = test_quadrature([0.0], [2.0])  # Will use default test
        
        n = 3
        result = compute_gauss_legendre_golub_welsch(n)
        nodes = result["nodes"]
        weights = result["weights"]
        
        integral = sum(w * x**2 for w, x in zip(weights, nodes))
        expected = 2.0 / 3.0
        assert abs(integral - expected) < 1e-12
    
    def test_high_degree_polynomial(self):
        """Test that n-point rule is exact for polynomials of degree ≤ 2n-1."""
        n = 5  # Should be exact for degree ≤ 9
        result = compute_gauss_legendre_golub_welsch(n)
        nodes = result["nodes"]
        weights = result["weights"]
        
        # Test x^8 (degree 8, should be exact)
        integral = sum(w * x**8 for w, x in zip(weights, nodes))
        expected = 2.0 / 9.0  # ∫_{-1}^{1} x^8 dx = 2/9
        assert abs(integral - expected) < 1e-12


class TestNumericalStability:
    """Test numerical stability for high n."""
    
    def test_n32(self):
        """Test n=32 computation."""
        result = compute_gauss_legendre_golub_welsch(32)
        
        assert len(result["nodes"]) == 32
        assert len(result["weights"]) == 32
        assert abs(sum(result["weights"]) - 2.0) < 1e-10
    
    def test_n64(self):
        """Test n=64 computation."""
        result = compute_gauss_legendre_golub_welsch(64)
        
        assert len(result["nodes"]) == 64
        assert len(result["weights"]) == 64
        assert abs(sum(result["weights"]) - 2.0) < 1e-9


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow_n32(self):
        """Test complete workflow for n=32."""
        n = 32
        
        # Compute nodes and weights
        result = compute_gauss_legendre_golub_welsch(n)
        assert result["num_points"] == n
        
        # Compute collocation matrices
        nodes = np.array(result["nodes"])
        weights = np.array(result["weights"])
        matrices = orthogonal_collocation_matrices(nodes, weights)
        
        assert matrices["A_shape"] == [n, n]
        assert matrices["B_shape"] == [n, n]
        
        # Test quadrature accuracy
        test_result = test_quadrature(result["nodes"], result["weights"])
        assert test_result["relative_error"] < 1e-10