import numpy as np
from app import gaussian_quadrature


def test_golub_welsch_basic():
    n = 8
    nodes, weights = gaussian_quadrature.golub_welsch(n)
    assert len(nodes) == n and len(weights) == n
    # weights sum should be approximately 2.0
    assert abs(np.sum(weights) - 2.0) < 1e-8


def test_collocation_plots_return_dict():
    nodes, _ = gaussian_quadrature.golub_welsch(6)
    out = gaussian_quadrature.collocation_matrices_plots_base64(nodes)
    assert isinstance(out, dict)
    assert "D" in out and "D2" in out
