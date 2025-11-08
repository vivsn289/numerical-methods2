import numpy as np
from app import heat_equation


def test_heat_pipeline_basic():
    res = heat_equation.compute_heat_pipeline(n=8, eta_max=2.0)
    assert "etas" in res and "numeric" in res and "analytic" in res and "plot" in res
    etas = np.array(res["etas"])
    numeric = np.array(res["numeric"])
    analytic = np.array(res["analytic"])
    assert etas.size == numeric.size == analytic.size
    # rough agreement
    assert np.mean(np.abs(numeric - analytic)) < 0.2
