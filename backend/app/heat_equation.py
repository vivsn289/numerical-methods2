"""
Heat equation solver (numerical vs analytic), stable for large n.
"""

import numpy as np, math, base64, io
import matplotlib.pyplot as plt
from typing import Dict, Optional, Callable, Any
from .gaussian_quadrature import golub_welsch

ProgressCB = Optional[Callable[[int, Optional[str]], None]]


def integrate_exp_neg_sqr_0_to_b(b: float, n: int) -> float:
    nodes, weights = golub_welsch(n)
    s = 0.5 * b * (nodes + 1)
    return 0.5 * b * np.dot(weights, np.exp(-s**2))


def compute_heat_pipeline(n: int = 32, eta_max: float = 3.0, progress_callback: ProgressCB = None) -> Dict[str, Any]:
    etas = np.linspace(0, eta_max, 250)
    numeric = []
    analytic = []
    total = len(etas)
    for i, eta in enumerate(etas):
        numeric.append((2 / math.sqrt(math.pi)) * integrate_exp_neg_sqr_0_to_b(abs(eta), n))
        analytic.append(math.erf(eta))
        if progress_callback and i % max(1, total // 100) == 0:
            progress_callback(int(100 * i / total), f"η={eta:.3f}")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(etas, analytic, '--', label="analytic")
    ax.plot(etas, numeric, '-', label=f"n={n} numeric")
    ax.legend(); ax.grid(True)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return {"etas": etas.tolist(), "numeric": numeric, "analytic": analytic,
            "plot": base64.b64encode(buf.getvalue()).decode()}
