"""
Numerical Gauss–Legendre quadrature, large-n stable, iterative.
"""

import numpy as np
from numpy.linalg import eigh
import base64, io
import matplotlib.pyplot as plt
from typing import Callable, Optional, Dict, Any

ProgressCB = Optional[Callable[[int, Optional[str]], None]]


def golub_welsch(n: int, progress_callback: ProgressCB = None):
    if n < 1:
        raise ValueError("n must be ≥ 1")
    J = np.zeros((n, n))
    i = np.arange(1, n)
    beta = i / np.sqrt(4*i*i - 1)
    np.fill_diagonal(J[1:], beta)
    np.fill_diagonal(J[:, 1:], beta)
    vals, vecs = eigh(J)
    nodes = vals
    weights = 2 * (vecs[0, :] ** 2)
    if progress_callback:
        progress_callback(100, f"computed n={n}")
    return nodes, weights


def collocation_matrices_plots_base64(nodes):
    D = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                D[i, j] = 1 / (nodes[i] - nodes[j])
        D[i, i] = -np.sum(D[i])
    D2 = D @ D
    out = {}
    for name, M in (("D", D), ("D2", D2)):
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(M, aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(name)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        out[name] = base64.b64encode(buf.getvalue()).decode()
    return out


def compute_quadrature_pipeline(n: int, progress_callback: ProgressCB = None) -> Dict[str, Any]:
    nodes, weights = golub_welsch(n, progress_callback)
    plots = collocation_matrices_plots_base64(nodes)
    return {"n": n, "nodes": nodes.tolist(), "weights": weights.tolist(), "plots": plots}
