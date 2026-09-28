"""Graph-level structural features for the Hypergraph Compressibility Predictor.

Implements the 10-dimensional descriptor ``phi(H)`` of Eq. (hcp_features):

    phi = [ n, m, mean_edge_degree, density, skew(deg), cv(deg),
            spectral_gap, avg_clustering, entropy(edge_size), K ]

All features are computed from structure alone -- no node features, no labels --
which is what makes the predictor transferable across domains (Section 3.1).
Complexity is O(n + m) except the spectral gap (top eigen-pairs of the
normalised hypergraph Laplacian) and the clustering coefficient, both of which
are computed once per graph.
"""
from __future__ import annotations

from typing import List

import numpy as np

FEATURE_NAMES = [
    "n", "m", "mean_edge_degree", "density", "skew_deg", "cv_deg",
    "spectral_gap", "avg_clustering", "entropy_edge_size", "K",
]
FEATURE_DIM = len(FEATURE_NAMES)


def _incidence_to_numpy(incidence) -> List[np.ndarray]:
    out = []
    for H in incidence:
        if hasattr(H, "detach"):
            H = H.detach().cpu().numpy()
        out.append((np.asarray(H) > 0).astype(np.float64))
    return out


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu, sd = x.mean(), x.std()
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 3))


def _cv(x: np.ndarray) -> float:
    mu = x.mean()
    if abs(mu) < 1e-12:
        return 0.0
    return float(x.std() / mu)


def _entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _spectral_gap(Hs: List[np.ndarray]) -> float:
    """Second-smallest eigenvalue of the normalised hypergraph Laplacian.

    We build the (clique-reduced) node operator
    ``P = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}`` aggregated over components, then
    ``L = I - P`` and return its second-smallest eigenvalue.
    """
    n = Hs[0].shape[0]
    P = np.zeros((n, n), dtype=np.float64)
    for H in Hs:
        de = H.sum(axis=0)            # hyperedge degree
        de_inv = np.divide(1.0, de, out=np.zeros_like(de), where=de > 0)
        dv = H.sum(axis=1)            # node degree in this component
        # node operator for this component
        Pk = H @ (de_inv[:, None] * H.T)   # [n, n]
        dv = Pk.sum(axis=1)
        dv_inv_sqrt = np.divide(1.0, np.sqrt(dv), out=np.zeros_like(dv), where=dv > 0)
        Pk = dv_inv_sqrt[:, None] * Pk * dv_inv_sqrt[None, :]
        P += Pk
    if len(Hs) > 0:
        P /= len(Hs)
    L = np.eye(n) - P
    L = 0.5 * (L + L.T)              # symmetrise against numerical drift
    try:
        w = np.linalg.eigvalsh(L)
        w = np.sort(w)
        return float(w[1]) if n > 1 else 0.0
    except np.linalg.LinAlgError:
        return 0.0


def _avg_clustering(Hs: List[np.ndarray]) -> float:
    """Average clustering coefficient of the clique-expansion graph."""
    n = Hs[0].shape[0]
    A = np.zeros((n, n), dtype=np.float64)
    for H in Hs:
        A += H @ H.T
    np.fill_diagonal(A, 0.0)
    A = (A > 0).astype(np.float64)
    deg = A.sum(axis=1)
    # number of closed triangles through each node = diag(A^3) / 2
    tri = np.diag(A @ A @ A) / 2.0
    denom = deg * (deg - 1.0)
    cc = np.divide(2.0 * tri, denom, out=np.zeros_like(tri), where=denom > 0)
    return float(cc.mean()) if n > 0 else 0.0


def structural_features(incidence, k: int | None = None) -> np.ndarray:
    """Return the 10-dim structural descriptor ``phi(H)`` as a float64 vector."""
    Hs = _incidence_to_numpy(incidence)
    n = Hs[0].shape[0]
    m = int(sum(H.shape[1] for H in Hs))
    K = len(Hs) if k is None else k

    edge_sizes = np.concatenate([H.sum(axis=0) for H in Hs]) if m > 0 else np.zeros(1)
    node_deg = np.sum([H.sum(axis=1) for H in Hs], axis=0)

    mean_edge_degree = float(edge_sizes.mean()) if m > 0 else 0.0
    density = float(m / max(1.0, n * (n - 1) / 2.0))

    phi = np.array([
        float(n),
        float(m),
        mean_edge_degree,
        density,
        _skew(node_deg),
        _cv(node_deg),
        _spectral_gap(Hs),
        _avg_clustering(Hs),
        _entropy(np.bincount(edge_sizes.astype(int)) if m > 0 else np.zeros(1)),
        float(K),
    ], dtype=np.float64)
    return phi
