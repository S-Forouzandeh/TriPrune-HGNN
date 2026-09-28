"""Synthetic heterogeneous-hypergraph generator and empirical compressibility.

The generator (:func:`generate_synthetic_hypergraph`) samples graphs with
controlled structural parameters -- density, degree skewness, number of
components ``K`` and size ``n`` -- matching the corpus ranges quoted in
Section 3.1 of the paper.  Node features are class-conditional so that node
classification is learnable and message passing carries signal.

:func:`empirical_compressibility` implements the *operational* definition of the
HCP training target ``r_emp`` used in the paper: the minimum overall retention
at which task accuracy degrades by more than ``delta_acc``.  It is computed with
a **decoupled** procedure that consumes no HCP signal, which is exactly the
protocol described in Section 3.2 (``Avoiding the HCP--NAHP bootstrap
circularity``): importance-ordered structural pruning at a grid of retention
targets, with a quick classifier refit at each level.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from .hetero_hypergraph import HeteroHypergraph


def generate_synthetic_hypergraph(
    n: int = 400,
    K: int = 3,
    density: float = 0.05,
    skew: float = 1.5,
    num_classes: int = 5,
    feature_dim: int = 16,
    mean_edge_size: float = 5.0,
    homophily: float = 0.7,
    seed: int | None = None,
) -> HeteroHypergraph:
    """Sample a heterogeneous hypergraph with controlled structure.

    Parameters mirror the synthetic-corpus controls in the paper
    (``rho``, ``Sk(d)``, ``K``, ``n``).  ``homophily`` biases hyperedge
    membership toward a shared class so that message passing is informative.
    """
    rng = np.random.default_rng(seed)

    # -- node labels and class-conditional features ------------------------
    y = rng.integers(0, num_classes, size=n)
    class_means = rng.normal(0.0, 1.5, size=(num_classes, feature_dim))
    x = class_means[y] + rng.normal(0.0, 1.0, size=(n, feature_dim))

    # -- popularity weights controlling degree skew ------------------------
    # a heavier power-law exponent -> larger degree skewness
    ranks = np.arange(1, n + 1)
    exponent = 0.2 + 0.6 * np.clip(skew, 0.0, 6.0)
    pop = 1.0 / np.power(ranks, exponent)
    rng.shuffle(pop)
    pop = pop / pop.sum()

    # target total number of hyperedges from density = m / C(n,2)
    max_pairs = n * (n - 1) / 2.0
    m_total = int(np.clip(density * max_pairs, K, 4000))
    per_comp = max(1, m_total // K)

    incidence: List[np.ndarray] = []
    for _ in range(K):
        H = np.zeros((n, per_comp), dtype=np.float32)
        for e in range(per_comp):
            size = max(2, int(rng.poisson(mean_edge_size)))
            size = min(size, n)
            if rng.random() < homophily:
                # cohesive hyperedge: bias toward a seed node's class
                seed_node = rng.choice(n, p=pop)
                cls = y[seed_node]
                pool = np.where(y == cls)[0]
                if len(pool) >= size:
                    members = rng.choice(pool, size=size, replace=False)
                else:
                    members = rng.choice(n, size=size, replace=False, p=pop)
            else:
                members = rng.choice(n, size=size, replace=False, p=pop)
            H[members, e] = 1.0
        incidence.append(H)

    # -- train / val / test split -----------------------------------------
    perm = rng.permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    train_mask = np.zeros(n, dtype=bool); train_mask[perm[:n_tr]] = True
    val_mask = np.zeros(n, dtype=bool);   val_mask[perm[n_tr:n_tr + n_va]] = True
    test_mask = np.zeros(n, dtype=bool);  test_mask[perm[n_tr + n_va:]] = True

    return HeteroHypergraph(
        x=torch.tensor(x, dtype=torch.float32),
        incidence=[torch.tensor(H, dtype=torch.float32) for H in incidence],
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
        num_classes=num_classes,
        meta={"density": density, "skew": skew, "seed": seed},
    )


# ---------------------------------------------------------------------------
# Empirical compressibility (operational definition of r_emp)
# ---------------------------------------------------------------------------
def _prune_incidence(incidence, retention: float, x: torch.Tensor):
    """Importance-ordered structural pruning to an overall ``retention``.

    Decoupled from HCP: hyperedge importance is the mean pairwise cosine
    cohesion of its member features; the lowest-importance hyperedges are
    dropped until the target retention is reached.  This is the ``no HCP
    signal`` variant used to build the HCP training target.
    """
    xd = torch.nn.functional.normalize(x, dim=1)
    pruned = []
    for H in incidence:
        m = H.shape[1]
        keep = max(1, int(round(retention * m)))
        # cohesion score per hyperedge
        deg = H.sum(0).clamp(min=1.0)               # [m]
        centroid = (H.t() @ xd) / deg[:, None]      # [m, d]
        centroid = torch.nn.functional.normalize(centroid, dim=1)
        # average member-to-centroid similarity
        sim = (H.t() * (xd @ centroid.t()).t()).sum(1) / deg
        idx = torch.argsort(sim, descending=True)[:keep]
        mask = torch.zeros(m, dtype=torch.bool)
        mask[idx] = True
        pruned.append(H[:, mask])
    return pruned


@torch.no_grad()
def _quick_eval(encoder_fn, classifier, hg, incidence, device) -> float:
    from ..utils.metrics import accuracy
    z = encoder_fn(hg.x.to(device), [H.to(device) for H in incidence])
    logits = classifier(z)
    return accuracy(logits, hg.y.to(device), hg.val_mask.to(device))


def empirical_compressibility(
    hg: HeteroHypergraph,
    retention_grid=(0.9, 0.7, 0.5, 0.31, 0.2, 0.12),
    delta_acc: float = 0.01,
    encoder_hidden: int = 32,
    epochs: int = 60,
    refit_steps: int = 40,
    device: str = "cpu",
    seed: int = 0,
) -> float:
    """Return ``r_emp``: the minimum overall retention with acc drop <= delta_acc.

    Trains a small base HGNN + linear head on the full graph, then, for each
    retention level (descending), prunes structurally, refits the head briefly
    on the pruned embeddings, and measures validation accuracy.
    """
    torch.manual_seed(seed)
    from ..models.hgnn import HGNNEncoder

    hg = hg.to(device)
    enc = HGNNEncoder(hg.d, encoder_hidden, layers=2).to(device)
    head = torch.nn.Linear(encoder_hidden, hg.num_classes).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-2)
    ce = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        enc.train(); head.train(); opt.zero_grad()
        z = enc(hg.x, hg.incidence)
        loss = ce(head(z)[hg.train_mask], hg.y[hg.train_mask])
        loss.backward(); opt.step()

    from ..utils.metrics import accuracy
    enc.eval(); head.eval()
    with torch.no_grad():
        full_acc = accuracy(head(enc(hg.x, hg.incidence)), hg.y, hg.val_mask)

    r_emp = 1.0
    for r in retention_grid:
        pruned = _prune_incidence(hg.incidence, r, hg.x)
        # brief head refit on frozen encoder features
        head_r = torch.nn.Linear(encoder_hidden, hg.num_classes).to(device)
        opt_r = torch.optim.Adam(head_r.parameters(), lr=1e-2)
        with torch.no_grad():
            z = enc(hg.x, pruned)
        for _ in range(refit_steps):
            opt_r.zero_grad()
            loss = ce(head_r(z)[hg.train_mask], hg.y[hg.train_mask])
            loss.backward(); opt_r.step()
        with torch.no_grad():
            acc_r = accuracy(head_r(z), hg.y, hg.val_mask)
        if full_acc - acc_r <= delta_acc:
            r_emp = float(r)
        else:
            break
    return r_emp
