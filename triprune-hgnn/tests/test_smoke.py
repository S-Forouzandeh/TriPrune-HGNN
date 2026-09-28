"""Fast smoke test: every module runs and produces finite, well-shaped output.

Run with either:
    python -m pytest -q
    python tests/test_smoke.py
"""
from __future__ import annotations

import math

import numpy as np
import torch

from triprune import generate_synthetic_hypergraph, train_triprune, TrainConfig
from triprune.utils.features import structural_features, FEATURE_DIM
from triprune.models.hcp import train_hcp, HCP
from triprune.models.triprune import TriPruneHGNN


def test_features_shape():
    hg = generate_synthetic_hypergraph(n=120, K=2, seed=0)
    phi = structural_features(hg.incidence, k=hg.K)
    assert phi.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(phi))


def test_hcp_fits():
    feats = np.random.RandomState(0).randn(64, FEATURE_DIM).astype("float32")
    targets = np.clip(0.3 + 0.1 * feats[:, 3], 0.05, 0.95).astype("float32")
    model, info = train_hcp(feats, targets, epochs=100)
    assert info["train_rmse"] < 0.5
    assert 0.0 < model.predict([torch.ones(20, 5)], k=1) < 1.0


def test_forward_and_losses_finite():
    hg = generate_synthetic_hypergraph(n=150, K=3, seed=1)
    model = TriPruneHGNN(hg.d, hg.num_classes, hg.K, hidden=16)
    lam = torch.full((4,), 0.25)
    out = model(hg, r_star=0.31, lambdas=lam, beta_hcp=0.1, compute_aux=True)
    for k, v in out["losses"].items():
        assert math.isfinite(float(v.detach())), f"loss {k} not finite"
    assert out["logits"].shape == (hg.n, hg.num_classes)
    assert 0.0 <= float(out["retentions"]["overall"].detach()) <= 1.0


def test_training_runs_and_learns():
    hg = generate_synthetic_hypergraph(n=250, K=3, num_classes=4, seed=2)
    cfg = TrainConfig(epochs=40, eval_every=40, verbose=False, use_meta=True)
    res = train_triprune(hg, r_star=0.31, cfg=cfg)
    assert math.isfinite(res.final["test_acc"])
    # should beat random chance on this separable synthetic task
    assert res.final["test_acc"] > 1.0 / hg.num_classes


if __name__ == "__main__":
    test_features_shape(); print("ok features")
    test_hcp_fits(); print("ok hcp")
    test_forward_and_losses_finite(); print("ok forward")
    test_training_runs_and_learns(); print("ok training")
    print("ALL SMOKE TESTS PASSED")
