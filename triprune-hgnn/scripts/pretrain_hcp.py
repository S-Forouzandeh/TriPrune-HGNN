"""Pre-train the Hypergraph Compressibility Predictor on a synthetic corpus.

Builds ``--corpus`` synthetic graphs spanning controlled structural ranges,
computes the empirical compressibility ``r_emp`` for each with the *decoupled*
(no-HCP) procedure, extracts the 10-dim structural descriptor, and fits HCP by
MSE.  The trained weights are saved to ``--out`` and can be passed to the demo /
training scripts as a frozen prior.

    python -m scripts.pretrain_hcp --corpus 200 --out checkpoints/hcp.pt

The paper uses 5,000 graphs; the default here is small so the script finishes
quickly on CPU.  Increase --corpus for a stronger predictor.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from triprune.data.synthetic import generate_synthetic_hypergraph, empirical_compressibility
from triprune.models.hcp import train_hcp
from triprune.utils.features import structural_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--out", type=str, default="checkpoints/hcp.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    feats, targets = [], []
    print(f"Building corpus of {args.corpus} synthetic hypergraphs ...")
    for i in range(args.corpus):
        n = int(rng.integers(150, 600))
        K = int(rng.integers(2, 6))
        density = float(rng.uniform(0.01, 0.12))
        skew = float(rng.uniform(0.0, 5.0))
        classes = int(rng.integers(3, 8))
        hg = generate_synthetic_hypergraph(
            n=n, K=K, density=density, skew=skew, num_classes=classes, seed=int(rng.integers(1 << 30))
        )
        phi = structural_features(hg.incidence, k=hg.K)
        r_emp = empirical_compressibility(hg, seed=args.seed)
        feats.append(phi); targets.append(r_emp)
        if (i + 1) % max(1, args.corpus // 10) == 0:
            print(f"  {i+1:4d}/{args.corpus}  last r_emp={r_emp:.3f}")

    feats = np.stack(feats); targets = np.array(targets)
    model, info = train_hcp(feats, targets, epochs=args.epochs, verbose=True)
    print(f"HCP train RMSE = {info['train_rmse']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved HCP weights -> {args.out}")


if __name__ == "__main__":
    main()
