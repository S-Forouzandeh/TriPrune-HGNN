"""End-to-end demo on a single synthetic heterogeneous hypergraph.

Generates a graph, obtains a compressibility prior (from a pretrained HCP if a
checkpoint is given, else a heuristic constant), trains TriPrune-HGNN through the
density curriculum, and prints validation/test metrics.

    python -m scripts.demo                 # quick run, heuristic r_star
    python -m scripts.demo --hcp checkpoints/hcp.pt --epochs 200
"""
from __future__ import annotations

import argparse

import torch

from triprune import generate_synthetic_hypergraph, train_triprune, TrainConfig
from triprune.models.hcp import HCP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--skew", type=float, default=1.5)
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hcp", type=str, default=None, help="path to a pretrained HCP checkpoint")
    ap.add_argument("--r-star", type=float, default=None, help="override compressibility prior")
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hg = generate_synthetic_hypergraph(
        n=args.n, K=args.K, density=args.density, skew=args.skew,
        num_classes=args.classes, seed=args.seed,
    )
    print(f"graph: n={hg.n} K={hg.K} m={hg.m} d={hg.d} classes={hg.num_classes}")

    if args.r_star is not None:
        r_star = args.r_star
        print(f"r_star = {r_star:.3f} (override)")
    elif args.hcp:
        hcp = HCP(); hcp.load_state_dict(torch.load(args.hcp, map_location="cpu")); hcp.eval()
        r_star = hcp.predict(hg.incidence, k=hg.K)
        print(f"r_star = {r_star:.3f} (pretrained HCP)")
    else:
        r_star = 0.31
        print(f"r_star = {r_star:.3f} (heuristic default; pretrain HCP for a data-driven prior)")

    cfg = TrainConfig(epochs=args.epochs, use_meta=not args.no_meta, seed=args.seed)
    res = train_triprune(hg, r_star, cfg)
    print("\n=== final (test) ===")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.final.items()})


if __name__ == "__main__":
    main()
