"""Training entry point.

Trains TriPrune-HGNN on a dataset selected by ``--dataset``.  ``synthetic`` runs
out of the box; the real benchmarks (IMDB / DBLP / Yelp / Amazon / Douban and the
out-of-domain NTU2012 / ModelNet40 / House) are wired through
``load_real_dataset`` below, which you should implement to return a
:class:`triprune.HeteroHypergraph`.  The datasets are not bundled with this
repository (see README).

    python -m scripts.train --dataset synthetic --epochs 200 --hcp checkpoints/hcp.pt
"""
from __future__ import annotations

import argparse

import torch
import yaml

from triprune import generate_synthetic_hypergraph, train_triprune, TrainConfig
from triprune.models.hcp import HCP


def load_real_dataset(name: str):
    """Return a HeteroHypergraph for a named real benchmark.

    Implement this for your data. Expected output: a HeteroHypergraph with
    ``x``, ``incidence`` (one matrix per component), ``y`` and train/val/test
    masks. The five paper benchmarks use the heterogeneous-hypergraph
    formulation of Kim et al. (2024) (one hyperedge per node per component).
    """
    raise NotImplementedError(
        f"Real dataset '{name}' is not bundled. Implement load_real_dataset() to "
        f"return a HeteroHypergraph (see README, 'Using your own data')."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="synthetic")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--hcp", type=str, default=None)
    ap.add_argument("--r-star", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        c = yaml.safe_load(f)
    cfg = TrainConfig(**{k: v for k, v in c.get("train", {}).items()})
    if args.epochs is not None:
        cfg.epochs = args.epochs
    cfg.seed = args.seed

    if args.dataset == "synthetic":
        s = c.get("synthetic", {})
        hg = generate_synthetic_hypergraph(seed=args.seed, **s)
    else:
        hg = load_real_dataset(args.dataset)

    if args.r_star is not None:
        r_star = args.r_star
    elif args.hcp:
        hcp = HCP(); hcp.load_state_dict(torch.load(args.hcp, map_location="cpu")); hcp.eval()
        r_star = hcp.predict(hg.incidence, k=hg.K)
    else:
        r_star = 0.31
    print(f"dataset={args.dataset}  r_star={r_star:.3f}")

    res = train_triprune(hg, r_star, cfg)
    print("final:", {k: round(v, 4) if isinstance(v, float) else v for k, v in res.final.items()})


if __name__ == "__main__":
    main()
