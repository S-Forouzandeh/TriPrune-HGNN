# TriPrune-HGNN

Reference implementation of **TriPrune-HGNN — Adaptive Hypergraph Pruning with
Learned Threshold Control and Attention-Based Negative Mining** (TMLR).

The method turns hypergraph pruning into a *learned, operating-point* problem: a
compressibility predictor estimates how far a graph can be pruned *before*
training, a hierarchical controller performs the pruning with differentiable
gates, an attention-based contrastive term repairs the representation damage
pruning causes, and a meta-learner balances the objectives. The four parts are
trained jointly under a density curriculum.

This repository is a faithful, self-contained implementation of those mechanisms
that **runs end-to-end on synthetic data out of the box**. Please read
[Scope and limitations](#scope-and-limitations) before using it to reproduce
paper numbers.

---

## Method → code map

| Paper component | Section | Code |
|---|---|---|
| Structural descriptor `phi(H)` (10-dim) | 3.1 | `triprune/utils/features.py` |
| Hypergraph Compressibility Predictor (HCP) | 3.1 | `triprune/models/hcp.py` |
| Empirical compressibility `r_emp` (decoupled target) | 3.2 | `triprune/data/synthetic.py::empirical_compressibility` |
| Neural Adaptive Hierarchical Pruning (NAHP) | 3.2 | `triprune/models/pruning.py` |
| Soft gate / hard-sigmoid STE (`Eq. unified_gate`) | 3.2 | `pruning.py::soft_gate` |
| Threshold controllers + graph state (`Eq. neural_threshold`, `graph_state`) | 3.2 | `pruning.py::ThresholdController`, `NAHP._state` |
| Base hypergraph message passing (`Eq. hgnn_prop`) | 2 | `triprune/models/hgnn.py` |
| Attention-based contrastive mining (false / hard negatives) | 3.3 | `triprune/models/contrastive.py` |
| Meta-learned loss balancing + finite-difference hypergradient | 3.4 | `triprune/training/meta.py` |
| Density-curriculum training loop (Algorithm 1) | 3.5 | `triprune/training/curriculum.py` |
| Full model | 3 | `triprune/models/triprune.py` |

---

## Installation

```bash
git clone <your-fork-url> triprune-hgnn
cd triprune-hgnn
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt                        # torch, numpy, scipy, pyyaml
# or: pip install -e .
```

CPU is sufficient for the synthetic demo. For GPU, install the CUDA build of
PyTorch that matches your driver.

---

## Quickstart

**1. Train on a synthetic hypergraph (no downloads needed):**

```bash
PYTHONPATH=. python -m scripts.demo --n 400 --K 3 --epochs 200
```

Typical output — validation accuracy rises while the overall retention drifts to
the predicted operating point and the meta-weights adapt:

```
[ 20/200] loss=1.65 val_acc=0.475 ret=0.151 lambda=[0.234, 0.241, 0.252, 0.272]
[ 80/200] loss=0.95 val_acc=0.800 ret=0.317 lambda=[0.219, 0.365, 0.000, 0.417]
```

**2. Pre-train the compressibility predictor, then use it as the prior:**

```bash
PYTHONPATH=. python -m scripts.pretrain_hcp --corpus 2000 --out checkpoints/hcp.pt
PYTHONPATH=. python -m scripts.demo --hcp checkpoints/hcp.pt --epochs 200
```

`pretrain_hcp` builds a synthetic corpus, computes each graph's empirical
compressibility `r_emp` with the **decoupled** (no-HCP) procedure described in
Section 3.2, and fits the predictor by MSE. Use a few thousand graphs for a
predictor that generalises — a tiny corpus will overfit.

**3. Run the tests:**

```bash
PYTHONPATH=. python tests/test_smoke.py     # or: python -m pytest -q
```

---

## Repository structure

```
triprune-hgnn/
├── triprune/
│   ├── data/
│   │   ├── hetero_hypergraph.py   # HeteroHypergraph container
│   │   └── synthetic.py           # generator + empirical compressibility
│   ├── models/
│   │   ├── hgnn.py                # base hypergraph message passing
│   │   ├── hcp.py                 # compressibility predictor
│   │   ├── pruning.py             # NAHP: gates, controllers, importance
│   │   ├── contrastive.py         # attention-based fn/hn mining
│   │   └── triprune.py            # full model
│   ├── training/
│   │   ├── meta.py                # simplex projection + FD hypergradient
│   │   └── curriculum.py          # Algorithm 1 trainer
│   └── utils/
│       ├── features.py            # 10-dim structural descriptor
│       └── metrics.py
├── scripts/    demo.py · pretrain_hcp.py · train.py
├── configs/    default.yaml
├── tests/      test_smoke.py
└── checkpoints/
```

---

## Using your own data

The model consumes a `HeteroHypergraph`:

```python
from triprune import HeteroHypergraph, train_triprune, TrainConfig
import torch

hg = HeteroHypergraph(
    x=torch.randn(n, d),                 # node features [n, d]
    incidence=[H_1, ..., H_K],           # one [n, m_k] incidence matrix per component
    y=labels,                            # [n] integer labels
    train_mask=train_mask,               # [n] bool
    val_mask=val_mask,
    test_mask=test_mask,
    num_classes=C,
)
res = train_triprune(hg, r_star=0.31, cfg=TrainConfig(epochs=200))
```

To wire in the paper's benchmarks (IMDB / DBLP / Yelp / Amazon / Douban, and the
out-of-domain NTU2012 / ModelNet40 / House), implement `load_real_dataset` in
`scripts/train.py` to return a `HeteroHypergraph`, then:

```bash
PYTHONPATH=. python -m scripts.train --dataset imdb --hcp checkpoints/hcp.pt
```

The heterogeneous-hypergraph construction (one hyperedge per node per behavioural
component) follows the formulation cited in the paper.

---

## Reproducing paper results

The exact tables in the paper depend on the real datasets, the released
pretrained HCP, and the seed protocol, none of which are bundled here (see
below). To reproduce them you need to: (1) provide the datasets via
`load_real_dataset`; (2) pre-train the HCP on a corpus of the size reported in
Section 3.1; (3) run the seeds and splits described in the experimental protocol.
The synthetic demo reproduces the *qualitative* behaviour — accuracy held while
retention converges to the predicted operating point — not the benchmark numbers.

---
