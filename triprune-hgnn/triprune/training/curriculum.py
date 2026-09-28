"""Density-curriculum trainer -- Algorithm 1 of the paper.

Runs the coupled training loop: per epoch it predicts thresholds from the graph
state, prunes hierarchically, message-passes on the pruned graph, computes the
four losses combined by the current meta-learned weights plus the decaying HCP
regulariser, and takes an optimiser step.  Every ``meta_period`` epochs the loss
weights are updated by the finite-difference meta-step.  Retention ratios drift
smoothly and stabilise, so the model trains at its deployed sparsity for the tail
of training -- the density-curriculum property (Section 3.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..models.triprune import TriPruneHGNN
from ..utils.metrics import accuracy, mae
from .meta import finite_difference_meta_step


@dataclass
class TrainConfig:
    hidden: int = 32
    layers: int = 2
    eps: float = 0.01
    tau0: float = 0.5
    alpha: float = 1.0
    dropout: float = 0.3
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 5e-4
    eta_inner: float = 1e-3
    eta_meta: float = 1e-2
    delta: float = 1e-5
    meta_period: int = 5             # Delta_meta
    beta_hcp0: float = 0.1
    hcp_decay_frac: float = 0.5      # decay beta_hcp to 0 over first 50% of epochs
    grad_clip: float = 10.0
    eval_every: int = 20
    seed: int = 0
    device: str = "cpu"
    use_meta: bool = True
    verbose: bool = True


@dataclass
class TrainResult:
    model: TriPruneHGNN
    history: List[Dict] = field(default_factory=list)
    best_val_acc: float = 0.0
    final: Dict = field(default_factory=dict)


def train_triprune(hg, r_star: float, cfg: TrainConfig = TrainConfig()) -> TrainResult:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    hg = hg.to(device)

    model = TriPruneHGNN(
        in_dim=hg.d, num_classes=hg.num_classes, num_components=hg.K,
        hidden=cfg.hidden, layers=cfg.layers, eps=cfg.eps,
        tau=cfg.tau0, alpha=cfg.alpha, dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    lambdas = torch.full((4,), 0.25)
    prev_ret = (1.0, 1.0, 1.0)
    prev_total = None
    gradnorm = 0.0
    result = TrainResult(model=model)

    for t in range(cfg.epochs):
        model.train()
        progress = t / max(1, cfg.epochs)
        beta_hcp = cfg.beta_hcp0 * max(0.0, 1.0 - progress / cfg.hcp_decay_frac)
        dloss = 0.0 if prev_total is None else float(prev_total)

        opt.zero_grad()
        out = model(hg, r_star, lambdas, beta_hcp=beta_hcp, progress=progress,
                    dloss=dloss, gradnorm=gradnorm, prev_ret=prev_ret, hard=False)
        total = out["losses"]["total"]
        total.backward()
        gradnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        opt.step()

        # update running state
        r = out["retentions"]
        prev_ret = (float(r["comp"].detach()), float(r["edge"].detach()), float(r["node"].detach()))
        cur_total = float(total.detach())
        prev_total = 0.0 if prev_total is None else (cur_total - prev_total)
        prev_total = cur_total

        # meta-update of loss weights every meta_period epochs
        if cfg.use_meta and (t + 1) % cfg.meta_period == 0:
            lambdas = finite_difference_meta_step(
                model, hg, r_star, lambdas,
                eta_inner=cfg.eta_inner, eta_meta=cfg.eta_meta, delta=cfg.delta,
            )

        if cfg.verbose and (t + 1) % cfg.eval_every == 0:
            metrics = evaluate(model, hg, r_star, lambdas)
            metrics.update({"epoch": t + 1, "beta_hcp": round(beta_hcp, 4),
                            "overall_ret": round(float(r["overall"].detach()), 4),
                            "lambda": [round(float(x), 3) for x in lambdas]})
            result.history.append(metrics)
            result.best_val_acc = max(result.best_val_acc, metrics["val_acc"])
            print(f"[{t+1:3d}/{cfg.epochs}] loss={cur_total:.4f} "
                  f"val_acc={metrics['val_acc']:.4f} val_mae={metrics['val_mae']:.4f} "
                  f"ret={float(r['overall'].detach()):.3f} lambda={metrics['lambda']}")

    result.final = evaluate(model, hg, r_star, lambdas, split="test")
    result.final["retention"] = float(r["overall"].detach())
    return result


@torch.no_grad()
def evaluate(model, hg, r_star, lambdas, split: str = "val") -> Dict:
    """Evaluate at the *deployed* (hard-pruned) sparsity."""
    model.eval()
    out = model(hg, r_star, lambdas, beta_hcp=0.0, hard=True, compute_aux=False)
    logits = out["logits"]
    mask = hg.val_mask if split == "val" else hg.test_mask
    return {
        f"{split}_acc": accuracy(logits, hg.y, mask),
        f"{split}_mae": mae(logits, hg.y, mask),
        "val_acc": accuracy(logits, hg.y, hg.val_mask),
        "val_mae": mae(logits, hg.y, hg.val_mask),
    }
