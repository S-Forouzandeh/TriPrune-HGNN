"""The full TriPrune-HGNN model (Section 3, Figure 1).

Couples the four learnable modules inside one forward pass:
HCP prior -> NAHP pruning -> message passing on the pruned graph ->
attention-based contrastive mining, with a meta-learned loss combination.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrastive import ContrastiveMiner
from .hgnn import HGNNEncoder
from .pruning import NAHP


class TriPruneHGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_components: int,
        hidden: int = 32,
        layers: int = 2,
        eps: float = 0.01,
        tau: float = 0.5,
        alpha: float = 1.0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = HGNNEncoder(in_dim, hidden, layers=layers, dropout=dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self.nahp = NAHP(num_components, eps=eps, hidden=hidden)
        self.miner = ContrastiveMiner(hidden, num_components, tau=tau, alpha=alpha)

    def forward(
        self,
        hg,
        r_star: float,
        lambdas: torch.Tensor,
        beta_hcp: float = 0.0,
        progress: float = 0.0,
        dloss: float = 0.0,
        gradnorm: float = 0.0,
        prev_ret=(1.0, 1.0, 1.0),
        hard: bool = False,
        compute_aux: bool = True,
    ) -> Dict:
        # ---- Stage 1-2: hierarchical pruning --------------------------------
        out = self.nahp(hg.x, hg.incidence, r_star, progress, dloss, gradnorm,
                        prev_ret=prev_ret, hard=hard)
        x_pruned = hg.x * out.node_gate[:, None]

        # ---- Stage 3: message passing on pruned graph ----------------------
        Z = self.encoder(x_pruned, out.incidence)
        logits = self.classifier(Z)

        result = {
            "logits": logits,
            "Z": Z,
            "retentions": out.retentions,
            "thresholds": out.thresholds,
        }

        # ---- Stage 4: losses ----------------------------------------------
        losses = {}
        if hg.y is not None and hg.train_mask is not None:
            losses["cls"] = F.cross_entropy(logits[hg.train_mask], hg.y[hg.train_mask])
        else:
            losses["cls"] = logits.new_zeros(())

        if compute_aux:
            losses["cl"] = self.miner.contrastive_loss(Z, out.incidence, hg.incidence, out.retentions)
            losses["fn"] = self.miner.false_negative_loss(Z, out.incidence, hg.incidence)
            losses["hard"] = self.miner.hard_negative_loss(Z, out.incidence, hg.x)
        else:
            for k in ("cl", "fn", "hard"):
                losses[k] = logits.new_zeros(())

        losses["hcp"] = (out.retentions["overall"] - r_star) ** 2

        total = (
            lambdas[0] * losses["cls"]
            + lambdas[1] * losses["cl"]
            + lambdas[2] * losses["fn"]
            + lambdas[3] * losses["hard"]
            + beta_hcp * losses["hcp"]
        )
        losses["total"] = total
        result["losses"] = losses
        return result
