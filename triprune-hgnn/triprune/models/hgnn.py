"""Base hypergraph message passing (Eq. hgnn_prop).

    X^{l+1} = sigma( Dv^{-1/2} H De^{-1} H^T Dv^{-1/2} X Theta )

For a heterogeneous hypergraph the per-component operators are averaged before
the linear map, giving one shared node representation.  Pruning acts on the
incidence matrices passed in, so the same encoder runs unchanged on the pruned
graph -- exactly the property the density curriculum relies on.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def hypergraph_operator(H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Apply the symmetric normalised hypergraph operator of one component."""
    dv = H.sum(dim=1)                                   # node degree  [n]
    de = H.sum(dim=0)                                   # edge degree  [m]
    de_inv = 1.0 / (de + EPS)
    # vertex -> hyperedge -> vertex
    HX = H.t() @ X                                      # [m, d]
    HX = de_inv[:, None] * HX
    PX = H @ HX                                         # [n, d]
    dv_inv_sqrt = 1.0 / torch.sqrt(dv + EPS)
    PX = dv_inv_sqrt[:, None] * PX                      # left  Dv^{-1/2}
    # right normalisation folded via the incoming X being already scaled below
    return PX, dv_inv_sqrt


class HGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, incidence: List[torch.Tensor]) -> torch.Tensor:
        agg = torch.zeros_like(X)
        for H in incidence:
            dv = H.sum(dim=1)
            dv_inv_sqrt = 1.0 / torch.sqrt(dv + EPS)
            Xn = dv_inv_sqrt[:, None] * X               # right Dv^{-1/2}
            PX, left = hypergraph_operator(H, Xn)
            agg = agg + PX
        if len(incidence) > 0:
            agg = agg / len(incidence)
        return self.lin(agg)


class HGNNEncoder(nn.Module):
    """Stack of :class:`HGNNLayer` producing node embeddings."""

    def __init__(self, in_dim: int, hidden: int, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.layers = nn.ModuleList(
            HGNNLayer(dims[i], dims[i + 1]) for i in range(layers)
        )
        self.dropout = dropout

    def forward(self, X: torch.Tensor, incidence: List[torch.Tensor]) -> torch.Tensor:
        h = X
        for i, layer in enumerate(self.layers):
            h = layer(h, incidence)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
