"""Heterogeneous-hypergraph data container.

A heterogeneous hypergraph (Eq. 3 in the paper) is a set of shared *primary*
nodes together with one incidence matrix ``H_k`` per behavioural *component*
(context).  This module keeps the representation deliberately small and
framework-light: incidence matrices are dense ``torch.Tensor`` objects, which is
fine for the synthetic graphs used in the demo and for the small real
benchmarks.  For very large graphs, swap the dense tensors for
``torch.sparse_coo_tensor`` -- every consumer in this package only relies on
matrix multiplication, so the change is local.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class HeteroHypergraph:
    """A heterogeneous hypergraph with ``K`` components.

    Attributes
    ----------
    x:
        Node feature matrix ``[n, d]`` for the ``n`` primary nodes.
    incidence:
        List of ``K`` incidence matrices, each ``[n, m_k]`` with entries in
        ``[0, 1]`` (``H_k[i, e] = 1`` iff node ``i`` belongs to hyperedge ``e``
        of component ``k``).
    y:
        Optional integer label per node ``[n]`` for node classification.
    train_mask / val_mask / test_mask:
        Optional boolean masks ``[n]`` selecting the labelled splits.
    """

    x: torch.Tensor
    incidence: List[torch.Tensor]
    y: Optional[torch.Tensor] = None
    train_mask: Optional[torch.Tensor] = None
    val_mask: Optional[torch.Tensor] = None
    test_mask: Optional[torch.Tensor] = None
    num_classes: Optional[int] = None
    meta: dict = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def n(self) -> int:
        return self.x.shape[0]

    @property
    def d(self) -> int:
        return self.x.shape[1]

    @property
    def K(self) -> int:
        return len(self.incidence)

    @property
    def m(self) -> int:
        return int(sum(H.shape[1] for H in self.incidence))

    def to(self, device) -> "HeteroHypergraph":
        self.x = self.x.to(device)
        self.incidence = [H.to(device) for H in self.incidence]
        for name in ("y", "train_mask", "val_mask", "test_mask"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(device))
        return self
