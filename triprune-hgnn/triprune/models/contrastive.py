"""Attention-based contrastive learning on the pruned graph (Section 3.3).

Three terms:

* ``L_cl``   -- component-weighted InfoNCE between a node and its assigned
               hyperedge embedding (Eq. L_cl_pr), down-weighted per component by
               a retention factor (Eq. pi_retain).
* ``L_fn``   -- false-negative correction: learned attention (Eq. fn_attention)
               identifies node pairs that were connected before pruning but
               disconnected after *and* remain similar, and pulls them together.
* ``L_hard`` -- hard-negative correction: learned attention (Eq. hn_attention)
               identifies dissimilar pairs whose embeddings drift together and
               pushes them apart.

This is a faithful but deliberately compact realisation: the contrastive and
mining terms are estimated over sampled nodes / pairs for tractability.  The
sampling sizes are configurable and do not change the objective in expectation.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _assigned_hyperedge_emb(H: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Per-node embedding of its assigned hyperedge in one component (Eq. hyperedge_emb)."""
    deg = H.sum(0).clamp(min=1.0)                     # [m]
    He = (H.t() @ Z) / deg[:, None]                   # [m, d]
    assigned = H.argmax(dim=1)                        # [n] (unique per component here)
    return He[assigned]                               # [n, d]


class ContrastiveMiner(nn.Module):
    def __init__(self, dim: int, num_components: int, hidden: int = 64,
                 tau: float = 0.5, alpha: float = 1.0):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.component_emb = nn.Parameter(torch.randn(num_components, dim) * 0.1)
        self.fn_attn = nn.Sequential(
            nn.Linear(3 * dim + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.hn_attn = nn.Sequential(
            nn.Linear(4 * dim, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )

    # -- component-weighted contrastive loss ------------------------------
    def contrastive_loss(self, Z, incidence, orig_incidence, retentions) -> torch.Tensor:
        n = Z.shape[0]
        idx = torch.randperm(n, device=Z.device)[: min(n, 256)]
        Zi = F.normalize(Z[idx], dim=1)
        loss = Z.new_zeros(())
        wsum = 0.0
        for k, H in enumerate(incidence):
            if H.shape[1] == 0:
                continue
            He = F.normalize(_assigned_hyperedge_emb(H, Z), dim=1)[idx]
            logits = (Zi @ He.t()) / self.tau            # [b, b]
            labels = torch.arange(idx.shape[0], device=Z.device)
            # per-component retention weight (Eq. pi_retain)
            o = float(orig_incidence[k].detach().sum())
            hh = float(H.detach().sum())
            change = abs(o - hh) / (o + 1e-8)
            pi_ret = float(retentions["edge"].detach()) * float(np.exp(-self.alpha * change))
            loss = loss + pi_ret * F.cross_entropy(logits, labels)
            wsum += pi_ret
        return loss / max(wsum, 1e-6)

    # -- false-negative correction ----------------------------------------
    def false_negative_loss(self, Z, incidence, orig_incidence, n_pairs: int = 256) -> torch.Tensor:
        device = Z.device
        Zn = F.normalize(Z, dim=1)
        losses = []
        for k, (H, H0) in enumerate(zip(incidence, orig_incidence)):
            A0 = (H0 @ H0.t() > 0).float()
            A = (H @ H.t() > 0).float()
            removed = ((A0 - A) > 0)                      # disconnected by pruning
            removed.fill_diagonal_(False)
            ii, jj = torch.where(removed)
            if ii.numel() == 0:
                continue
            sel = torch.randperm(ii.numel(), device=device)[:n_pairs]
            ii, jj = ii[sel], jj[sel]
            zi, zj = Z[ii], Z[jj]
            delta = torch.ones(ii.shape[0], 1, device=device)   # edge removed => 1
            ck = self.component_emb[k].expand(ii.shape[0], -1)
            alpha_fn = torch.sigmoid(self.fn_attn(torch.cat([zi, zj, delta, ck], dim=1))).squeeze(-1)
            sim = (Zn[ii] * Zn[jj]).sum(1)
            # pull selected pairs together, weighted by learned attention
            losses.append((alpha_fn * (1.0 - sim)).mean())
        if not losses:
            return Z.new_zeros(())
        return torch.stack(losses).mean()

    # -- hard-negative correction -----------------------------------------
    def hard_negative_loss(self, Z, incidence, x, n_pairs: int = 256) -> torch.Tensor:
        device = Z.device
        Zn = F.normalize(Z, dim=1)
        xn = F.normalize(x, dim=1)
        losses = []
        for k, H in enumerate(incidence):
            if H.shape[1] == 0:
                continue
            He = _assigned_hyperedge_emb(H, Z)
            A = (H @ H.t() > 0).float(); A.fill_diagonal_(0.0)
            n = Z.shape[0]
            a = torch.randint(0, n, (n_pairs,), device=device)
            b = torch.randint(0, n, (n_pairs,), device=device)
            feat_sim = (xn[a] * xn[b]).sum(1)
            shared = (A[a] * A[b]).sum(1)                 # shared retained neighbours
            cand = (feat_sim < 0.2) & (shared >= 1)
            if cand.sum() == 0:
                continue
            a, b = a[cand], b[cand]
            inp = torch.cat([Z[a], Z[b], He[a], He[b]], dim=1)
            alpha_hn = torch.sigmoid(self.hn_attn(inp)).squeeze(-1)
            sim = (Zn[a] * Zn[b]).sum(1)
            # push selected pairs apart
            losses.append((alpha_hn * torch.clamp(sim, min=0.0)).mean())
        if not losses:
            return Z.new_zeros(())
        return torch.stack(losses).mean()
