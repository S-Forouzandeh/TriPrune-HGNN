"""Neural Adaptive Hierarchical Pruning (NAHP), Section 3.2.

Three coupled levels -- components, hyperedges, nodes -- each driven by a small
learned threshold controller conditioned on a 7-dim graph-state vector
(Eq. graph_state) and the HCP prior ``r_star``.  Gating uses the piecewise-linear
hard sigmoid with straight-through gradients (Eq. unified_gate):

    g = clamp( (pi - theta) / eps + 0.5, 0, 1 )

which is differentiable almost everywhere with slope ``1/eps`` inside the active
band -- the estimator whose bias Proposition 1 bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_NORM = 1e-8


def soft_gate(pi: torch.Tensor, theta: torch.Tensor, eps: float) -> torch.Tensor:
    """Hard-sigmoid soft gate (Eq. unified_gate)."""
    return torch.clamp((pi - theta) / eps + 0.5, 0.0, 1.0)


def minmax(s: torch.Tensor) -> torch.Tensor:
    """Normalise importance scores to spread over [0, 1].

    A threshold in [0, 1] then selects a controllable *fraction* of elements --
    unlike softmax, which concentrates the mass and makes the gate collapse.
    Gradients still flow to the score parameters (differentiable a.e.).
    """
    lo = s.min()
    hi = s.max()
    return (s - lo) / (hi - lo + 1e-8)


class ThresholdController(nn.Module):
    """MLP mapping a graph-state vector to a threshold (Eq. neural_threshold)."""

    def __init__(self, state_dim: int = 7, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(state)).squeeze(-1)


@dataclass
class PruneOutput:
    incidence: List[torch.Tensor]          # soft-masked incidence per component
    node_gate: torch.Tensor                # [n]
    retentions: Dict[str, torch.Tensor]    # comp / edge / node / overall (scalars)
    thresholds: Dict[str, torch.Tensor]
    importance: Dict[str, torch.Tensor]


class NAHP(nn.Module):
    def __init__(
        self,
        num_components: int,
        eps: float = 0.01,
        hidden: int = 32,
        k_min_frac: float = 0.1,
        node_min_frac: float = 0.05,
    ):
        super().__init__()
        self.K = num_components
        self.eps = eps
        self.k_min = max(1, int(k_min_frac * num_components))
        self.node_min_frac = node_min_frac

        # per-component learnable ||W_k^{(L)}||-proxy
        self.comp_weight = nn.Parameter(torch.ones(num_components))
        # beta_k predictor (Eq. beta_learnable): inputs [|W_k|, S_att, var]
        self.balance = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )
        self.thr_comp = ThresholdController(7, hidden)
        self.thr_edge = ThresholdController(7, hidden)
        self.thr_node = ThresholdController(7, hidden)

    # -- importance scores -------------------------------------------------
    def _component_scores(self, x, incidence):
        xn = F.normalize(x, dim=1)
        w = torch.abs(self.comp_weight)
        s_att, act, var = [], [], []
        for H in incidence:
            deg = H.sum(0).clamp(min=1.0)                      # [m]
            member = (H.sum(1) > 0).float()                    # [n]
            # edge cohesion -> S_att(k) in [0,1]
            centroid = F.normalize((H.t() @ xn) / deg[:, None], dim=1)
            coh = ((H.t() * (xn @ centroid.t()).t()).sum(1) / deg).mean()
            s_att.append(((coh + 1.0) / 2.0))
            # activation strength / variance over member nodes
            cnt = member.sum().clamp(min=1.0)
            act.append((member * x.norm(dim=1)).sum() / cnt)
            var.append(((member[:, None] * x).var(dim=0).mean()))
        s_att = torch.stack(s_att); act = torch.stack(act); var = torch.stack(var)
        beta = torch.sigmoid(self.balance(
            torch.stack([w, s_att, var], dim=1)
        )).squeeze(-1)                                          # [K]
        act_n = act / (act.max() + EPS_NORM)
        score = beta * w * act_n + (1 - beta) * s_att
        pi_comp = minmax(score)
        return pi_comp, beta

    def _edge_scores(self, H, xn):
        deg = H.sum(0).clamp(min=1.0)
        centroid = F.normalize((H.t() @ xn) / deg[:, None], dim=1)
        coh = (H.t() * (xn @ centroid.t()).t()).sum(1) / deg
        return minmax(coh)                                     # [m]

    def _node_scores(self, x, incidence):
        xn = F.normalize(x, dim=1)
        # connectivity strength: mean cosine to co-members across components
        A = torch.zeros(x.shape[0], x.shape[0], device=x.device)
        for H in incidence:
            A = A + (H @ H.t())
        A.fill_diagonal_(0.0)
        deg = (A > 0).float().sum(1).clamp(min=1.0)
        sim = (A > 0).float() * (xn @ xn.t())
        bar_s = sim.sum(1) / deg
        score = bar_s * x.norm(dim=1)
        return minmax(score)                                   # [n]

    @staticmethod
    def _state(prev_ret, pi, progress, dloss, gradnorm, r_star, device):
        pi = pi.detach()
        return torch.tensor([
            float(prev_ret), float(pi.mean()), float(pi.std()),
            float(progress), float(dloss), float(gradnorm), float(r_star),
        ], dtype=torch.float32, device=device)

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        incidence: List[torch.Tensor],
        r_star: float,
        progress: float = 0.0,
        dloss: float = 0.0,
        gradnorm: float = 0.0,
        prev_ret=(1.0, 1.0, 1.0),
        hard: bool = False,
    ) -> PruneOutput:
        device = x.device
        xn = F.normalize(x, dim=1)

        # ---- component level ----
        pi_comp, beta = self._component_scores(x, incidence)
        s_c = self._state(prev_ret[0], pi_comp, progress, dloss, gradnorm, r_star, device)
        theta_c = self.thr_comp(s_c)
        g_comp = soft_gate(pi_comp, theta_c, self.eps)          # [K]
        # safety: keep at least k_min components (straight-through floor)
        if self.k_min > 0:
            topk = torch.topk(pi_comp, self.k_min).indices
            floor = torch.zeros_like(g_comp); floor[topk] = 1.0
            g_comp = torch.maximum(g_comp, floor)

        # ---- edge level (shared threshold, per-component gates) ----
        pi_edge_all = [self._edge_scores(H, xn) for H in incidence]
        pi_edge_cat = torch.cat(pi_edge_all)
        s_e = self._state(prev_ret[1], pi_edge_cat, progress, dloss, gradnorm, r_star, device)
        theta_e = self.thr_edge(s_e)

        # ---- node level ----
        pi_node = self._node_scores(x, incidence)
        s_n = self._state(prev_ret[2], pi_node, progress, dloss, gradnorm, r_star, device)
        theta_n = self.thr_node(s_n)
        g_node = soft_gate(pi_node, theta_n, self.eps)          # [n]
        # safety floor: keep node_min_frac of nodes
        n_keep = max(2, int(self.node_min_frac * x.shape[0]))
        topn = torch.topk(pi_node, n_keep).indices
        nfloor = torch.zeros_like(g_node); nfloor[topn] = 1.0
        g_node = torch.maximum(g_node, nfloor)

        # ---- assemble pruned incidence ----
        pruned = []
        edge_gates = []
        for k, H in enumerate(incidence):
            g_edge = soft_gate(pi_edge_all[k], theta_e, self.eps)   # [m_k]
            # keep >=1 edge per retained component
            if g_edge.numel() > 0:
                top1 = torch.argmax(pi_edge_all[k])
                ef = torch.zeros_like(g_edge); ef[top1] = 1.0
                g_edge = torch.maximum(g_edge, ef)
            edge_gates.append(g_edge)
            mask = g_comp[k] * g_edge[None, :] * g_node[:, None]     # [n, m_k]
            pruned.append(H * mask)

        if hard:
            pruned = [ (H > 0).float() * (self._binarise(g_comp[k]) *
                        self._binarise(edge_gates[k])[None, :] *
                        self._binarise(g_node)[:, None])
                       for k, H in enumerate(incidence) ]

        edge_ret = torch.cat(edge_gates).mean() if edge_gates else torch.tensor(1.0)
        ret = {
            "comp": g_comp.mean(),
            "edge": edge_ret,
            "node": g_node.mean(),
        }
        ret["overall"] = ret["comp"] * ret["edge"] * ret["node"] ** 2
        return PruneOutput(
            incidence=pruned,
            node_gate=g_node,
            retentions=ret,
            thresholds={"comp": theta_c, "edge": theta_e, "node": theta_n},
            importance={"comp": pi_comp, "node": pi_node, "beta": beta},
        )

    @staticmethod
    def _binarise(g: torch.Tensor) -> torch.Tensor:
        return (g > 0.5).float()
