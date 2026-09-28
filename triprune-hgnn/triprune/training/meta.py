"""Meta-learned loss balancing (Section 3.4).

Loss weights ``lambda`` live on the probability simplex and are updated by
projected gradient descent on the one-step-unrolled validation loss.  The
hypergradient is the central finite-difference estimate of Eq. (fd_bias) --
O(|lambda|) inner evaluations, i.e. cheap because ``lambda`` is 4-dimensional.
Proposition 2 bounds this estimate's bias; Proposition 5 turns it into a local
approximate-stationarity statement for this very update.
"""
from __future__ import annotations

import copy

import numpy as np
import torch


def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the probability simplex (Duchi et al., 2008)."""
    v = v.detach().cpu().double()
    n = v.numel()
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, n + 1, dtype=torch.double)
    cond = u - cssv / ind > 0
    rho = int(cond.nonzero().max().item()) + 1
    theta = cssv[rho - 1] / rho
    w = torch.clamp(v - theta, min=0.0)
    return w.float()


@torch.no_grad()
def _val_loss_after_inner_step(model, hg, r_star, lambdas, eta_inner) -> float:
    """One inner SGD step on the training loss, then evaluate validation loss."""
    clone = copy.deepcopy(model)
    clone.train()
    params = [p for p in clone.parameters() if p.requires_grad]

    with torch.enable_grad():
        out = clone(hg, r_star, lambdas, beta_hcp=0.0, compute_aux=True)
        grads = torch.autograd.grad(out["losses"]["total"], params, allow_unused=True)
    for p, g in zip(params, grads):
        if g is not None:
            p.data.add_(g, alpha=-eta_inner)

    clone.eval()
    with torch.enable_grad():
        out = clone(hg, r_star, lambdas, beta_hcp=0.0, compute_aux=False)
        logits = out["logits"]
        val_loss = torch.nn.functional.cross_entropy(
            logits[hg.val_mask], hg.y[hg.val_mask]
        )
    return float(val_loss)


def finite_difference_meta_step(
    model, hg, r_star, lambdas: torch.Tensor,
    eta_inner: float = 1e-3, eta_meta: float = 1e-2, delta: float = 1e-5,
) -> torch.Tensor:
    """Return updated loss weights after one projected FD meta-update."""
    dim = lambdas.numel()
    grad = torch.zeros(dim)
    for i in range(dim):
        e = torch.zeros(dim); e[i] = delta
        lp = _val_loss_after_inner_step(model, hg, r_star, lambdas + e, eta_inner)
        lm = _val_loss_after_inner_step(model, hg, r_star, lambdas - e, eta_inner)
        grad[i] = (lp - lm) / (2.0 * delta)
    updated = lambdas - eta_meta * grad
    return project_to_simplex(updated)
