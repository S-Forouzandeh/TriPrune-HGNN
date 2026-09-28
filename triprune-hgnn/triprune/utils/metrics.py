"""Small metric helpers used across training and evaluation."""
from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    pred = logits.argmax(dim=-1)
    if mask is not None:
        pred, y = pred[mask], y[mask]
    if y.numel() == 0:
        return 0.0
    return float((pred == y).float().mean())


@torch.no_grad()
def mae(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Mean absolute error between the predicted-class index and the label.

    The paper reports MAE on ordinal rating targets; for the synthetic
    classification demo we treat the label index as the ordinal value, which
    keeps the same reporting interface.
    """
    pred = logits.argmax(dim=-1).float()
    yy = y.float()
    if mask is not None:
        pred, yy = pred[mask], yy[mask]
    if yy.numel() == 0:
        return 0.0
    return float((pred - yy).abs().mean())
