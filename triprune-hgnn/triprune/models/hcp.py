"""Hypergraph Compressibility Predictor (HCP), Section 3.1 / Eq. hcp_predictor.

A small MLP maps the 10-dimensional structural descriptor ``phi(H)`` to an
estimate ``r_star in (0, 1)`` of the achievable overall retention, *before*
training begins.  It is pre-trained once on a synthetic corpus and applied
**frozen** at deployment, which is what makes it transferable across domains.

Feature standardisation statistics are stored inside the module so a saved
checkpoint is fully self-contained.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..utils.features import FEATURE_DIM, structural_features


class HCP(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden=(64, 32)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )
        # standardisation buffers (filled by fit / load)
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        z = (phi - self.feat_mean) / (self.feat_std + 1e-8)
        return torch.sigmoid(self.net(z)).squeeze(-1)

    # -- convenience -------------------------------------------------------
    @torch.no_grad()
    def predict(self, incidence, k: int | None = None) -> float:
        phi = structural_features(incidence, k=k)
        t = torch.tensor(phi, dtype=torch.float32, device=self.feat_mean.device)
        return float(self.forward(t.unsqueeze(0)).item())

    def set_standardisation(self, feats: np.ndarray) -> None:
        self.feat_mean.copy_(torch.tensor(feats.mean(0), dtype=torch.float32))
        self.feat_std.copy_(torch.tensor(feats.std(0) + 1e-8, dtype=torch.float32))


def train_hcp(
    features: np.ndarray,
    targets: np.ndarray,
    hidden=(64, 32),
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[HCP, dict]:
    """Fit HCP to ``(phi, r_emp)`` pairs by minimising MSE (Eq. of Section 3.1).

    Returns the trained model and a dict with train RMSE.
    """
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    model = HCP(in_dim=features.shape[1], hidden=hidden).to(device)
    model.set_standardisation(features)

    X = torch.tensor(features, device=device)
    y = torch.tensor(targets, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % max(1, epochs // 10) == 0:
            print(f"[HCP] epoch {ep+1:4d}  MSE={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        rmse = float(torch.sqrt(torch.mean((model(X) - y) ** 2)))
    return model, {"train_rmse": rmse}
