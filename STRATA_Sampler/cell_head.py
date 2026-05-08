"""Online-co-trained vision-to-cell-cluster head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def nearest_centroid(X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Return the index of the nearest centroid (L2) for each row of X.

    X: (m, d), centroids: (K, d).  Returns (m,) int64.
    """
    # ||x - c||^2 = ||x||^2 - 2 x*c + ||c||^2
    x2 = (X * X).sum(1, keepdim=True)            # (m, 1)
    c2 = (centroids * centroids).sum(1)           # (K,)
    xc = X @ centroids.t()                         # (m, K)
    d2 = x2 - 2 * xc + c2.unsqueeze(0)             # (m, K)
    return d2.argmin(dim=1)


class CellHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden: int = 256, n_clusters: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_clusters),
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        # reservoir: all (conch, label) pairs seen so far
        self._X_buf: list = []
        self._y_buf: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (B, K)."""
        return self.net(x)

    @torch.no_grad()
    def predict_all(self, pool_conch: torch.Tensor, batch_size: int = 2048) -> np.ndarray:
        """Forward the whole pool CONCH tensor, returns (N, K) float32 numpy softmax."""
        self.eval()
        out = torch.zeros(pool_conch.shape[0], self.n_clusters,
                          dtype=torch.float32, device=pool_conch.device)
        for i in range(0, pool_conch.shape[0], batch_size):
            chunk = pool_conch[i : i + batch_size]
            out[i : i + batch_size] = F.softmax(self.net(chunk), dim=-1)
        self.train()
        return out.cpu().numpy()

    def fit_incremental(
        self,
        X_new: torch.Tensor,
        y_new: torch.Tensor,
        steps: int = 5,
        bs: int = 128,
        lr: float | None = None,
    ) -> dict:
        """Add (X_new, y_new) to the internal reservoir then train for `steps`
        SGD steps on random mini-batches over the *entire accumulated* buffer.

        Returns {'loss': last_loss, 'n_buffer': N_total, 'step_losses': [...]}.
        """
        self._X_buf.append(X_new.detach())
        self._y_buf.append(y_new.detach())
        X_all = torch.cat(self._X_buf, dim=0)
        y_all = torch.cat(self._y_buf, dim=0)
        n = X_all.shape[0]
        if n == 0:
            return {"loss": 0.0, "n_buffer": 0, "step_losses": []}

        if lr is not None:
            for pg in self.opt.param_groups:
                pg["lr"] = lr

        self.train()
        losses = []
        for _ in range(steps):
            if n <= bs:
                idx = torch.randperm(n, device=X_all.device)
            else:
                idx = torch.randint(0, n, (bs,), device=X_all.device)
            xb = X_all[idx]
            yb = y_all[idx]
            logits = self.net(xb)
            loss = F.cross_entropy(logits, yb)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(loss.item()))
        return {"loss": float(np.mean(losses)), "n_buffer": int(n), "step_losses": losses}
