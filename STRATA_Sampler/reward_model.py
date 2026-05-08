"""Learned dense reward model trained against the true validation-loss delta."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) -> (B,) predicted per-step reward."""
        return self.net(x).squeeze(-1)


class RewardModelTrainer:
    """Wraps a RewardModel with a buffer + training step.

    Each entry in the buffer is a *window*: the per-step features of k
    actions plus the single scalar R_true that resulted from those k picks.
    The model is trained so that the SUM of its k predicted dense rewards
    matches R_true (MSE).
    """

    def __init__(self, in_dim: int, hidden: int = 128, lr: float = 1e-3, device: str = "cuda"):
        self.model = RewardModel(in_dim, hidden).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.buffer: List[Tuple[torch.Tensor, float]] = []  # (window_feats (k,d), R_true)

    @torch.no_grad()
    def predict(self, feat: torch.Tensor) -> float:
        """Predict dense reward for a single (1,d) feature row."""
        feat = feat.to(self.device)
        return float(self.model(feat).item())

    def push_window(self, window_feats: torch.Tensor, R_true: float):
        """window_feats: (k, d) on CPU. R_true: scalar."""
        self.buffer.append((window_feats.detach().cpu(), float(R_true)))

    def update(self, batch_size: int = 32, n_steps: int = 4) -> float:
        if len(self.buffer) < 2:
            return 0.0
        losses: List[float] = []
        for _ in range(n_steps):
            idx = torch.randperm(len(self.buffer))[:batch_size].tolist()
            preds_sum = []
            tgts = []
            for i in idx:
                feats, R = self.buffer[i]
                feats = feats.to(self.device)
                preds_sum.append(self.model(feats).sum())
                tgts.append(torch.tensor(R, device=self.device, dtype=torch.float32))
            preds_t = torch.stack(preds_sum)
            tgts_t = torch.stack(tgts)
            loss = F.mse_loss(preds_t, tgts_t)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    @torch.no_grad()
    def calibration_pearson(self) -> float:
        """Pearson correlation between predicted-window-sum and true R on the buffer."""
        if len(self.buffer) < 5:
            return 0.0
        preds = []
        tgts = []
        for feats, R in self.buffer:
            feats = feats.to(self.device)
            preds.append(self.model(feats).sum().item())
            tgts.append(R)
        p = torch.tensor(preds)
        t = torch.tensor(tgts)
        if p.std() < 1e-9 or t.std() < 1e-9:
            return 0.0
        return float(((p - p.mean()) * (t - t.mean())).mean() / (p.std() * t.std() + 1e-9))
