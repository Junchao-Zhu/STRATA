"""Sampling policy: pointer decoder with trajectory memory."""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# 
class FrozenStaticProjector(nn.Module):
    def __init__(self, in_dim: int = 1542, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x))


# 
class PointerDecoderPolicy(nn.Module):
    def __init__(
        self,
        h_static_dim: int = 128,
        cell_prior_dim: int = 128,
        dyn_dim: int = 583,      # will be validated at runtime
        h_ctx_dim: int = 128,
        hidden: int = 128,
    ):
        super().__init__()
        in_dim = h_static_dim + cell_prior_dim + dyn_dim + h_ctx_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # value head (optional, unused - kept to avoid breaking old forward sig)
        self.value_head = nn.Linear(hidden, 1)
        self.h_static_dim = h_static_dim
        self.cell_prior_dim = cell_prior_dim
        self.dyn_dim = dyn_dim
        self.h_ctx_dim = h_ctx_dim
        self.hidden = hidden

    def score(
        self,
        h_static: torch.Tensor,    # (N, h_static_dim)
        cell_prior: torch.Tensor,  # (N, K)
        dyn: torch.Tensor,         # (N, dyn_dim)
        h_ctx: torch.Tensor,       # (h_ctx_dim,) single vector
    ) -> torch.Tensor:
        """Return logits (N,) per-spot. Caller is responsible for masking."""
        N = h_static.shape[0]
        ctx_b = h_ctx.unsqueeze(0).expand(N, -1)
        x = torch.cat([h_static, cell_prior, dyn, ctx_b], dim=1)
        logits = self.score_mlp(x).squeeze(-1)
        return logits

    def forward(self, *a, **kw):
        """Compatibility shim - prefer calling .score() directly."""
        return self.score(*a, **kw)


# 
def sample_action(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    deterministic: bool = False,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Mask + softmax + sample. Returns (action, log_prob, entropy)."""
    logits = logits.masked_fill(~candidate_mask, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if deterministic:
        a = int(torch.argmax(probs).item())
    else:
        a = int(torch.multinomial(probs, num_samples=1).item())
    log_probs_safe = torch.log(probs.clamp_min(1e-12))
    logp = log_probs_safe[a]
    ent = -(probs * log_probs_safe).sum()
    return a, logp, ent
