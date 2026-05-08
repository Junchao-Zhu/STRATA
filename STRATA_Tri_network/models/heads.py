"""Model heads, contrastive projection, regression, 3D interpolation, fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning (BLEEP style)"""

    def __init__(self, input_dim: int, projection_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class GeneEncoder(nn.Module):
    """Encodes gene expression vectors for contrastive retrieval"""

    def __init__(self, input_dim: int, projection_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                  nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                           nn.GELU(), nn.Dropout(dropout)])
        self.encoder = nn.Sequential(*layers)
        self.projection = ProjectionHead(hidden_dim, projection_dim, dropout)

    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        x = self.encoder(gene_expression)
        x = self.projection(x)
        return F.normalize(x, dim=-1)


class RegressionHead(nn.Module):
    """MLP regression head: features -> gene prediction"""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: list = [512, 256], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.LayerNorm(hd),
                           nn.GELU(), nn.Dropout(dropout)])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class InterpHead(nn.Module):
    """3D cross-layer interpolation via visual-similarity-weighted averaging."""

    def __init__(self, query_dim: int = 768, neighbor_vis_dim: int = 512,
                 hidden_dim: int = 256, tau: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(neighbor_vis_dim, hidden_dim)
        self.tau = tau

    def forward(self, query_feat, neighbor_vis, neighbor_expr, neighbor_mask=None):
        q = F.normalize(self.query_proj(query_feat), dim=-1)
        k = F.normalize(self.neighbor_proj(neighbor_vis), dim=-1)
        sim = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / self.tau

        if neighbor_mask is not None:
            sim = sim.masked_fill(~neighbor_mask, float('-inf'))

        attn = F.softmax(sim, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        pred_3d = torch.bmm(attn.unsqueeze(1), neighbor_expr).squeeze(1)
        return pred_3d, attn


class FusionHead(nn.Module):
    """Confidence-gated fusion of retrieval, regression, and 3D branches."""

    def __init__(self):
        super().__init__()
        self.w3 = nn.Parameter(torch.tensor(0.0))
        self.wr = nn.Parameter(torch.tensor(0.0))

    def forward(self, confidence):
        max_a3 = torch.sigmoid(self.w3)
        a3 = confidence * max_a3
        ratio_ret = torch.sigmoid(self.wr)
        a1 = (1 - a3) * ratio_ret
        a2 = (1 - a3) * (1 - ratio_ret)
        return torch.stack([a1, a2, a3], dim=-1)

    def get_readable_params(self):
        max_a3 = torch.sigmoid(self.w3).item()
        ratio_ret = torch.sigmoid(self.wr).item()
        return {
            'max_alpha_3d': max_a3,
            'ratio_ret_in_rest': ratio_ret,
            'at_conf_1': {
                'alpha_ret': (1 - max_a3) * ratio_ret,
                'alpha_reg': (1 - max_a3) * (1 - ratio_ret),
                'alpha_3d': max_a3,
            },
            'at_conf_0': {
                'alpha_ret': ratio_ret,
                'alpha_reg': 1 - ratio_ret,
                'alpha_3d': 0.0,
            },
        }
