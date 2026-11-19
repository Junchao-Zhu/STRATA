"""Triple-branch fusion variants V1, V3, V4."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionV1_2Param(nn.Module):
    """Two global scalars gating 3D-vs-rest and ret-vs-reg."""
    def __init__(self, **kwargs):
        super().__init__()
        self.w3 = nn.Parameter(torch.tensor(0.0))
        self.wr = nn.Parameter(torch.tensor(0.0))

    def forward(self, pred_ret, pred_reg, pred_3d, confidence):
        max_a3 = torch.sigmoid(self.w3)
        a3 = confidence * max_a3
        ratio_ret = torch.sigmoid(self.wr)
        a1 = (1 - a3) * ratio_ret
        a2 = (1 - a3) * (1 - ratio_ret)
        return (a1.unsqueeze(-1) * pred_ret +
                a2.unsqueeze(-1) * pred_reg +
                a3.unsqueeze(-1) * pred_3d)


class FusionV3_PerGeneGate(nn.Module):
    """Per-gene softmax weights with per-sample confidence gate on 3D."""
    def __init__(self, n_genes, **kwargs):
        super().__init__()
        self.n_genes = n_genes
        self.gene_alpha = nn.Parameter(torch.zeros(n_genes, 3))

    def forward(self, pred_ret, pred_reg, pred_3d, confidence):
        base = F.softmax(self.gene_alpha, dim=-1)
        a3 = base[:, 2].unsqueeze(0) * confidence.unsqueeze(-1)
        remain = 1.0 - a3
        sum_rr = base[:, 0] + base[:, 1] + 1e-8
        ratio_ret = (base[:, 0] / sum_rr).unsqueeze(0)
        a1 = remain * ratio_ret
        a2 = remain * (1.0 - ratio_ret)
        return a1 * pred_ret + a2 * pred_reg + a3 * pred_3d


class FusionV4_GateMLP(nn.Module):
    """Per-sample MLP that maps a quality vector to (a_ret, a_reg, a_3d)."""
    def __init__(self, n_genes=None, quality_dim=1, hidden=32, **kwargs):
        super().__init__()
        self.quality_dim = quality_dim
        self.net = nn.Sequential(
            nn.Linear(quality_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, pred_ret, pred_reg, pred_3d, confidence,
                quality_features=None):
        q = confidence.unsqueeze(-1) if quality_features is None else quality_features
        alphas = F.softmax(self.net(q), dim=-1)
        return (alphas[:, 0:1] * pred_ret +
                alphas[:, 1:2] * pred_reg +
                alphas[:, 2:3] * pred_3d)


def build_fusion_variant(name, n_genes, **kwargs):
    registry = {
        'V1_2param':        FusionV1_2Param,
        'V3_pergene_gate':  FusionV3_PerGeneGate,
        'V4_gate_mlp':      FusionV4_GateMLP,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown fusion variant: {name}. Choose from {list(registry.keys())}"
        )
    return registry[name](n_genes=n_genes, **kwargs)


FUSION_VARIANT_NAMES = ['V1_2param', 'V3_pergene_gate', 'V4_gate_mlp']
