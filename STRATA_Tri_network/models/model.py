"""Triple-Branch model: contrastive retrieval, regression, and 3D interpolation."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .backbones import DenseNet121Backbone, ViTBBackbone
from .heads import RegressionHead, GeneEncoder, ProjectionHead


def bleep_cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    return loss


class TripleBranchModel(nn.Module):
    """Retrieval (DenseNet+contrastive), regression (ViT-B), and parameter-free 3D interpolation."""

    def __init__(
        self,
        n_target_genes: int = 300,
        projection_dim: int = 256,
        temperature: float = 0.07,
        top_k: int = 150,
        tau_3d: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        self.n_target_genes = n_target_genes
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.top_k = top_k
        self.tau_3d = tau_3d

        # Branch 1: contrastive retrieval (DenseNet-121).
        self.bleep_encoder = DenseNet121Backbone(pretrained=pretrained, trainable=True)
        self.bleep_projection = ProjectionHead(
            input_dim=self.bleep_encoder.feature_dim,
            projection_dim=projection_dim,
            dropout=0.1,
        )
        self.gene_encoder = GeneEncoder(
            input_dim=n_target_genes,
            projection_dim=projection_dim,
            hidden_dim=512,
            num_layers=2,
            dropout=0.1,
        )

        # Branch 2: regression (ViT-B).
        self.vit_encoder = ViTBBackbone(pretrained=pretrained, trainable=True)
        self.regression_head = RegressionHead(
            input_dim=self.vit_encoder.feature_dim,
            output_dim=n_target_genes,
            hidden_dims=[512, 256],
            dropout=0.1,
        )

        # Branch 3: parameter-free 3D interpolation (computed inline).

    def get_bleep_image_embed(self, images: torch.Tensor) -> torch.Tensor:
        features = self.bleep_encoder(images)
        embed = self.bleep_projection(features)
        return F.normalize(embed, dim=-1)

    def compute_contrastive_loss(self, image_embed, gene_embed):
        logits = (image_embed @ gene_embed.T) / self.temperature
        image_similarity = image_embed @ image_embed.T
        gene_similarity  = gene_embed  @ gene_embed.T
        targets = F.softmax(
            ((image_similarity + gene_similarity) / 2) / self.temperature, dim=-1
        )
        image_loss = bleep_cross_entropy(logits,   targets,   reduction='none')
        gene_loss  = bleep_cross_entropy(logits.T, targets.T, reduction='none')
        return ((image_loss + gene_loss) / 2.0).mean()

    def retrieve_topk(self, image_embed, gene_embeds_db, target_genes_db, top_k=None):
        """Retrieve top-k similar gene embeddings and mix their target genes."""
        if top_k is None:
            top_k = self.top_k
        top_k = min(top_k, gene_embeds_db.size(0))
        sim = image_embed @ gene_embeds_db.T
        top_sims, top_indices = sim.topk(top_k, dim=-1)
        top_genes = target_genes_db[top_indices]
        weights = F.softmax(top_sims / self.temperature, dim=-1)
        pred_ret = (weights.unsqueeze(-1) * top_genes).sum(dim=1)
        return pred_ret, top_sims

    def forward_reg(self, images):
        feat = self.vit_encoder(images)
        pred_reg = self.regression_head(feat)
        return feat, pred_reg

    @staticmethod
    def forward_3d_teacher(query_conch, neighbor_vis, neighbor_expr,
                            neighbor_mask, neighbor_weights, tau=0.1):
        """Parameter-free 3D interpolation: spatial weight times visual cosine similarity."""
        q = F.normalize(query_conch, dim=-1)
        n = F.normalize(neighbor_vis, dim=-1)
        cos_sim = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1)

        spatial_log = torch.log(neighbor_weights.clamp_min(1e-8))
        logits = spatial_log + cos_sim / tau
        logits = logits.masked_fill(~neighbor_mask, float('-inf'))

        attn = F.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        pred_3d = torch.bmm(attn.unsqueeze(1), neighbor_expr).squeeze(1)
        confidence = attn.max(dim=-1).values
        return pred_3d, confidence, attn

    def forward_joint(self, batch, compute_ret=True, compute_3d_teacher=True):
        out = {}

        imgs   = batch['image']
        target = batch['target_genes']

        if compute_ret:
            out['img_emb']  = self.get_bleep_image_embed(imgs)
            out['gene_emb'] = self.gene_encoder(target)

        feat, pred_reg = self.forward_reg(imgs)
        out['feat']     = feat
        out['pred_reg'] = pred_reg

        if compute_3d_teacher:
            with torch.no_grad():
                pred_3d, confidence, attn_3d = self.forward_3d_teacher(
                    batch['query_conch'],
                    batch['neighbor_vis'],
                    batch['neighbor_expr'],
                    batch['neighbor_mask'],
                    batch['neighbor_weights'],
                    tau=self.tau_3d,
                )
            out['pred_3d']    = pred_3d
            out['confidence'] = confidence
            out['attn_3d']    = attn_3d

        return out

    def compute_joint_loss(self, out, batch, epoch: int,
                           ret_epochs: int = 100,
                           lambda_ret: float = 0.1,
                           lambda_con: float = 0.1) -> Dict[str, torch.Tensor]:
        """Total = L_reg + lambda_ret*L_ret (warmup-only) + lambda_con*L_con."""
        target = batch['target_genes']
        losses = {
            'total': target.new_zeros(()),
            'ret':   target.new_zeros(()),
            'reg':   target.new_zeros(()),
            'con':   target.new_zeros(()),
        }

        L_reg = F.mse_loss(out['pred_reg'], target)
        losses['reg'] = L_reg
        losses['total'] = losses['total'] + L_reg

        if epoch < ret_epochs and 'img_emb' in out:
            L_ret = self.compute_contrastive_loss(out['img_emb'], out['gene_emb'])
            losses['ret'] = L_ret
            losses['total'] = losses['total'] + lambda_ret * L_ret

        if lambda_con > 0 and 'pred_3d' in out:
            nbr_mask = batch['neighbor_mask']
            has_nbr = nbr_mask.any(dim=-1)
            if has_nbr.any():
                conf = out['confidence'][has_nbr].detach()            # (M,)
                diff = out['pred_reg'][has_nbr] - out['pred_3d'][has_nbr].detach()
                L_con = (conf.unsqueeze(-1) * diff.pow(2)).mean()
                losses['con'] = L_con
                losses['total'] = losses['total'] + lambda_con * L_con

        return losses

    def get_ret_params(self):
        return (list(self.bleep_encoder.parameters()) +
                list(self.bleep_projection.parameters()) +
                list(self.gene_encoder.parameters()))

    def get_reg_params(self):
        return (list(self.vit_encoder.parameters()) +
                list(self.regression_head.parameters()))

    def print_parameter_summary(self, log_fn=print):
        def count(m): return sum(p.numel() for p in m.parameters())
        bleep = count(self.bleep_encoder) + count(self.bleep_projection) + count(self.gene_encoder)
        reg   = count(self.vit_encoder) + count(self.regression_head)
        log_fn(f"{'='*60}")
        log_fn("TRIPLE-BRANCH MODEL (joint)")
        log_fn(f"{'='*60}")
        log_fn(f"  Ret (DenseNet+Proj+GeneEnc):  {bleep:>12,}")
        log_fn(f"  Reg (ViT-B+RegHead):          {reg:>12,}")
        log_fn(f"  3D:                           {0:>12,}   (zero-parameter)")
        log_fn(f"  TOTAL:                        {bleep + reg:>12,}")
        log_fn(f"{'='*60}")


@torch.no_grad()
def build_gene_embed_database(model, target_genes_all, batch_size=256, device=None):
    """Precompute gene embeddings for retrieval database."""
    model.gene_encoder.eval()
    device = device or next(model.parameters()).device
    gene_embeds = []
    for i in range(0, len(target_genes_all), batch_size):
        batch = target_genes_all[i:i + batch_size].to(device)
        embed = model.gene_encoder(batch)
        gene_embeds.append(embed.cpu())
    return torch.cat(gene_embeds, dim=0)
