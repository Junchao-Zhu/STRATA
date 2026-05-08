"""Vision encoder backbones (DenseNet-121 and ViT-B)."""
import os
import torch
import torch.nn as nn
import timm
from typing import Dict


class DenseNet121Backbone(nn.Module):
    """DenseNet-121 backbone, output (B, 1024)."""

    def __init__(self, pretrained: bool = True, trainable: bool = True):
        super().__init__()
        densenet = timm.create_model('densenet121', pretrained=pretrained)
        self.features = densenet.features
        self.global_pool = densenet.global_pool
        self.feature_dim = 1024

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return x  # (B, 1024)


class ViTBBackbone(nn.Module):
    """ViT-B/16 with CONCH pretrained weights, output (B, 768) CLS token."""
    CONCH_PATH = os.environ.get(
        "CONCH_CKPT", "./checkpoints/conch_vit_b_16.bin"
    )

    def __init__(self, pretrained: bool = True, trainable: bool = True):
        super().__init__()
        self.feature_dim = 768

        if pretrained:
            from conch.open_clip_custom import create_model_from_pretrained
            model = create_model_from_pretrained(
                'conch_ViT-B-16',
                checkpoint_path=self.CONCH_PATH,
                force_image_size=224,
                return_transform=False
            )
            trunk = model.visual.trunk
        else:
            trunk = timm.create_model(
                'vit_base_patch16_224', pretrained=False, num_classes=0
            )

        self.patch_embed = trunk.patch_embed
        self.cls_token = trunk.cls_token
        self.pos_embed = trunk.pos_embed
        self.pos_drop = trunk.pos_drop
        self.blocks = trunk.blocks
        self.norm = trunk.norm

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        if x.dim() == 4:
            x = x.reshape(B, -1, x.shape[-1])
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # (B, 768) CLS token
