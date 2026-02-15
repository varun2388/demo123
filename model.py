"""
Vision Transformer (ViT) model implementation from scratch using PyTorch.

Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
(Dosovitskiy et al., 2020).
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Split an image into fixed-size patches and project them into embeddings."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, embed_dim=256, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """A single transformer encoder block: LayerNorm -> MHSA -> LayerNorm -> MLP."""

    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image classification.

    Args:
        img_size: Input image size (assumes square images).
        patch_size: Size of each patch.
        in_channels: Number of input channels (3 for RGB).
        num_classes: Number of output classes.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embed dim.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer encoder
        x = self.blocks(x)
        x = self.norm(x)

        # Classification head uses [CLS] token output
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits


def build_vit(preset="default", num_classes=10):
    """Build a ViT model with preset configurations.

    Presets:
        default: ViT tuned for CIFAR-10 (32x32 images)
        tiny:    Smaller model for quick experiments
        small:   Larger model for better accuracy
    """
    configs = {
        "tiny": dict(
            img_size=32, patch_size=4, embed_dim=128, depth=4, num_heads=4,
            mlp_ratio=2.0, dropout=0.1,
        ),
        "default": dict(
            img_size=32, patch_size=4, embed_dim=256, depth=6, num_heads=8,
            mlp_ratio=4.0, dropout=0.1,
        ),
        "small": dict(
            img_size=32, patch_size=4, embed_dim=384, depth=8, num_heads=8,
            mlp_ratio=4.0, dropout=0.1,
        ),
    }
    cfg = configs[preset]
    return VisionTransformer(num_classes=num_classes, **cfg)
