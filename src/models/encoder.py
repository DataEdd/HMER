# -*- coding: utf-8 -*-
"""
encoder.py

PAL-v2 encoder (faithful to Table 3).
Produces a 400-channel feature map with total stride 32.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121


# --------------------------------------------------------------------------- #
# Minimal MD-LSTM stub – replace with your real implementation if available
# --------------------------------------------------------------------------- #

class MDLSTM(nn.Module):
    """Multi-directional 2-D LSTM layer (4 directions)."""

    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       return torch.relu(self.conv(x))


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #

class DenseMD(nn.Module):
    """6-layer dense MD-LSTM stack with growth rate 8."""

    def __init__(self, in_channels: int, growth: int = 8, n_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList(
            MDLSTM(in_channels + i * growth, hidden=growth) for i in range(n_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = [x]
        for layer in self.layers:
            out = layer(torch.cat(feats, dim=1))
            feats.append(out)
        return torch.cat(feats, dim=1)  # concat over channel dim


class Encoder(nn.Module):
    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()

        # 1. Stem + DenseNet-121 backbone
        self.backbone = densenet121(weights="DEFAULT" if pretrained_backbone else None)
        self.backbone.features.conv0 = nn.Conv2d(  # 1-channel input
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Remove classifier head – we use only .features
        self.backbone.classifier = nn.Identity()

        # 2. DenseMD block (concat +48 channels)
        self.dmd = DenseMD(in_channels=1024, growth=8, n_layers=6)  # 1024+6*8 = 1072

        # 3. 1×1 bottleneck to 400 channels (matches Table 3)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1072, 400, kernel_size=1, bias=False),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
        )

        # 4. Absolute 2-D positional embedding
        self.pos_emb_y = nn.Embedding(32, 400)   # max H' (64/32 = 2) < 32
        self.pos_emb_x = nn.Embedding(256, 400)  # max W'/32 ≤ 256 for CROHME

        # 5. 3×3 fuse conv
        self.fuse = nn.Sequential(
            nn.Conv2d(400, 400, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
        )

    # --------------------------------------------------------------------- #
    # Positional-embedding helper
    # --------------------------------------------------------------------- #
    def _add_position(self, f: torch.Tensor) -> torch.Tensor:
        """
        Add learnable 2-D absolute positional encoding.
        Args:
            f : (B, 400, H, W)
        Returns:
            (B, 400, H, W)
        """
        B, C, H, W = f.shape
        device = f.device

        yy = torch.arange(H, device=device)          # (H,)
        xx = torch.arange(W, device=device)          # (W,)

        y_emb = self.pos_emb_y(yy)                   # (H, C)
        x_emb = self.pos_emb_x(xx)                   # (W, C)
        pos   = y_emb[:, None, :] + x_emb[None]      # (H, W, C)
        pos   = pos.permute(2, 0, 1)                 # (C, H, W)
        pos   = pos.unsqueeze(0).expand(B, -1, -1, -1)

        return f + pos

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1, 64, W)  –  pre-rasterised handwriting
        Returns:
            f : (B, 400, 2, W'//32)
        """
        f = self.backbone.features(x)         # (B, 1024, H/32, W/32)
        f = self.dmd(f)                       # (B, 1072, H/32, W/32)
        f = self.bottleneck(f)                # (B, 400,  H/32, W/32)
        f = self._add_position(f)             # add absolute pos
        f = self.fuse(f)                      # (B, 400, H/32, W/32)
        return f
