# -*- coding: utf-8 -*-
"""attention.py

Pre‑aware Coverage Attention (PCA) layer from the PAL‑v2 paper.
Implements equations (8‑12).  Input shapes follow the decoder/encoder design
in this repository.

Notation (per forward step)
--------------------------
* **enc_feat** : (B, C_e=400, H, W)   – encoder feature map `F`.
* **h_t**      : (B, D=256)            – decoder hidden state at step *t*.
* **cov**      : (B, 1, H, W)          – accumulated attention up to *t-1*.

Returns
~~~~~~~
* *ctx_t*   : (B, C_e)         – context vector (eq. 12).
* *alpha_t* : (B, H, W)        – normalised attention weights.
* *cov_new* : (B, 1, H, W)     – updated coverage (`cov + alpha`).

The module is **state‑free**; the caller manages the running coverage tensor.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCALayer(nn.Module):
    """Pre‑aware Coverage Attention (per‑step)."""

    def __init__(
        self,
        enc_channels: int = 400,
        dec_hidden: int = 256,
        attn_dim: int = 256,
    ) -> None:
        super().__init__()

        # Linear proj of decoder hidden
        self.proj_h = nn.Linear(dec_hidden, attn_dim, bias=False)

        # 1×1 conv proj of encoder feature map
        self.proj_enc = nn.Conv2d(enc_channels, attn_dim, kernel_size=1, bias=False)

        # Coverage feature (3×3 conv on running alpha sum)
        self.proj_cov = nn.Conv2d(1, attn_dim, kernel_size=3, padding=1, bias=False)

        # Energy vector (v^T tanh(.)) implemented as 1×1 conv after tanh
        self.v = nn.Conv2d(attn_dim, 1, kernel_size=1, bias=True)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        h_t: torch.Tensor,              # (B, dec_hidden)
        enc_feat: torch.Tensor,         # (B, C_e, H, W)
        cov: torch.Tensor,              # (B, 1,   H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C_e, H, W = enc_feat.shape

        # 1. Project terms into attn space
        proj_h = self.proj_h(h_t)                      # (B, attn_dim)
        proj_h = proj_h.unsqueeze(-1).unsqueeze(-1)    # (B, attn_dim, 1, 1)

        proj_enc = self.proj_enc(enc_feat)             # (B, attn_dim, H, W)
        proj_cov = self.proj_cov(cov)                  # (B, attn_dim, H, W)

        # 2. Energy
        e = self.v(torch.tanh(proj_h + proj_enc + proj_cov))  # (B,1,H,W)
        e = e.view(B, -1)                                      # (B, H*W)

        # 3. Normalised attention α (eq. 11)
        alpha = torch.softmax(e, dim=-1).view(B, 1, H, W)      # (B,1,H,W)

        # 4. Context vector (weighted sum, eq. 12)
        ctx = (alpha * enc_feat).view(B, C_e, -1).sum(dim=-1)  # (B, C_e)

        # 5. Update coverage
        cov_new = cov + alpha

        return ctx, alpha.squeeze(1), cov_new


# --------------------------------------------------------------------------- #
# Minimal unit‑test when running this file directly
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    layer = PCALayer()
    enc = torch.randn(2, 400, 2, 8)
    h   = torch.randn(2, 256)
    cov = torch.zeros(2, 1, 2, 8)
    ctx, alpha, cov_new = layer(h, enc, cov)

    print("ctx shape  :", ctx.shape)       # (2,400)
    print("alpha shape:", alpha.shape)     # (2,2,8)
    # softmax check
    print("alpha sum  :", alpha[0].sum())  # ≈1.0
