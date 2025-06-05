# -*- coding: utf-8 -*-
"""decoder.py – Conv decoder + PCA, now **stores ctx_seq & emb_seq** for GAN"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn

from .attention import PCALayer

__all__ = ["ConvDecoder"]


class CausalGLU(nn.Module):
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.pad = k - 1
        self.conv = wn(nn.Conv1d(channels, channels * 2, k, padding=self.pad))

    def forward(self, x):
        y = self.conv(x)[:, :, :-self.pad]  # remove future time
        a, b = y.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class ConvDecoder(nn.Module):
    def __init__(self, vocab: int, hidden: int = 256, C_e: int = 400):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.conv = nn.Sequential(CausalGLU(hidden),
                                  CausalGLU(hidden),
                                  CausalGLU(hidden))
        self.attn = PCALayer(enc_channels=C_e, dec_hidden=hidden)
        self.fc   = nn.Linear(hidden + C_e, vocab)

        # buffers to expose per-step feats for discriminator
        self.ctx_seq: torch.Tensor | None = None
        self.emb_seq: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def forward(self, enc_feat: torch.Tensor, tgt_in: torch.Tensor):
        """Teacher-forcing forward → logits (B,T,V)  + stores ctx/emb seq."""
        B, C_e, H, W = enc_feat.shape
        T = tgt_in.size(1)

        cov = torch.zeros(B, 1, H, W, device=enc_feat.device)

        emb  = self.embed(tgt_in)                 # B×T×Hid
        hidd = self.conv(emb.transpose(1, 2)).transpose(1, 2)  # B×T×Hid

        logits, ctx_li, emb_li = [], [], []
        for t in range(T):
            h_t  = hidd[:, t, :]                      # B×Hid
            ctx, _, cov = self.attn(h_t, enc_feat, cov)
            fusion = torch.cat([h_t, ctx], 1)
            logits.append(self.fc(fusion))
            ctx_li.append(ctx)
            emb_li.append(self.embed(tgt_in[:, t]))

        self.ctx_seq = torch.stack(ctx_li, 1)         # B×T×400
        self.emb_seq = torch.stack(emb_li, 1)         # B×T×256
        return torch.stack(logits, 1)                 # B×T×V
