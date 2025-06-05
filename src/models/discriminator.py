# -*- coding: utf-8 -*-
"""discriminator.py

Paired Adversarial Learning (PAL‑v2) **Discriminator**.

The discriminator judges whether a *(context‑vector, token‑embedding)* pair
came from the ground‑truth sequence ("real") or from the decoder’s predicted
sequence ("fake").  It is applied *token‑wise* after the decoder's PCA step.

Architecture (Table 3)
~~~~~~~~~~~~~~~~~~~~~~
* Input:  (B, C=400) context vector  ‖  (B, E=256) token embedding → concat
  → (B, 656)
* FC 512 → LeakyReLU(0.2) → FC 256 → LeakyReLU(0.2) → FC 1 → sigmoid

During training we minimise:
  * 𝓛ᴰ = − E_real [log D(x)] − E_fake [log(1−D(x))]
  * 𝓛ᴳ = λ * E_fake [log D(x)]   (added to CE loss)

Call‑sites
~~~~~~~~~~
>>> disc = Discriminator()
>>> real_score = disc(ctx_t, emb_t)           # ctx_t: B×400, emb_t: B×256
>>> fake_score = disc(ctx_hat, emb_hat)
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Discriminator", "gan_d_loss", "gan_g_loss"]


class Discriminator(nn.Module):
    """Token‑level discriminator (MLP)."""

    def __init__(self, ctx_dim: int = 400, emb_dim: int = 256):
        super().__init__()
        in_dim = ctx_dim + emb_dim  # 656
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, ctx: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Returns probability of *real* for each token (B, 1)."""
        x = torch.cat([ctx, emb], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Loss helpers (adversarial, binary cross‑entropy)
# ---------------------------------------------------------------------------
_bce = nn.BCELoss()


def gan_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator loss."""
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)
    return _bce(real_logits, real_labels) + _bce(fake_logits, fake_labels)


def gan_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator (decoder) adversarial loss."""
    real_labels = torch.ones_like(fake_logits)
    return _bce(fake_logits, real_labels)
