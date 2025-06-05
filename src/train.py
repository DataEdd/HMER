# -*- coding: utf-8 -*-
"""
train.py – PAL-v2 training loop **with discriminator** (GAN λ = 0.4)

Run:
  python -m src.train --lmdb data/crohme2013/processed/train.lmdb \
                      --epochs 5 --batch 24 --workers 4
"""
from __future__ import annotations
import argparse, math, time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocess.lmdb_dataset import CrohmeLMDB, pad_collate
from src.models.encoder        import Encoder
from src.models.decoder        import ConvDecoder
from src.models.discriminator  import Discriminator


# --------------------------------------------------------------------------- #
def ce_loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy ignoring <pad>=0."""
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        target.reshape(B * T),
        ignore_index=0,
    )


# --------------------------------------------------------------------------- #
def train_one_epoch(loader, model, opt_G, opt_D, device, lambda_gan=0.4):
    model.train()
    ce_tot, n_tok = 0.0, 0

    bce = nn.BCELoss()

    for batch in tqdm(loader, ncols=90):
        imgs = batch["image"].to(device)          # B×1×64×W
        tgt  = batch["y"].to(device)              # B×Tmax
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

        # ---------------------------------------------------- #
        #   Forward generator (enc + dec)
        # ---------------------------------------------------- #
        enc_f    = model["enc"](imgs)
        logits   = model["dec"](enc_f, tgt_in)            # B×T×V
        ctx_seq  = model["dec"].ctx_seq.detach()          # B×T×400
        emb_seq  = model["dec"].emb_seq.detach()          # B×T×256

        # main CE loss
        ce = ce_tot_batch = ce_loss_fn(logits, tgt_out)

        # ---------------------------------------------------- #
        #   Discriminator update
        # ---------------------------------------------------- #
        B, T, _ = ctx_seq.shape
        ctx_flat = ctx_seq.reshape(B * T, -1)
        emb_flat = emb_seq.reshape(B * T, -1)

        # real pairs: use the *ground-truth* embeddings (shifted target)
        gt_emb  = model["dec"].embed(tgt_out).detach()    # B×T×256
        real_ctx = ctx_seq                                # already detached
        real_flat = real_ctx.reshape(B * T, -1)
        real_emb  = gt_emb.reshape(B * T, -1)

        D = model["disc"]

        real_scores = D(real_flat, real_emb)   # label = 1
        fake_scores = D(ctx_flat, emb_flat)    # label = 0

        lbl_real = torch.ones_like(real_scores)
        lbl_fake = torch.zeros_like(fake_scores)

        loss_D = bce(real_scores, lbl_real) + bce(fake_scores, lbl_fake)

        # ---------------------------------------------------- #
        #   Combined generator loss  (CE  +  λ·GAN_G)
        #   GAN_G = encourage fake_scores -> 1
        # ---------------------------------------------------- #
        gan_g = bce(fake_scores, lbl_real)
        loss_G = ce + lambda_gan * gan_g

        # ----------- update generator (enc + dec) ------------
        opt_G.zero_grad()
        loss_G.backward(retain_graph=True)
        opt_G.step()

        # ----------- update discriminator --------------------
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # accounting
        ce_tot += ce_tot_batch.item() * (tgt_out != 0).sum().item()
        n_tok  += (tgt_out != 0).sum().item()

    return ce_tot / n_tok


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser("PAL-v2 trainer (+ discriminator)")
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch",  type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr",      type=float, default=2e-4)
    ap.add_argument("--model-out", type=str, default="palv2.pt")
    args = ap.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print("Device:", device)

    # ---------------- dataset ----------------
    ds = CrohmeLMDB(args.lmdb)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=pad_collate)

    # ---------------- model ------------------
    vocab_size = 1704           # 1700 BPE + 4 specials
    model = nn.ModuleDict({
        "enc":  Encoder(),
        "dec":  ConvDecoder(vocab_size),
        "disc": Discriminator(),
    }).to(device)

    opt_G = optim.AdamW(list(model["enc"].parameters()) +
                        list(model["dec"].parameters()), lr=args.lr)
    opt_D = optim.AdamW(model["disc"].parameters(), lr=args.lr)

    # ---------------- training ---------------
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        ppl = math.exp(train_one_epoch(dl, model, opt_G, opt_D, device))
        print(f"[{ep:02d}] ppl={ppl:.1f}  time={ (time.time()-t0)/60:.1f}m")

        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                    "epoch": ep},
                   args.model_out)

    print("✓ done")


if __name__ == "__main__":
    main()
