# -*- coding: utf-8 -*-
"""
lmdb_dataset.py

Data-loading utilities for CROHME LMDB.

* CrohmeLMDB – torch.utils.data.Dataset that streams handwriting images and
  metadata from an LMDB created by make_dataset.py.
* pad_collate – collate_fn that right-pads variable-width images so they can be
  stacked into a single tensor.

Example
-------
    from torch.utils.data import DataLoader
    from src.preprocess.lmdb_dataset import CrohmeLMDB, pad_collate

    ds = CrohmeLMDB("data/crohme2013/processed/train.lmdb")
    dl = DataLoader(ds, batch_size=4, collate_fn=pad_collate)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad as torch_pad
from torch.nn.utils.rnn import pad_sequence

__all__ = ["CrohmeLMDB", "pad_collate"]


class CrohmeLMDB(Dataset):
    """Memory‑efficient LMDB reader (handwritten expression samples)."""

    def __init__(self, lmdb_path: str | Path):
        self.env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        with self.env.begin() as txn:
            self.n_samples = txn.stat()["entries"]
        self._txn = None  # lazy per‑process txn

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.n_samples

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Dict[str, object]:
        if self._txn is None:
            self._txn = self.env.begin()

        key = f"{idx:08d}".encode("ascii")
        blob = self._txn.get(key)
        if blob is None:
            raise IndexError(key)

        sample = pickle.loads(blob)

        img = np.frombuffer(sample["xh"], dtype=np.uint8).reshape(sample["shape"])
        img = torch.from_numpy(img).float() / 255.0  # → (H,W)
        img = img.unsqueeze(0)                       # → (1,H,W)

        return {
            "image": img,                      # 1×H×W float32
            "y":     torch.tensor(sample["y"], dtype=torch.long),
            "latex": sample["latex"],
            "uid":   sample["uid"],
        }


# --------------------------------------------------------------------------- #
# Collate fn
# --------------------------------------------------------------------------- #

def pad_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Pad variable‑width images + variable‑length label sequences."""
    # 1. pad images on the right to the max width in batch
    max_w = max(item["image"].shape[-1] for item in batch)
    imgs = []
    for item in batch:
        img = item["image"]
        pad_w = max_w - img.shape[-1]
        if pad_w:
            img = torch_pad(img, (0, pad_w))  # (left,right)
        imgs.append(img)
    imgs = torch.stack(imgs)  # B×1×H×Wmax

    # 2. pad target sequences with 0 (<pad>)
    ys = [item["y"] for item in batch]
    y_pad = pad_sequence(ys, batch_first=True, padding_value=0)  # B×Tmax
    lengths = torch.tensor([t.numel() for t in ys], dtype=torch.long)

    return {
        "image": imgs,
        "y":     y_pad,        # padded LongTensor
        "len":   lengths,     # original lengths (for mask)
        "latex": [b["latex"] for b in batch],
        "uid":   [b["uid"]   for b in batch],
    }
