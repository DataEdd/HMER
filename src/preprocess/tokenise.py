from __future__ import annotations

#  -*- coding: utf-8 -*-
"""tokenise.py – utility wrappers around SentencePiece.

Exports
~~~~~~~
* **load_sp(model_path)** – singleton loader that returns a
  `sentencepiece.SentencePieceProcessor`.
* **sp_encode(text, sp=None)** – returns `[bos] + ids + [eos]` as `List[int]`.
* **sp_decode(ids, sp=None)** – inverse.
* CLI with sub‑commands: `train`, `encode`, `decode`.
"""

import argparse
import tempfile
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm

__all__ = ["load_sp", "sp_encode", "sp_decode"]

_SP: Optional[spm.SentencePieceProcessor] = None  # cached singleton

# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def load_sp(model_path: str | Path = "data/crohme2013/processed/bpe.model") -> spm.SentencePieceProcessor:
    """Load (or return cached) SentencePieceProcessor."""
    global _SP
    if _SP is None:
        _SP = spm.SentencePieceProcessor()
        _SP.load(str(model_path))
    return _SP


def sp_encode(text: str, sp: Optional[spm.SentencePieceProcessor] = None) -> List[int]:
    sp = sp or load_sp()
    ids = sp.encode(text, out_type=int)
    return [1] + ids + [2]              # add <bos>=1, <eos>=2


def sp_decode(ids: List[int], sp: Optional[spm.SentencePieceProcessor] = None) -> str:
    sp = sp or load_sp()
    # remove special tokens
    ids = [i for i in ids if i > 3]
    return sp.decode(ids)

# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_bpe(lmdb_path: str | Path, out_prefix: str | Path, vocab_size: int = 1700):
    """Extract LaTeX from LMDB and train a SentencePiece BPE model."""
    import lmdb, pickle

    lmdb_path = Path(lmdb_path)
    out_prefix = Path(out_prefix)

    # 1. Dump LaTeX strings into a tmp text file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        with env.begin() as txn:
            cur = txn.cursor()
            for _, v in cur:
                latex = pickle.loads(v)["latex"].strip()
                tmp.write(latex + "\n")
        env.close()

    # 2. Train
    spm.SentencePieceTrainer.train(
        input=str(tmp_path),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        bos_id=1, eos_id=2, pad_id=0, unk_id=3,
        user_defined_symbols="",
    )
    tmp_path.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_train(args):
    train_bpe(args.lmdb, args.out, args.vocab)


def _cli_encode(args):
    print(sp_encode(args.text))


def _cli_decode(args):
    print(sp_decode([int(i) for i in args.ids]))


def _main():
    ap = argparse.ArgumentParser(description="SentencePiece helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train sub‑cmd
    ap_tr = sub.add_parser("train", help="train BPE model")
    ap_tr.add_argument("--lmdb", required=True)
    ap_tr.add_argument("--out", required=True)
    ap_tr.add_argument("--vocab", type=int, default=1700)
    ap_tr.set_defaults(func=_cli_train)

    # encode
    ap_en = sub.add_parser("encode", help="encode string")
    ap_en.add_argument("text")
    ap_en.set_defaults(func=_cli_encode)

    # decode
    ap_de = sub.add_parser("decode", help="decode ids")
    ap_de.add_argument("ids", nargs="+")
    ap_de.set_defaults(func=_cli_decode)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    _main()
"""lmdb_dataset.py – v2

* Fixes NameError inside `pad_collate` and collects target sequences
  correctly.
* Adds helper `pad_sequence` usage so caller gets a single `LongTensor`
  (`y_pad`) plus `lengths` for masking.
"""

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
