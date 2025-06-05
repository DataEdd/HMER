# -*- coding: utf-8 -*-  # refreshed
"""make_dataset.py

Bulk-convert the CROHME-2013 InkML corpus into an LMDB where each entry is a
serialized dict with fields:

  xh     : handwritten raster (uint8 PNG bytes)
  xp     : printed template raster (uint8 PNG bytes)
  y      : list[int] (LaTeX token IDs; placeholder for now)
  latex  : raw LaTeX string (pre-tokenizer)
  uid    : unique sample ID

Usage:
  python -m src.preprocess.make_dataset \
      --split train \
      --inkml-root data/crohme2013/raw/TrainINKML \
      --lmdb-out data/crohme2013/processed/train.lmdb \
      --n-workers 8
"""

from __future__ import annotations

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lmdb
from tqdm import tqdm
import numpy as np

# Local imports
from src.preprocess.parse_inkml import parse_inkml
from src.preprocess.render import normalise_strokes, rasterise


###############################################################################
# Worker function: parse, rasterise, stub for template
###############################################################################

def _worker(inkml_path: str) -> dict | Exception:
    """Load one InkML, rasterise handwriting and printed template, return a dict."""
    try:
        sample = parse_inkml(inkml_path)
        uid = sample["uid"]

        # 1. Handwritten: normalise and rasterise
        strokes = sample["strokes"]
        strokes_norm, shape = normalise_strokes(strokes)
        xh = rasterise(strokes_norm, canvas_shape=shape)

        # 2. Printed template: placeholder blank image of same shape (to be replaced)
        xp = 255 * np.ones_like(xh, dtype=np.uint8)

        # 3. LaTeX: raw string (tokenisation later)
        latex = sample["latex"]

        data_dict = {
            "uid": uid,
            "latex": latex,
            "y": [],            # token IDs (to be filled after BPE)
            "xh": xh.tobytes(),
            "xp": xp.tobytes(),
            "shape": xh.shape,
        }
        return data_dict
    except Exception as e:
        return e


###############################################################################
# Main: walk paths, bulk write LMDB
###############################################################################

def main() -> None:
    ap = argparse.ArgumentParser(description="Build LMDB for CROHME2013")
    ap.add_argument("--split", required=True, choices=["train", "tiny"],
                    help="Split name (train or tiny for testing)")
    ap.add_argument("--inkml-root", required=True,
                    help="Path to CROHME InkML root folder")
    ap.add_argument("--lmdb-out", required=True,
                    help="Path to output LMDB file (will be created)")
    ap.add_argument("--n-workers", type=int, default=4,
                    help="Number of parallel workers")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional limit on number of samples (for tiny)")
    args = ap.parse_args()

    # Gather InkML paths
    inkml_paths = [str(p) for p in Path(args.inkml_root).rglob("*.inkml")]
    if args.limit:
        inkml_paths = inkml_paths[: args.limit]
    inkml_paths.sort()

    n_total = len(inkml_paths)
    print(f"Found {n_total:,} InkML files under {args.inkml_root}")

    # Create LMDB environment (make sure parent folder exists)
    os.makedirs(Path(args.lmdb_out).parent, exist_ok=True)
    env = lmdb.open(args.lmdb_out, map_size=1099511627776)

    # Write in parallel
    with env.begin(write=True) as txn:
        with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
            futures = {exe.submit(_worker, path): path for path in inkml_paths}

            for i, fut in enumerate(tqdm(futures, total=n_total)):
                res = fut.result()
                if isinstance(res, Exception):
                    # Print the path + error, then skip
                    print(f"Error at {futures[fut]} -> {res}")
                    continue

                # Use the loop index `i` as a stable integer key
                data_dict = res
                key = f"{i:08d}".encode("ascii")
                txn.put(key, pickle.dumps(data_dict))

    env.sync()
    env.close()
    print("Done building LMDB.")


if __name__ == "__main__":
    main()
