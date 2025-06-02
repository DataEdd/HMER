# -*- coding: utf-8 -*-
"""render.py

Rasterise vector *strokes* (as produced by ``parse_inkml``) into a binary
image ready for the CNN encoder.  The normalisation follows the protocol used
in the PAL‑v2 paper:

1. **Scale** so ink height equals ``target_height`` pixels.
2. **Translate** so the centre of mass is the image centre.
3. **Optional padding** – add equal left/right borders until the width reaches
   the next power‑of‑two (helps with conv‑stride divisibility).

CLI usage
~~~~~~~~~
::

    # Preview on‑screen only
    python -m src.preprocess.render                         \
        data/.../2_em_3.inkml --show

    # Save PNG inside processed/previews/<uid>.png (auto‑path)
    python -m src.preprocess.render                         \
        data/.../2_em_3.inkml --save

    # Save to a custom file (still auto‑creates folders)
    python -m src.preprocess.render                         \
        data/.../2_em_3.inkml --save /tmp/example.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw

###############################################################################
# Normalisation helpers
###############################################################################

def normalise_strokes(strokes: List[np.ndarray], *, target_height: int = 64,
                       pad_to_pow2: bool = True) -> List[np.ndarray]:
    """Scale + translate strokes; optionally pad width to the next power‑of‑two."""
    # Concatenate all points to compute global bbox + centre
    pts = np.vstack(strokes)  # shape [N, 2]
    min_xy = pts.min(0)
    max_xy = pts.max(0)
    height = max_xy[1] - min_xy[1]
    if height == 0:
        height = 1  # avoid div/0 for degenerate single‑point ink

    scale = target_height / height
    pts_scaled = (pts - min_xy) * scale

    # Compute centre of mass → used for centring
    com = pts_scaled.mean(0)
    strokes_scaled = [((s - min_xy) * scale) - com for s in strokes]

    # After translation, compute final bbox to know required canvas size
    pts2 = np.vstack(strokes_scaled)
    min2, max2 = pts2.min(0), pts2.max(0)
    width = int(math.ceil(max2[0] - min2[0]))
    height = int(math.ceil(max2[1] - min2[1]))  # should be ~= target_height

    if pad_to_pow2:
        pow2 = 1 << (width - 1).bit_length()  # next power of two >= width
        pad_left = (pow2 - width) // 2
        for i, s in enumerate(strokes_scaled):
            strokes_scaled[i] = s + np.array([pad_left - min2[0], -min2[1]])
        canvas_w = pow2
    else:
        for i, s in enumerate(strokes_scaled):
            strokes_scaled[i] = s - min2  # shift into positive coords
        canvas_w = width

    return strokes_scaled, (target_height, int(canvas_w))

###############################################################################
# Rasterisation
###############################################################################

def rasterise(strokes_norm: List[np.ndarray], canvas_shape: tuple[int, int] | None = None,
              line_width: int = 2) -> np.ndarray:
    """Return a *uint8* image with values 0/255."""
    if canvas_shape is None:
        pts = np.vstack(strokes_norm)
        h = int(np.ceil(pts[:, 1].max())) + line_width * 2
        w = int(np.ceil(pts[:, 0].max())) + line_width * 2
        canvas_shape = (h, w)

    img = Image.new("L", (canvas_shape[1], canvas_shape[0]), 0)  # black background
    draw = ImageDraw.Draw(img)
    for s in strokes_norm:
        xy = list(map(tuple, s))
        draw.line(xy, fill=255, width=line_width, joint="curve")
    return np.array(img, dtype=np.uint8)

###############################################################################
# CLI helper
###############################################################################

def _cli() -> None:
    from .parse_inkml import parse_inkml

    ap = argparse.ArgumentParser(description="Render a CROHME InkML to PNG")
    ap.add_argument("inkml_file", type=str)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", nargs="?", const="__AUTO__", default=None,
                    help="Save PNG; if no path is given the file is placed under "
                         "data/crohme2013/processed/previews/<uid>.png")
    ap.add_argument("--target-height", type=int, default=64)
    args = ap.parse_args()

    sample = parse_inkml(args.inkml_file)
    strokes_norm, shape = normalise_strokes(sample["strokes"],
                                            target_height=args.target_height)
    img = rasterise(strokes_norm, shape)

    if args.save is not None:
        if args.save == "__AUTO__":
            uid = sample["uid"]
            out_path = Path("data/crohme2013/processed/previews") / f"{uid}.png"
        else:
            out_path = Path(args.save)
            if out_path.parts[-1].find("/") == -1 and len(out_path.parts) == 1:
                # only a filename → still put it in previews/
                out_path = Path("data/crohme2013/processed/previews") / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out_path)
        print(f"Saved → {out_path}  | shape = {img.shape}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.imshow(img, cmap="gray")
        plt.title(f"{sample['uid']}  {sample['latex']}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    _cli()
