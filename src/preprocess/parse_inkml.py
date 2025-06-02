# -*- coding: utf-8 -*-
"""parse_inkml.py

Utility to load a CROHME InkML file and extract:
  • UID (file stem)
  • LaTeX ground‑truth (annotation type="truth")
  • Raw pen‑strokes as a list of N×2 float32 numpy arrays

It is deliberately minimal but robust against the weird spacing / empty
strings that appear inside some trace elements.

CLI demo
--------
$ python3 -m src.preprocess.parse_inkml data/crohme2013/raw/TrainINKML/extension/2_em_3.inkml --summary

You should get something like::

  UID      : 2_em_3
  LaTeX    : x^{2}+y^{2}=z^{2}
  #strokes : 30
  Total pts: 418

Add the flag ``--show`` to display a quick matplotlib rendering of the
vector strokes (handy sanity‑check as we build the rasteriser next).
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

###############################################################################
# Core parser
###############################################################################

def parse_inkml(path: str | Path) -> Dict[str, Any]:
    """Parse a CROHME InkML file.

    Parameters
    ----------
    path: str or Path
        Path to the ``.inkml`` file.

    Returns
    -------
    sample : dict with keys
        ``uid``     – file stem (str)
        ``latex``   – LaTeX ground‑truth or ``None`` if missing (str | None)
        ``strokes`` – list of numpy arrays, each of shape (N_i, 2)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Parse XML with namespace‑agnostic tag search (InkML files declare the
    # default namespace but we can ignore it by using {*} wild‑cards).
    root = ET.parse(path).getroot()

    # ---------------------------------------------------------------
    # 1) LaTeX ground truth (may be absent in the *unlabeled* test set)
    # ---------------------------------------------------------------
    latex = None
    for ann in root.findall('.//{*}annotation'):
        if ann.attrib.get('type') == 'truth':
            latex = ann.text.strip() if ann.text else ''
            break

    # ---------------------------------------------------------------
    # 2) Strokes: each <trace> element contains a comma‑separated list of
    #    points; each point is whitespace‑separated x y (optionally plus t).
    #    We ignore time and Z values and keep only (x, y).
    # ---------------------------------------------------------------
    strokes: List[np.ndarray] = []
    for trace in root.findall('.//{*}trace'):
        txt = trace.text
        if not txt:
            continue  # empty element – rare but possible
        pts: List[List[float]] = []
        for token in txt.strip().split(','):
            token = token.strip()
            if not token:
                continue  # handles stray commas like ", ,"
            coords = token.split()
            if len(coords) < 2:
                continue  # malformed point – skip
            try:
                x, y = float(coords[0]), float(coords[1])
            except ValueError:
                # Occasionally one of coords is an empty string → skip point
                continue
            pts.append([x, y])
        if pts:
            strokes.append(np.asarray(pts, dtype=np.float32))

    return {
        'uid': path.stem,
        'latex': latex,
        'strokes': strokes,
    }

###############################################################################
# CLI helper – lets us poke a single file quickly
###############################################################################

def _cli() -> None:
    ap = argparse.ArgumentParser(description="Parse and summarise a CROHME InkML file")
    ap.add_argument('inkml_file', type=str, help='Path to .inkml')
    ap.add_argument('--show', action='store_true', help='Quick matplotlib preview of strokes')
    ap.add_argument('--summary', action='store_true', help='Print textual summary (default behaviour)')

    args = ap.parse_args()
    sample = parse_inkml(args.inkml_file)

    if args.summary or (not args.show):
        tot_pts = sum(len(s) for s in sample['strokes'])
        print(f"UID      : {sample['uid']}")
        print(f"LaTeX    : {sample['latex']}")
        print(f"#strokes : {len(sample['strokes'])}")
        print(f"Total pts: {tot_pts}")

    if args.show:
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise SystemExit("matplotlib is required for --show") from e

        for stroke in sample['strokes']:
            plt.plot(stroke[:, 0], -stroke[:, 1], linewidth=1)  # invert Y for natural look
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.title(sample['uid'])
        plt.show()


if __name__ == '__main__':
    _cli()
