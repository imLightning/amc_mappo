"""paired_test.py -- paired-by-seed comparison of two arms in the bench JSONs.

The main-table significance script (`scripts/significance.py`) covers the rows of
the main table.  The CVaR failure analysis adds rows that live in their own
files (acc2, bc, bc2, postfix, lambda sweep, ...), and each of those files
contains BOTH the constrained arm and its own unconstrained starting point,
evaluated on the SAME three environment seeds.  That makes a paired test the
right comparison, and it is the only one that controls for the crowd sample.

Usage:
  python scripts/paired_test.py acc2_P9.json:AMC-acc2:MAPPO-acc2 \\
      bc_P9.json:BC-detour:bc_amc_P9.json:BC+AMC
  # pairs across files are written  fileA:labelA:fileB:labelB
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def load_row(spec):
    """'file.json:label' -> {seed: row}"""
    path, label = spec.rsplit(":", 1)
    d = json.load(open(os.path.join("results/bench", path)))
    out = {}
    for r in d.get("per_seed", []):
        if r.get("label") == label:
            out[r["seed"]] = r
    if not out:
        raise SystemExit(f"label {label!r} not found in {path}")
    return out


def main():
    args = sys.argv[1:]
    if len(args) % 2 != 0:
        raise SystemExit("give an even number of file:label specs")
    for i in range(0, len(args), 2):
        a = load_row(args[i])
        b = load_row(args[i + 1])
        seeds = sorted(set(a) & set(b))
        print(f"\n=== {args[i]}  vs  {args[i+1]}   (paired over seeds {seeds})")
        for key in ("success", "cvar_raw", "coll_rp", "min_rp", "wid", "nav_s"):
            va = np.array([a[s][key] for s in seeds], float)
            vb = np.array([b[s][key] for s in seeds], float)
            d = va - vb
            # paired t-test (n is small; report p as a reference only)
            if len(seeds) > 1 and d.std(ddof=1) > 0:
                t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                from math import erf, sqrt
                # two-sided p via the normal approximation (n=3 -> indicative)
                p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
            else:
                t, p = float("nan"), float("nan")
            print(f"  {key:9s} A {va.mean():8.4f}  B {vb.mean():8.4f}  "
                  f"diff {d.mean():+8.4f} +- {d.std(ddof=1) if len(d)>1 else 0:.4f}  "
                  f"paired t {t:+5.2f}  p~{p:.3f}")


if __name__ == "__main__":
    main()
