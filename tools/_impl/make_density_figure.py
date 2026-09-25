"""make_density_figure.py -- how the task/social trade-off moves with pedestrian
density, and how badly the learned arms transfer to an unseen density.

For each method the three points (P=5, 9, 16) are connected in order, so the
arrow of increasing density is visible.  Classical controllers are evaluated
per density; the learned arms are TRAINED at P=9 and evaluated zero-shot at
P=5/16 (their P=16 collapse is a result, not a plotting artefact).

Usage: python scripts/make_density_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = os.path.join(PROJ, "results", "bench")
K = [5, 9, 16]

# method -> per-density (file template, label)
SERIES = {
    "ORCA": ("cls_orca_P{k}.json", "ORCA-RVO2", "tab:blue", "s"),
    "SFM (k=5)": ("cls_sfm_bal_P{k}.json", "SFM-k5-A6-B0.5", "tab:cyan", "s"),
    "DWA": ("cls_dwa_P{k}.json", "DWA", "tab:olive", "s"),
    "scripted-detour": ("cls_orca_P{k}.json", "scripted-detour", "tab:green", "o"),
    "scripted-beeline": ("cls_orca_P{k}.json", "scripted-beeline", "gray", "o"),
    "MAPPO (trained@9)": (None, "MAPPO-s0", "tab:red", "^"),
    "AMC (trained@9)": (None, "AMC-s0", "tab:purple", "^"),
    "AMC+CBF (trained@9)": (None, "AMC-s0", "tab:pink", "D"),
}


def g(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def summary(fn):
    d = json.load(open(os.path.join(B, fn)))
    return {r["label"]: r for r in (d["summary"] if isinstance(d, dict) else d)}


def learned_files(k):
    """learned arms: P=9 from p9_main.json, else the zero-shot files"""
    if k == 9:
        return "p9_main.json"
    return f"learned_P{k}_zeroshot.json"


def main():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for name, (tmpl, label, colour, marker) in SERIES.items():
        xs, ys = [], []
        for k in K:
            fn = tmpl.format(k=k) if tmpl else learned_files(k)
            try:
                rows = summary(fn)
            except FileNotFoundError:
                continue
            r = rows.get(label)
            if r is None:
                continue
            ss = r.get("social_succ") or {}
            xs.append(100 * g(r["success"]))
            ys.append(100 * g(ss.get("0.55", ss.get(0.55))))
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, color=colour, label=name, lw=1.4, ms=7)
        for x, y, k in zip(xs, ys, K[:len(xs)]):
            ax.annotate(f"P={k}", (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=colour)
    ax.set_xlabel("reach success (%)  → task")
    ax.set_ylabel("social success @0.55 m (%)  → social")
    ax.set_title("Density sweep: P = 5 → 9 → 16 (arrows point to higher density)\n"
                 "learned arms trained at P=9 only (P=5/16 zero-shot); "
                 "classical arms evaluated per density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = os.path.join(B, "pareto_density_curves.png")
    fig.savefig(out, dpi=140)
    print("saved", os.path.relpath(out, PROJ))


if __name__ == "__main__":
    main()
