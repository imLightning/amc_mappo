"""make_pareto_figure.py -- the paper's headline figure from the measured jsons.

Plots reach success (x) against social success @0.55 m (y) for every arm that has
both numbers, so the "task vs social" trade-off is visible at a glance:

    * reference envelope  (beeline / detour / creep / stop)
    * classical           (ORCA, SFM x2, DWA)
    * learned baselines   (MAPPO, AMC-CVaR)
    * explicit filter     (MAPPO+CBF, AMC+CBF)
    * social DRL          (SARL port; marked as not converged)

Everything is read from results/bench/*.json -- no number is typed in.

Usage: python scripts/make_pareto_figure.py [--density 9]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = os.path.join(PROJ, "results", "bench")

# (label, file, arm labels, marker, colour, group)
SPECS = [
    ("beeline", "p9_main.json", ["scripted-beeline"], "o", "tab:gray", "reference"),
    ("detour", "p9_main.json", ["scripted-detour"], "o", "tab:green", "reference"),
    ("creep", "p9_main.json", ["scripted-creep"], "o", "tab:brown", "reference"),
    ("stop", "p9_main.json", ["scripted-stop"], "o", "black", "reference"),
    ("ORCA", "cls_orca_P9.json", ["ORCA-RVO2"], "s", "tab:blue", "classical"),
    ("SFM", "cls_sfm_bal_P9.json", ["SFM-k5-A6-B0.5"], "s", "tab:cyan", "classical"),
    ("SFM-safe", "cls_sfm_safe_P9.json", ["SFM-k3-A6-B0.5"], "s", "deepskyblue", "classical"),
    ("DWA", "cls_dwa_P9.json", ["DWA"], "s", "tab:olive", "classical"),
    ("MAPPO", "p9_main.json", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"], "^", "tab:red", "learned"),
    ("AMC-CVaR", "p9_main.json", ["AMC-s0", "AMC-s1", "AMC-s2"], "^", "tab:purple", "learned"),
    ("MAPPO+CBF", "cbf3_mappo_0.6_0.9.json", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"],
     "D", "tab:orange", "filtered"),
    ("AMC+CBF", "cbf3_amc_0.6_0.8.json", ["AMC-s0", "AMC-s1", "AMC-s2"],
     "D", "tab:pink", "filtered"),
    ("SARL (unconverged)", "sarl_P9_s1.json", ["SARL-s1"], "x", "magenta", "social DRL"),
]


def summ(fn):
    d = json.load(open(os.path.join(B, fn)))
    return d["summary"] if isinstance(d, dict) else d


def g(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(B, "pareto_social_vs_task.png"))
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    xs, ys, ms = [], [], []
    for name, fn, labels, marker, colour, group in SPECS:
        try:
            rows = {r["label"]: r for r in summ(fn)}
        except FileNotFoundError:
            print("skip (missing)", fn)
            continue
        for lab in labels:
            r = rows.get(lab)
            if r is None:
                continue
            ss = r.get("social_succ") or {}
            s55 = g(ss.get("0.55", ss.get(0.55)))
            x, y = 100 * g(r["success"]), 100 * s55
            ax.scatter(x, y, marker=marker, color=colour, s=90, zorder=3)
            ax.annotate(name if len(labels) == 1 else f"{name}",
                        (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
            xs.append(x)
            ys.append(y)
            ms.append((x, y))
            break
    # frontier: pareto-optimal points (max success and max social success)
    pts = sorted(ms, key=lambda p: p[0])
    front = []
    best_y = -1
    for x, y in reversed(pts):
        if y > best_y:
            front.append((x, y))
            best_y = y
    front = sorted(front)
    if len(front) > 1:
        ax.plot([p[0] for p in front], [p[1] for p in front], "--",
                color="k", lw=1, alpha=0.6, label="empirical frontier")
    ax.set_xlabel("reach success (%)  → task")
    ax.set_ylabel("social success @0.55 m (%)  → social")
    ax.set_title("Task vs social trade-off, P=9, held-out env seeds 1000-1002\n"
                 "(learned arms: mean over 3 training seeds; higher-right is better)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print("saved", os.path.relpath(a.out, PROJ))
    print("frontier points:", [(round(x, 1), round(y, 1)) for x, y in front])


if __name__ == "__main__":
    main()
