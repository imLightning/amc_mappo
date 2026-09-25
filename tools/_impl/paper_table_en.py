"""paper_table_en.py -- the main table in English, as markdown and as LaTeX.

Reuses the pooling logic of scripts/fill_main_table.py so the two versions can
never drift apart.  Column set = the paper's Tab. 1:

  method | reach success | social success @0.55 | social success @0.8 |
  min robot-pedestrian distance | robot-pedestrian contact rate |
  robot-robot contact rate | navigation time | wid | cvar

Usage:  python scripts/paper_table_en.py --out results/bench/PAPER_TABLE_EN.md
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._impl.fill_main_table import MAIN, load, pool, f  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUP_EN = {"参考包络": "reference", "经典·安全": "classical (safety)",
            "经典·社会": "classical (social)", "社交 DRL": "social DRL",
            "学习基线": "learned baseline", "本方法": "**ours (AMC-MAPPO)**",
            "对照·约束学习基线": "constrained-learning baseline",
            "学习+滤波": "learned + geometric filter"}


def en(name):
    import re
    name = name.replace("脚本-", "scripted-").replace("**", "")
    name = (name.replace("（本移植，未收敛）", " (port, not converged)")
                .replace("（未收敛）", " (not converged)")
                .replace("（未训练）", " (untrained)"))
    return re.sub(r"[（(][^）)]*[\u4e00-\u9fff][^）)]*[）)]", "", name).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "bench",
                                                  "PAPER_TABLE_EN.md"))
    a = ap.parse_args()
    rows = []
    for name, labels, fn, group in MAIN:
        r = load(fn, labels) if fn else []
        rows.append((GROUP_EN.get(group, group), en(name), pool(r) if r else None))
    md = ["# Table 1 (English). Task/social trade-off, P=9, held-out env seeds 1000-1002",
          "",
          "Values are mean over 3 training seeds (learned arms) or over 3 environment",
          "seeds (non-learned arms). `social@r` = all robots reach their goals AND the",
          "minimum robot-pedestrian distance never drops below r.  Arrows: higher is",
          "better except the contact rates and the time/cost columns.",
          "",
          "| group | method | reach success (%) ↑ | social@0.55 (%) ↑ | social@0.8 (%) ↑ | "
          "min r-p dist (m) ↑ | r-p contact (%) ↓ | r-r contact (%) ↓ | nav time (s) ↓ | "
          "wid ↓ | cvar ↓ |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Task/social trade-off at $P{=}9$ (held-out environment seeds). "
           r"$\mathrm{social}@r$ requires all robots to reach their goals while never "
           r"coming closer than $r$ to a pedestrian.}",
           r"\label{tab:main}", r"\begin{tabular}{llccccccccc}", r"\toprule",
           r"group & method & succ. & soc@0.55 & soc@0.8 & $d_{\min}$ & "
           r"$c_{rp}$ & $c_{rr}$ & $t$ & wid & cvar \\", r"\midrule"]
    for group, name, p in rows:
        if p is None:
            md.append(f"| {group} | {name} |  |  |  |  |  |  |  |  |  |")
            tex.append(f"{group} & {name} & -- & -- & -- & -- & -- & -- & -- & -- & -- \\\\")
            continue
        cells = [f(p['success'], pct=True), f(p['social_succ@0.55'], pct=True),
                 f(p['social_succ@0.8'], pct=True), f(p['min_rp']),
                 f(p['coll_rp'], pct=True), f(p['coll_rr'], pct=True),
                 f(p['nav_s'], 2), f(p['wid']), f(p['cvar_raw'], 2)]
        md.append(f"| {group} | {name} | " + " | ".join(cells) + " |")
        tex.append(f"{group} & {name} & " + " & ".join(cells).replace("±", r"$\pm$") + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    txt = "\n".join(md) + "\n\n## LaTeX\n\n```latex\n" + "\n".join(tex) + "\n```\n"
    print(txt)
    open(a.out, "w").write(txt)
    print("saved", os.path.relpath(a.out, PROJ))


if __name__ == "__main__":
    main()
