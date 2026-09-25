"""frontier_summary.py -- every constrained arm vs the analytic frontier.

Synthesises the whole CVaR investigation into one table: for each evaluated
comparison file, take the reference envelope (scripted-beeline / scripted-detour
from the SAME file, so scene and seeds match) and every learned or constrained
arm, and report the gap to the detour.

The decisive question the table answers is not "did cvar fall?" but "is the
constrained policy Pareto-dominated by a controller we already have?".  A
constrained arm that gives up task success to buy a little cost is dominated by
the analytic detour whenever the detour has BOTH higher success and lower cvar.

Usage:  python scripts/frontier_summary.py > results/bench/FRONTIER_SUMMARY.md
"""
from __future__ import annotations

import glob
import json
import os

# comparison files, in the order the investigation produced them
FILES = [
    "p9_main.json", "amc_bind_P9.json", "amc_enc_P9.json", "amc_surrogate_P9.json",
    "acc2_P9.json", "acc2_lam10_P9.json", "acc2_lam20_P9.json",
    "acc1fix_P9.json", "amc_fix_P9.json", "bc_P9.json", "bc_amc_P9.json",
    "bc2_P9.json", "bc2_amc_P9.json", "bounded_P9.json", "long_P9.json",
    "acc2_lam50_P9.json",
]


def load(name):
    p = os.path.join("results/bench", name)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {s["label"]: s for s in d.get("summary", [])}


def val(s, k):
    v = s.get(k)
    if isinstance(v, list):
        return v[0]
    return v


DISPLAY = {"BC+AMC": "BC + CVaR-MAPPO", "BC2+AMC": "BC2 + CVaR-MAPPO"}


def main():
    print("# 约束臂 vs 解析前沿（同文件、同场景、同种子）\n")
    print("`Δ` 一律是**相对同一文件里的 scripted-detour**；`dominated` = 该臂的 success 更低"
          "且 cvar 更高，即被绕行控制器**双目标支配**。\n")
    hdr = ("| 文件 | 臂 | success | cvar | coll_rp | social@0.55 | Δsuccess | Δcvar | 判定 |\n"
           "|---|---|---|---|---|---|---|---|---|")
    print(hdr)
    for f in FILES:
        rows = load(f)
        if rows is None:
            print(f"| `{f}` | — | | | | | | | 未跑 |")
            continue
        det = rows.get("scripted-detour")
        if det is None:
            continue
        d_s, d_c = val(det, "success"), val(det, "cvar_raw")
        for lab, s in rows.items():
            if lab.startswith("scripted"):
                continue
            succ, cv = val(s, "success"), val(s, "cvar_raw")
            soc = (s.get("social_succ") or {}).get("0.55")
            soc = soc[0] if isinstance(soc, list) else soc
            ds, dc = succ - d_s, cv - d_c
            if ds < 0 and dc > 0:
                verdict = "**被支配**"
            elif ds >= 0 and dc <= 0:
                verdict = "**支配绕行**"
            else:
                verdict = "互有胜负"
            shown = DISPLAY.get(lab, lab)
            print(f"| `{f}` | {shown} | {succ:.3f} | {cv:.2f} | "
                  f"{val(s, 'coll_rp'):.3f} | {soc if soc is None else round(soc,3)} | "
                  f"{ds:+.3f} | {dc:+.2f} | {verdict} |")
    print("\n（参考：`scripted-detour` 在 P=9/amax=1.0 上是 success 0.878 / cvar 20.17；"
          "在 amax=2.0 上是 0.945 / 18.31。绕行控制器本身**不用学习**，"
          "所以任何以任务换成本的学习臂，只要落在它的右上方就是被支配的。）")


if __name__ == "__main__":
    main()
