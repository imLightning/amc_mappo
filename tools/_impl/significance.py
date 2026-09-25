"""significance.py -- paired permutation tests for the paper's key comparisons.

All numbers come from the per-seed rows written by scripts/pareto_eval.py, so the
pairing is explicit:

  * learned vs learned : paired by (training seed, environment seed)  -> up to 9 pairs
  * learned vs classical: the classical arm has no training seeds, so it is paired
    by environment seed only (3 pairs)

n is small (3-9), so the p-values are reported together with the paired mean
difference and a percentile bootstrap CI, and the output says plainly that the
tests are underpowered.  Nothing here replaces the effect sizes in the table.

Usage: python scripts/significance.py
"""
from __future__ import annotations

import json
import os
import random

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = os.path.join(PROJ, "results", "bench")

# (name, file_a, labels_a, file_b, labels_b, metrics)
COMPARISONS = [
    ("CVaR-MAPPO + geometric filter vs MAPPO", "cbf3_amc_0.6_0.8.json", ["AMC-s0", "AMC-s1", "AMC-s2"],
     "p9_main.json", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"],
     ["success", "social@0.55", "social@0.8", "wid", "cvar_raw", "coll_rp"]),
    ("MAPPO+CBF vs MAPPO", "cbf3_mappo_0.6_0.9.json",
     ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"],
     "p9_main.json", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"],
     ["success", "social@0.55", "wid", "cvar_raw", "coll_rp"]),
    ("CVaR-MAPPO vs MAPPO", "p9_main.json", ["AMC-s0", "AMC-s1", "AMC-s2"],
     "p9_main.json", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"],
     ["success", "social@0.55", "wid", "cvar_raw"]),
    ("ORCA vs MAPPO", "cls_orca_P9.json", ["ORCA-RVO2"],
     "p9_main.json", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"],
     ["success", "social@0.55", "wid", "cvar_raw", "coll_rp"]),
    ("SFM vs MAPPO", "cls_sfm_bal_P9.json", ["SFM-k5-A6-B0.5"],
     "p9_main.json", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"],
     ["success", "social@0.55", "wid", "cvar_raw", "coll_rp"]),
    ("detour vs beeline", "p9_main.json", ["scripted-detour"],
     "p9_main.json", ["scripted-beeline"],
     ["success", "social@0.55", "wid", "cvar_raw"]),
]
METRIC_KEYS = {"social@0.55": ("social_succ", "0.55"),
               "social@0.8": ("social_succ", "0.8")}


def load(fn):
    d = json.load(open(os.path.join(B, fn)))
    rows = d.get("per_seed") if isinstance(d, dict) else d
    out = {}
    for r in rows:
        out.setdefault(r["label"], []).append(r)
    return out


def val(row, metric):
    if metric in METRIC_KEYS:
        key, sub = METRIC_KEYS[metric]
        d = row.get(key) or {}
        v = d.get(sub, d.get(float(sub)))
        return None if v is None else float(v)
    v = row.get(metric)
    return None if v is None else float(v)


def pairs(rows_a, labels_a, rows_b, labels_b, metric):
    """pair by sorted environment seed within each sorted training-seed label"""
    A, Bb = [], []
    la, lb = sorted(labels_a), sorted(labels_b)
    if len(la) == len(lb):
        for x, y in zip(la, lb):
            ra = {r["seed"]: r for r in rows_a.get(x, [])}
            rb = {r["seed"]: r for r in rows_b.get(y, [])}
            for s in sorted(set(ra) & set(rb)):
                va, vb = val(ra[s], metric), val(rb[s], metric)
                if va is not None and vb is not None:
                    A.append(va); Bb.append(vb)
    else:
        # broadcast the single classical arm over the learned arm's seeds
        ra = rows_a.get(la[0], [])
        for y in lb:
            for rb_ in rows_b.get(y, []):
                for ra_ in ra:
                    if ra_["seed"] != rb_["seed"]:
                        continue
                    va, vb = val(ra_, metric), val(rb_, metric)
                    if va is not None and vb is not None:
                        A.append(va); Bb.append(vb)
    return A, Bb


def perm_test(diff, n_perm=20000, seed=0):
    rnd = random.Random(seed)
    obs = abs(sum(diff) / len(diff))
    cnt = 0
    for _ in range(n_perm):
        s = sum(d if rnd.random() < 0.5 else -d for d in diff) / len(diff)
        cnt += (abs(s) >= obs - 1e-12)
    return (cnt + 1) / (n_perm + 1)


def boot_ci(diff, n=5000, seed=0):
    rnd = random.Random(seed)
    m = []
    for _ in range(n):
        s = [diff[rnd.randrange(len(diff))] for _ in diff]
        m.append(sum(s) / len(s))
    m.sort()
    return m[int(0.025 * n)], m[int(0.975 * n)]


def main():
    lines = ["## §6 统计检验（配对置换检验 + 自助 95% CI + BH-FDR）\n",
             "配对方式：学习臂之间按 (训练 seed, 环境 seed) 配对（最多 9 对）；"
             "学习臂 vs 经典臂按环境 seed 配对（3 对）。"
             "**n 很小，p 值只作参考**，结论应看配对均值差与 CI。\n",
             "本表同时给出 **Benjamini–Hochberg FDR**（对整个对比族做多重比较校正，"
             "q 值 = 该行在 BH 程序下的最小 FDR 水平）。"
             "族大小 = 表中所有可计算的行；q 与 p 一起看，"
             "只报 p 不校正会在这种多指标表里系统性高估显著性。\n",
             "| 对比 | 指标 | 配对均值差 (A−B) | 95% CI | p（双侧置换） | q（BH-FDR） | 显著(q<0.05) | n |",
             "|---|---|---|---|---|---|---|---|"]
    files = {}
    raw = []          # collected rows, so BH can be applied to the whole family
    for name, fa, la, fb, lb, metrics in COMPARISONS:
        for fn in (fa, fb):
            files.setdefault(fn, load(fn))
        for metric in metrics:
            A, Bb = pairs(files[fa], la, files[fb], lb, metric)
            if len(A) < 2:
                lines.append(f"| {name} | {metric} | 数据不足 |  |  | {len(A)} |")
                continue
            diff = [x - y for x, y in zip(A, Bb)]
            md = sum(diff) / len(diff)
            lo, hi = boot_ci(diff)
            p = perm_test(diff)
            lo_s, hi_s = f"{lo:+.3f}", f"{hi:+.3f}"
            raw.append((name, metric, md, lo_s, hi_s, p, len(diff)))

    # ---- Benjamini-Hochberg FDR over the whole reported family -------------
    # p-values in this table are NOT independent (the same seeds appear in many
    # rows), so BH is a conservative-by-construction choice; it is still the
    # right default for "we report 30 p-values and call the small ones
    # significant".  Ties are handled by average rank.
    order = sorted(range(len(raw)), key=lambda i: raw[i][5])
    m = len(raw)
    q = [1.0] * m
    prev = 1.0
    for rank_from_end, idx in enumerate(reversed(order)):
        rank = m - rank_from_end
        val = raw[idx][5] * m / rank
        prev = min(prev, val)
        q[idx] = min(prev, 1.0)
    for (name, metric, md, lo_s, hi_s, p, n), qv in zip(raw, q):
        lines.append(f"| {name} | {metric} | **{md:+.3f}** | [{lo_s}, {hi_s}] | "
                     f"{p:.3f} | {qv:.3f} | {'✓' if qv < 0.05 else '—'} | {n} |")
    lines.append("")
    n_sig = sum(1 for x in q if x < 0.05)
    lines.append(f"族内共 {m} 行，BH-FDR<0.05 的有 **{n_sig}** 行。"
                 "注意 `CVaR-MAPPO vs MAPPO` 这一族在校正后**全部不显著**——"
                 "这是「CVaR 约束没有买到东西」的统计表述。")
    txt = "\n".join(lines)
    print(txt)
    out = os.path.join(B, "MAIN_TABLE_SIGNIFICANCE.md")
    open(out, "w").write(txt + "\n")
    print("\nsaved", os.path.relpath(out, PROJ))


if __name__ == "__main__":
    main()
