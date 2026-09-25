"""density_table.py -- the consolidated cross-density table (paper Table A.2/A.4).

One place where every density row lives, with the training budget it came from:
zero-shot transfer, density-specific training, and the four density curricula
(800 it, 1600 it P=16-heavy, 300/400/900 difficulty-weighted, and the wide one
that covers P=20), each with and without the affect-aware safety layer.

Usage: python scripts/density_table.py > results/bench/DENSITY_TABLE.md
"""
from __future__ import annotations

import json
import os

ROWS = [
    # (kind, budget label, P, file, label, note)
    # ---- same-density classical references (must be read per density!) -------
    ("classical", "ORCA (RVO2), per-density", 5, "cls_orca_P5.json", "ORCA-RVO2", ""),
    ("classical", "ORCA (RVO2), per-density", 9, "cls_orca_P9.json", "ORCA-RVO2", ""),
    ("classical", "ORCA (RVO2), per-density", 16, "cls_orca_P16.json", "ORCA-RVO2", ""),
    ("classical", "ORCA (RVO2), per-density", 20, "cls_orca_P20.json", "ORCA-RVO2", ""),
    ("classical", "scripted-detour, per-density", 5, "cls_orca_P5.json", "scripted-detour", ""),
    ("classical", "scripted-detour, per-density", 9, "cls_orca_P9.json", "scripted-detour", ""),
    ("classical", "scripted-detour, per-density", 16, "cls_orca_P16.json", "scripted-detour", ""),
    ("classical", "scripted-detour, per-density", 20, "cls_orca_P20.json", "scripted-detour", ""),
    ("zero-shot", "P=9 policy, zero-shot", 5, "learned_P5_zeroshot.json", "MAPPO-s0", ""),
    ("zero-shot", "P=9 policy, zero-shot", 9, "scene_base.json", "MAPPO-P9", ""),
    ("zero-shot", "P=9 policy, zero-shot", 16, "learned_P16_zeroshot.json", "MAPPO-s0", ""),
    ("zero-shot", "P=9 policy + rms_clip", 16, "rmsclip_P16.json", "MAPPO-rmsclip-P16", ""),
    ("specific", "density-specific fine-tune", 16, "p16_base.json", "MAPPO-P16", ""),
    ("specific", "density-specific from scratch", 16, "p16s_base.json", "MAPPO-P16-scratch", ""),
    ("curriculum", "800 it: 200/200/200 @5/11/16", 5, "dens_P5_base.json", "dens-P5", ""),
    ("curriculum", "800 it: 200/200/200 @5/11/16", 9, "dens_P9_base.json", "dens-P9", ""),
    ("curriculum", "800 it: 200/200/200 @5/11/16", 16, "dens_P16_base.json", "dens-P16", ""),
    ("curriculum", "1600 it: 200/200/1200 @5/11/16", 5, "dlong_P5_base.json", "dlong-P5-base", ""),
    ("curriculum", "1600 it: 200/200/1200 @5/11/16", 9, "dlong_P9_base.json", "dlong-P9-base", ""),
    ("curriculum", "1600 it: 200/200/1200 @5/11/16", 16, "dlong_P16_base.json", "dlong-P16-base", ""),
    ("curriculum", "1600 it: 300/400/900 @5/11/16 (recommended)", 5, "wtd_P5_base.json", "wtd-P5-base", ""),
    ("curriculum", "1600 it: 300/400/900 @5/11/16 (recommended)", 9, "wtd_P9_base.json", "wtd-P9-base", ""),
    ("curriculum", "1600 it: 300/400/900 @5/11/16 (recommended)", 16, "wtd_P16_base.json", "wtd-P16-base", ""),
    ("curriculum", "1600 it: 200/200/200/1000 @5/11/16/20", 5, "wide_P5_base.json", "wide-P5-base", ""),
    ("curriculum", "1600 it: 200/200/200/1000 @5/11/16/20", 9, "wide_P9_base.json", "wide-P9-base", ""),
    ("curriculum", "1600 it: 200/200/200/1000 @5/11/16/20", 16, "wide_P16_base.json", "wide-P16-base", ""),
    ("curriculum", "1600 it: 200/200/200/1000 @5/11/16/20", 20, "wide_P20_base.json", "wide-P20-base", ""),
    # interleaved (domain-randomised) curricula
    ("interleave", "1600 it interleave 5/9/12/16/20 (320 each)", 5, "cyc_P5_base.json", "cyc-P5-base", ""),
    ("interleave", "1600 it interleave 5/9/12/16/20 (320 each)", 9, "cyc_P9_base.json", "cyc-P9-base", ""),
    ("interleave", "1600 it interleave 5/9/12/16/20 (320 each)", 16, "cyc_P16_base.json", "cyc-P16-base", ""),
    ("interleave", "1600 it interleave 5/9/12/16/20 (320 each)", 20, "cyc_P20_base.json", "cyc-P20-base", ""),
    ("interleave", "1600 it weighted interleave 1/2/2/3/4", 5, "wcyc_P5_base.json", "wcyc-P5-base", ""),
    ("interleave", "1600 it weighted interleave 1/2/2/3/4", 9, "wcyc_P9_base.json", "wcyc-P9-base", ""),
    ("interleave", "1600 it weighted interleave 1/2/2/3/4", 16, "wcyc_P16_base.json", "wcyc-P16-base", ""),
    ("interleave", "1600 it weighted interleave 1/2/2/3/4", 20, "wcyc_P20_base.json", "wcyc-P20-base", ""),
    # unseen densities under the 800-it curriculum (competence boundary)
    ("unseen", "800 it curriculum @ UNSEEN P=12", 12, "dens_P12_base.json", "dens-P12-base", ""),
    ("unseen", "800 it curriculum @ UNSEEN P=20", 20, "dens_P20_base.json", "dens-P20-base", ""),
    ("+layer", "800 it curriculum + layer @ P=12", 12, "dens_P12_layer.json", "dens-P12-layer", ""),
    ("+layer", "800 it curriculum + layer @ P=20", 20, "dens_P20_layer.json", "dens-P20-layer", ""),
    # with the safety layer
    ("+layer", "800 it curriculum + layer", 5, "dens_P5_t05p.json", "dens-P5+layer", ""),
    ("+layer", "800 it curriculum + layer", 9, "dens_P9_t05p.json", "dens-P9+layer", ""),
    ("+layer", "800 it curriculum + layer", 16, "dens_P16_t05p.json", "dens-P16+layer", ""),
    ("+layer", "1600 it long + layer", 5, "dlong_P5_t05p.json", "dlong-P5-t05p", ""),
    ("+layer", "1600 it long + layer", 9, "dlong_P9_t05p.json", "dlong-P9-t05p", ""),
    ("+layer", "1600 it long + layer", 16, "dlong_P16_t05p.json", "dlong-P16-t05p", ""),
    ("+layer", "**recommended** weighted + layer (tau=0.75, lex)", 5, "wtd_P5_t075p.json", "wtd-P5-t075p", ""),
    ("+layer", "**recommended** weighted + layer (tau=0.75, lex)", 9, "wtd_P9_t075p.json", "wtd-P9-t075p", ""),
    ("+layer", "**recommended** weighted + layer (tau=0.75, lex)", 16, "wtd_P16_t075p.json", "wtd-P16-t075p", ""),
    ("+layer", "wide + layer (tau=0.75, lex)", 5, "wide_P5_t05p.json", "wide-P5-t05p", ""),
    ("+layer", "wide + layer (tau=0.75, lex)", 9, "wide_P9_t05p.json", "wide-P9-t05p", ""),
    ("+layer", "wide + layer (tau=0.75, lex)", 16, "wide_P16_t05p.json", "wide-P16-t05p", ""),
    ("+layer", "wide + layer (tau=0.75, lex)", 20, "wide_P20_t05p.json", "wide-P20-t05p", ""),
    ("+layer", "interleave + layer (tau=0.75, lex)", 5, "cyc_P5_t075p.json", "cyc-P5-t075p", ""),
    ("+layer", "interleave + layer (tau=0.75, lex)", 9, "cyc_P9_t075p.json", "cyc-P9-t075p", ""),
    ("+layer", "interleave + layer (tau=0.75, lex)", 16, "cyc_P16_t075p.json", "cyc-P16-t075p", ""),
    ("+layer", "interleave + layer (tau=0.75, lex)", 20, "cyc_P20_t075p.json", "cyc-P20-t075p", ""),
    ("+layer", "weighted interleave + layer", 5, "wcyc_P5_t075p.json", "wcyc-P5-t075p", ""),
    ("+layer", "weighted interleave + layer", 9, "wcyc_P9_t075p.json", "wcyc-P9-t075p", ""),
    ("+layer", "weighted interleave + layer", 16, "wcyc_P16_t075p.json", "wcyc-P16-t075p", ""),
    ("+layer", "weighted interleave + layer", 20, "wcyc_P20_t075p.json", "wcyc-P20-t075p", ""),
]


def main():
    print("# 跨密度总表（P=5/9/12/16/20，3 个环境种子 1000/1001/1002）\n")
    print("`+layer` = 情绪安全层（各向异性个人空间 + 提前让路 τ + 词法式联合 QP + 互惠同伴）。\n")
    print("| 类别 | 训练预算 | P | 方法 | success | cvar ↓ | social@0.55 | R–P 接触 | R–R 接触 |")
    print("|---|---|---|---|---|---|---|---|---|")
    last = None
    for kind, budget, P, f, lab, _ in ROWS:
        p = os.path.join("results/bench", f)
        if not os.path.exists(p):
            print(f"| {kind} | {budget} | {P} | {lab} | | | | | |")
            continue
        d = json.load(open(p))
        s = next((r for r in d["summary"] if r["label"] == lab), None)
        if s is None:
            print(f"| {kind} | {budget} | {P} | {lab} | | | | | |")
            continue
        ss = s["social_succ"]["0.55"]
        print(f"| {kind} | {budget} | {P} | {lab} | {100*s['success'][0]:.1f} | "
              f"{s['cvar_raw'][0]:.2f} | {100*ss[0]:.1f} | {100*s['coll_rp'][0]:.1f} | "
              f"{100*s['coll_rr'][0]:.1f} |")
        last = kind
    print("\n⚠ 命名提示：`*_t05p.json` 是历史文件名，这些行实际用 **τ=0.75**"
          "（`--cbf-ttc-gain 0.75`）生成；以脚本里的参数为准，不要按文件名读 τ。")
    print("\n**seed 误差棒**：本表每行是 **seed 0**；`SEED_AGGREGATE.md` 给出各配置的"
          "多 seed 均值 ± 极差（推荐行 = 加权 300/400/900 + 安全层 τ=0.75：P=5 "
          "**74.0 ± 1.0**、P=9 62.8 ± 4.7、P=16 36.3 ± 5.5，n=2）。"
          "两种交错配置的 seed 1 见 `cycs1_*` / `wcycs1_*`。\n")
    print("注：`density-specific` 行是**同密度参考**，但它的定义在各密度**并不相同**，"
          "不能当作同一个基线："
          "P=5 用的是 **P=9 策略**（`learned_P5_zeroshot.json`；P=5 低于训练密度，"
          "属于向下迁移，不存在\"P=5 专用\"训练），P=9 用的是 P=9 策略（`scene_base.json`），"
          "P=16 才是真正在 P=16 微调的 `p16_base.json`。"
          "`curriculum` 行是**同一个策略**在不同密度上的评测。"
          "经典参考必须按密度读：ORCA 在 P=5/9/16/20 = 0.914 / 0.839 / 0.643 / 0.563，"
          "cvar 13.64 / 19.18 / 25.33 / 27.95，social@0.55 74.0% / 51.6% / 28.4% / 21.4%"
          "（`cls_orca_P*.json`）——**用 P=9 的 ORCA 数字去和其他密度比较是错的**。")


if __name__ == "__main__":
    main()
