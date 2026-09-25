"""seed_aggregate.py -- average the cross-density configurations over training seeds.

The cross-density rows come from separate runs per training seed (seed 0 everywhere,
plus seed 1 replicates for the long / weighted / wide curricula).  This script
prints mean +/- range over the available seeds per density, so the reported
numbers carry a training-seed error bar rather than a single run.

Usage: python scripts/seed_aggregate.py > results/bench/SEED_AGGREGATE.md
"""
from __future__ import annotations

import json
import os

# configuration -> {P: [(file, label), ...]} over training seeds
CONFIGS = {
    "long curriculum 200/200/1200": {
        5: [("dlong_P5_base.json", "dlong-P5-base"),
            ("dlongs1_P5.json", "dlongs1-P5")],
        9: [("dlong_P9_base.json", "dlong-P9-base"),
            ("dlongs1_P9.json", "dlongs1-P9")],
        16: [("dlong_P16_base.json", "dlong-P16-base"),
             ("dlongs1_P16.json", "dlongs1-P16")],
    },
    "wide curriculum (covers P=20)": {
        5: [("wide_P5_base.json", "wide-P5-base"),
            ("wides1_P5.json", "wides1-P5")],
        9: [("wide_P9_base.json", "wide-P9-base"),
            ("wides1_P9.json", "wides1-P9")],
        16: [("wide_P16_base.json", "wide-P16-base"),
             ("wides1_P16.json", "wides1-P16")],
        20: [("wide_P20_base.json", "wide-P20-base"),
             ("wides1_P20.json", "wides1-P20")],
    },
    "weighted 300/400/900 (recommended)": {
        5: [("wtd_P5_base.json", "wtd-P5-base"),
            ("wtds1_P5_base.json", "wtds1-P5-base")],
        9: [("wtd_P9_base.json", "wtd-P9-base"),
            ("wtds1_P9_base.json", "wtds1-P9-base")],
        16: [("wtd_P16_base.json", "wtd-P16-base"),
             ("wtds1_P16_base.json", "wtds1-P16-base")],
    },
    # the RECOMMENDED row itself (weighted + safety layer, tau=0.75 lex):
    # the seed-1 evaluation already existed (wtds1_P*_t075p.json) but was never
    # aggregated, so the headline row had no error bar in the documents.
    "weighted 300/400/900 + layer (tau=0.75, lex) -- RECOMMENDED": {
        5: [("wtd_P5_t075p.json", "wtd-P5-t075p"),
            ("wtds1_P5_t075p.json", "wtds1-P5-t075p")],
        9: [("wtd_P9_t075p.json", "wtd-P9-t075p"),
            ("wtds1_P9_t075p.json", "wtds1-P9-t075p")],
        16: [("wtd_P16_t075p.json", "wtd-P16-t075p"),
             ("wtds1_P16_t075p.json", "wtds1-P16-t075p")],
    },
    "wide + layer (tau=0.75, lex)": {
        5: [("wide_P5_t05p.json", "wide-P5-t05p"),
            ("wides1_P5_t075p.json", "wides1-P5-t075p")],
        9: [("wide_P9_t05p.json", "wide-P9-t05p"),
            ("wides1_P9_t075p.json", "wides1-P9-t075p")],
        16: [("wide_P16_t05p.json", "wide-P16-t05p"),
             ("wides1_P16_t075p.json", "wides1-P16-t075p")],
        20: [("wide_P20_t05p.json", "wide-P20-t05p"),
             ("wides1_P20_t075p.json", "wides1-P20-t075p")],
    },
    "uniform interleave 5/9/12/16/20": {
        5: [("cyc_P5_base.json", "cyc-P5-base"),
            ("cycs1_P5_base.json", "cycs1-P5-base")],
        9: [("cyc_P9_base.json", "cyc-P9-base"),
            ("cycs1_P9_base.json", "cycs1-P9-base")],
        16: [("cyc_P16_base.json", "cyc-P16-base"),
             ("cycs1_P16_base.json", "cycs1-P16-base")],
        20: [("cyc_P20_base.json", "cyc-P20-base"),
             ("cycs1_P20_base.json", "cycs1-P20-base")],
    },
    "difficulty-weighted interleave 5:1,9:2,12:2,16:3,20:4": {
        5: [("wcyc_P5_base.json", "wcyc-P5-base"),
            ("wcycs1_P5_base.json", "wcycs1-P5-base")],
        9: [("wcyc_P9_base.json", "wcyc-P9-base"),
            ("wcycs1_P9_base.json", "wcycs1-P9-base")],
        16: [("wcyc_P16_base.json", "wcyc-P16-base"),
             ("wcycs1_P16_base.json", "wcycs1-P16-base")],
        20: [("wcyc_P20_base.json", "wcyc-P20-base"),
             ("wcycs1_P20_base.json", "wcycs1-P20-base")],
    },
    "uniform interleave + layer (tau=0.75, lex)": {
        5: [("cyc_P5_t075p.json", "cyc-P5-t075p"),
            ("cycs1_P5_t075p.json", "cycs1-P5-t075p")],
        9: [("cyc_P9_t075p.json", "cyc-P9-t075p"),
            ("cycs1_P9_t075p.json", "cycs1-P9-t075p")],
        16: [("cyc_P16_t075p.json", "cyc-P16-t075p"),
             ("cycs1_P16_t075p.json", "cycs1-P16-t075p")],
        20: [("cyc_P20_t075p.json", "cyc-P20-t075p"),
             ("cycs1_P20_t075p.json", "cycs1-P20-t075p")],
    },
    "difficulty-weighted interleave + layer (tau=0.75, lex)": {
        5: [("wcyc_P5_t075p.json", "wcyc-P5-t075p"),
            ("wcycs1_P5_t075p.json", "wcycs1-P5-t075p")],
        9: [("wcyc_P9_t075p.json", "wcyc-P9-t075p"),
            ("wcycs1_P9_t075p.json", "wcycs1-P9-t075p")],
        16: [("wcyc_P16_t075p.json", "wcyc-P16-t075p"),
             ("wcycs1_P16_t075p.json", "wcycs1-P16-t075p")],
        20: [("wcyc_P20_t075p.json", "wcyc-P20-t075p"),
             ("wcycs1_P20_t075p.json", "wcycs1-P20-t075p")],
    },
}


def val(f, lab):
    p = os.path.join("results/bench", f)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    for s in d["summary"]:
        if s["label"] == lab:
            soc = s["social_succ"]["0.55"]
            return dict(success=100 * s["success"][0], cvar=s["cvar_raw"][0],
                        social=100 * (soc[0] if isinstance(soc, list) else soc),
                        coll_rp=100 * s["coll_rp"][0], coll_rr=100 * s["coll_rr"][0])
    return None


def main():
    print("# 跨密度配置的训练 seed 聚合（mean ± range）\n")
    print("每个配置的每个密度给出各 seed 单独值与均值；`range` = 最大值−最小值。\n")
    for cfg, perP in CONFIGS.items():
        print(f"## {cfg}\n")
        print("| P | seeds n | success (mean ± range) | 各 seed | cvar | social@0.55 |")
        print("|---|---|---|---|---|---|")
        for P, specs in sorted(perP.items()):
            got = [val(f, l) for f, l in specs]
            got = [g for g in got if g]
            if not got:
                print(f"| {P} | 0 | | | | |")
                continue
            n = len(got)
            su = [g["success"] for g in got]
            cv = [g["cvar"] for g in got]
            so = [g["social"] for g in got]
            rng = (max(su) - min(su)) if n > 1 else 0.0
            print(f"| {P} | {n} | {sum(su)/n:.1f} ± {rng:.1f} | "
                  f"{', '.join(f'{x:.1f}' for x in su)} | {sum(cv)/n:.2f} | {sum(so)/n:.1f} |")
        print()


if __name__ == "__main__":
    main()
