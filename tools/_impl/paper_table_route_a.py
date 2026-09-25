"""paper_table_route_a.py -- English paper-ready tables for the affect-aware
safety layer (route A).

Emits (markdown to stdout):
  Table A.1  safety-layer variants at P=9 (3 env seeds), joint QP solver
  Table A.2  density, including the in-density-trained policies
  Table A.3  the matched-dynamics control that de-confounds the main table

Usage: python scripts/paper_table_route_a.py > results/bench/PAPER_TABLE_ROUTE_A_EN.md
"""
from __future__ import annotations

import json
import os

A1 = [
    ("scene_base.json", "MAPPO-P9", "MAPPO (no filter)", False),
    ("emocbf_mappo_geo06.json", "mappo_geo06", "geometric filter, d=0.6", False),
    ("geo_matched_0.70.json", "geo-0.70", "geometric filter, d=0.70 (radius-matched)", False),
    ("emocbf_mappo_geo09.json", "mappo_geo09", "geometric filter, d=0.9", False),
    ("emocbf_mappo_aniso.json", "mappo_aniso", "**anisotropic affect space** (ours)", False),
    ("ttc_t05.json", "t05", "  + early-yield tau=0.5 s", False),
    ("ttc_t10.json", "t10", "  + early-yield tau=1.0 s (knee)", False),
    ("ttc_t15.json", "t15", "  + early-yield tau=1.5 s", False),
    ("ttc_t10p.json", "t10p", "  + tau=1.0 s + reciprocal peer barrier", False),
    ("final_cyc_t10.json", "cyc-tau1.0", "cyclic solver, same tau=1.0 (collapses)", False),
    ("final_orca_t10.json", "ORCA+filter", "ORCA + our layer (orthogonality)", False),
    ("lex_p9_joint.json", "p9_joint", "  flat joint QP (reference)", False),
    ("lex_p9_lex.json", "p9_lex", "  **pedestrian-first (lexicographic) QP**", False),
]
A2 = [
    (5, "learned_P5_zeroshot.json", "MAPPO-s0", "no filter (zero-shot from P=9)"),
    (5, "final_P5_t10.json", "mappo-joint-tau1.0-P5", "+ layer, tau=1.0"),
    (5, "dens_P5_base.json", "dens-P5", "**density curriculum**, no filter"),
    (5, "dens_P5_t05p.json", "dens-P5+layer", "density curriculum + layer"),
    (9, "scene_base.json", "MAPPO-P9", "no filter (P=9 baseline)"),
    (9, "ttc_t10.json", "t10", "+ layer, tau=1.0"),
    (9, "ttc_t10p.json", "t10p", "+ layer, tau=1.0 + peer"),
    (9, "dens_P9_base.json", "dens-P9", "**density curriculum**, no filter"),
    (9, "dens_P9_t05p.json", "dens-P9+layer", "density curriculum + layer"),
    (12, "dens_P12_base.json", "dens-P12-base", "density curriculum at UNSEEN P=12"),
    (16, "learned_P16_zeroshot.json", "MAPPO-s0", "no filter (zero-shot from P=9)"),
    (16, "rmsclip_P16.json", "MAPPO-rmsclip-P16", "no filter (zero-shot, rms_clip fix)"),
    (16, "p16_base.json", "MAPPO-P16", "no filter (fine-tuned at P=16)"),
    (16, "p16tau_t025.json", "t025", "+ layer, tau=0.25 + peer (at P=16)"),
    (16, "dens_P16_base.json", "dens-P16", "**density curriculum**, no filter"),
    (16, "dens_P16_t05p.json", "dens-P16+layer", "density curriculum + layer"),
    (20, "dens_P20_base.json", "dens-P20-base", "density curriculum at UNSEEN P=20"),
]


A3 = [
    ("p9_main.json", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"], "MAPPO (curriculum recipe)"),
    ("p9_acc1fix_3seed.json", ["MAPPO-acc1fix-s0", "MAPPO-acc1fix-s1", "MAPPO-acc1fix-s2"],
     "MAPPO (matched-dynamics recipe)"),
    ("cbf3_mappo_0.6_0.8.json", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"],
     "MAPPO (curriculum) + geometric filter 0.6/0.8"),
    ("cbf3_acc1fix_3seed.json", ["MAPPO-acc1fix-s0", "MAPPO-acc1fix-s1", "MAPPO-acc1fix-s2"],
     "MAPPO (matched dynamics) + geometric filter 0.6/0.8"),
    ("cbf3_amc_0.6_0.8.json", ["AMC-s0", "AMC-s1", "AMC-s2"],
     "CVaR-MAPPO (lambda=0) + geometric filter 0.6/0.8"),
]

# density-curriculum (one policy across densities) rows, per density.
# The classical references come FIRST and are per density on purpose: ORCA at
# P=5 (0.914 / 13.64 / 74.0%) is a very different comparator from ORCA at P=9
# (0.839 / 19.18 / 51.6%), and mixing the two is the mistake this table prevents.
A4 = [
    (5, "cls_orca_P5.json", "ORCA-RVO2", "**classical reference**: ORCA (RVO2) at P=5"),
    (9, "cls_orca_P9.json", "ORCA-RVO2", "**classical reference**: ORCA (RVO2) at P=9"),
    (16, "cls_orca_P16.json", "ORCA-RVO2", "**classical reference**: ORCA (RVO2) at P=16"),
    (20, "cls_orca_P20.json", "ORCA-RVO2", "**classical reference**: ORCA (RVO2) at P=20"),
    (5, "dens_P5_base.json", "dens-P5", "density curriculum, no filter"),
    (5, "dens_P5_t05p.json", "dens-P5+layer", "density curriculum + layer"),
    (9, "dens_P9_base.json", "dens-P9", "density curriculum, no filter"),
    (9, "dens_P9_t05p.json", "dens-P9+layer", "density curriculum + layer"),
    (12, "dens_P12_base.json", "dens-P12-base", "density curriculum @ unseen P=12"),
    (12, "dens_P12_layer.json", "dens-P12-layer", "density curriculum + layer @ P=12"),
    (16, "dens_P16_base.json", "dens-P16", "density curriculum, no filter"),
    (16, "dens_P16_t05p.json", "dens-P16+layer", "density curriculum + layer"),
    (20, "dens_P20_base.json", "dens-P20-base", "density curriculum @ unseen P=20"),
    (20, "dens_P20_layer.json", "dens-P20-layer", "density curriculum + layer @ P=20"),
    (5, "dlong_P5_base.json", "dlong-P5-base", "**long curriculum** (P=16 gets 1200 it)"),
    (9, "dlong_P9_base.json", "dlong-P9-base", "**long curriculum**"),
    (16, "dlong_P16_base.json", "dlong-P16-base", "**long curriculum**"),
    (5, "dlong_P5_t05p.json", "dlong-P5-t05p", "long curriculum + layer"),
    (9, "dlong_P9_t05p.json", "dlong-P9-t05p", "long curriculum + layer"),
    (16, "dlong_P16_t05p.json", "dlong-P16-t05p", "long curriculum + layer"),
    (5, "wide_P5_base.json", "wide-P5-base", "**wide curriculum** (covers P=20)"),
    (9, "wide_P9_base.json", "wide-P9-base", "**wide curriculum**"),
    (16, "wide_P16_base.json", "wide-P16-base", "**wide curriculum**"),
    (20, "wide_P20_base.json", "wide-P20-base", "**wide curriculum**"),
    (5, "wide_P5_t05p.json", "wide-P5-t05p", "wide curriculum + layer (tau=0.75)"),
    (9, "wide_P9_t05p.json", "wide-P9-t05p", "wide curriculum + layer"),
    (16, "wide_P16_t05p.json", "wide-P16-t05p", "wide curriculum + layer"),
    (20, "wide_P20_t05p.json", "wide-P20-t05p", "wide curriculum + layer"),
    (5, "cyc_P5_base.json", "cyc-P5-base", "**uniform interleave** (5/9/12/16/20)"),
    (9, "cyc_P9_base.json", "cyc-P9-base", "**uniform interleave**"),
    (16, "cyc_P16_base.json", "cyc-P16-base", "**uniform interleave**"),
    (20, "cyc_P20_base.json", "cyc-P20-base", "**uniform interleave**"),
    (5, "wcyc_P5_base.json", "wcyc-P5-base", "difficulty-weighted interleave"),
    (9, "wcyc_P9_base.json", "wcyc-P9-base", "difficulty-weighted interleave"),
    (16, "wcyc_P16_base.json", "wcyc-P16-base", "difficulty-weighted interleave"),
    (20, "wcyc_P20_base.json", "wcyc-P20-base", "difficulty-weighted interleave"),
    (5, "wcyc_P5_t075p.json", "wcyc-P5-t075p", "difficulty-weighted interleave + layer"),
    (9, "wcyc_P9_t075p.json", "wcyc-P9-t075p", "difficulty-weighted interleave + layer"),
    (16, "wcyc_P16_t075p.json", "wcyc-P16-t075p", "difficulty-weighted interleave + layer"),
    (20, "wcyc_P20_t075p.json", "wcyc-P20-t075p", "difficulty-weighted interleave + layer"),
]


def agg(fname, labels):
    p = os.path.join("results/bench", fname)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    rows = [r for r in d["summary"] if r["label"] in labels]
    if not rows:
        return None
    n = len(rows)

    def m(k):
        return sum((r[k][0] if isinstance(r[k], list) else r[k]) for r in rows) / n
    soc = [r["social_succ"]["0.55"] for r in rows]
    soc = sum((x[0] if isinstance(x, list) else x) for x in soc) / n
    return dict(success=100 * m("success"), cvar=m("cvar_raw"), coll_rp=100 * m("coll_rp"),
                coll_rr=100 * m("coll_rr"), min_rp=m("min_rp"), nav=m("nav_s"),
                social=100 * soc, n=n)


def line(name, r, star=False):
    nm = f"**{name}**" if star else name
    return (f"| {nm} | {r['success']:.1f} | {r['cvar']:.2f} | {r['social']:.1f} | "
            f"{r['coll_rp']:.1f} | {r['coll_rr']:.1f} | {r['min_rp']:.3f} | {r['nav']:.2f} |")


def main():
    print("# Route A — affect-aware safety layer (paper tables, English)\n")
    print("All learned arms evaluated at P=9 with held-out environment seeds 1000/1001/1002; "
          "`social@0.55` = every robot reaches its goal AND the minimum robot-pedestrian "
          "distance never drops below 0.55 m.\n")

    print("## Table A.1 — safety-layer variants (P=9, joint QP solver unless noted)\n")
    print("> These rows are **single training seed** (the full variant sweep). The\n"
          "> headline rows with **three training seeds** are in `LAYER_3SEED.md`, and the\n"
          "> shape/orientation controls in `SHAPE_CONTROL.md`. Cite those.\n")
    print("| method | reach success (%) | cvar ↓ | social@0.55 (%) | r-p contact (%) | r-r contact (%) | min r-p (m) | nav (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for f, lab, desc, star in A1:
        r = agg(f, [lab])
        print(line(desc, r, star) if r else f"| {desc} | | | | | | | |")

    print("\n## Table A.2 — density (zero-shot from P=9 unless stated)\n")
    print("| P | method | reach success (%) | cvar ↓ | social@0.55 (%) | r-p contact (%) | r-r contact (%) |")
    print("|---|---|---|---|---|---|---|")
    for P, f, lab, desc in A2:
        r = agg(f, [lab])
        print(f"| {P} | {desc} | {r['success']:.1f} | {r['cvar']:.2f} | {r['social']:.1f} | "
              f"{r['coll_rp']:.1f} | {r['coll_rr']:.1f} |" if r
              else f"| {P} | {desc} | | | | | |")

    print("\n## Table A.3 — training-recipe control (de-confounds the main table)\n")
    print("| policy + filter | reach success (%) | cvar ↓ | social@0.55 (%) | r-p contact (%) |")
    print("|---|---|---|---|---|")
    for f, labels, desc in A3:
        r = agg(f, labels)
        print(f"| {desc} | {r['success']:.1f} | {r['cvar']:.2f} | {r['social']:.1f} | "
              f"{r['coll_rp']:.1f} |" if r else f"| {desc} | | | | |")

    print("\n## Table A.4 — one policy for all densities (density curriculum)\n")
    print("| P | method | reach success (%) | cvar ↓ | social@0.55 (%) | r-p contact (%) | r-r contact (%) |")
    print("|---|---|---|---|---|---|---|")
    for P, f, lab, desc in A4:
        r = agg(f, [lab])
        print(f"| {P} | {desc} | {r['success']:.1f} | {r['cvar']:.2f} | {r['social']:.1f} | "
              f"{r['coll_rp']:.1f} | {r['coll_rr']:.1f} |" if r
              else f"| {P} | {desc} | | | | | |")

    print("\nNotes: the CVaR-MAPPO rows above were trained with `lambda ≡ 0` (slack budget), "
          "i.e. they are a recipe control, not evidence for the CVaR constraint; the "
          "matched-dynamics MAPPO row isolates the acceleration-curriculum difference. "
          "See NIGHT_REPORT sections 46 and 51 for the corresponding caveats.")
    print("\nTable A.4 caveats: (i) the classical references are evaluated AT EACH "
          "DENSITY and must be compared that way -- ORCA at P=5 (0.914 / 13.64 / 74.0%) "
          "is not the P=9 row (0.839 / 19.18 / 51.6%); (ii) \"density-specific\" is not "
          "one baseline: at P=5 it is the P=9 policy transferred DOWN (no P=5-specific "
          "training exists), at P=16 it is a policy fine-tuned at P=16; (iii) the layer "
          "only ties ORCA on the cvar axis at P=5/16/20 and loses the task, social and "
          "contact axes everywhere, so no row here supports a claim of beating the "
          "classical planner -- the clean result is orthogonality (ORCA + layer, "
          "cvar 19.18 -> 8.93 at P=9); (iv) the interleaved curricula now have two "
          "training seeds (see SEED_AGGREGATE.md) and the honest reading is that the "
          "schedule SHAPE matters less than how the fixed budget is split across "
          "densities: the three P=20-covering configurations are within 1.3pp on the "
          "four-density mean, while each density's score rises with the iterations it "
          "received.")


if __name__ == "__main__":
    main()
