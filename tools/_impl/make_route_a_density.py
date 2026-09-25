"""make_route_a_density.py -- density panel for route A.

Plots success and social@0.55 against pedestrian count P for
  * the un-filtered learned policy (zero-shot),
  * the affect-aware safety layer (tau=1.0, joint QP),
  * the layer + reciprocal peer barrier,
  * the scripted envelope and ORCA as references,
reading whatever is available on disk (missing points are simply skipped, so the
figure can be regenerated as the queue fills in).

Usage: python scripts/make_route_a_density.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def get(fname, label):
    p = os.path.join("results/bench", fname)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    for s in d.get("summary", []):
        if s["label"] == label:
            def v(k):
                x = s[k]
                return x[0] if isinstance(x, list) else x
            soc = s["social_succ"]["0.55"]
            soc = soc[0] if isinstance(soc, list) else soc
            return dict(success=100 * v("success"), cvar=v("cvar_raw"),
                        social=100 * soc,
                        coll_rp=100 * v("coll_rp"),
                        coll_rr=100 * v("coll_rr"))
    return None


# the zero-shot density files hold 3 training seeds under the MAPPO-s*/AMC-s*
# labels; the figure needs ONE number per density, so an aggregate label is used
# where available and the seed-0 label otherwise (documented in the caption).
LEARNED = {9: ("scene_base.json", "MAPPO-P9"),
           5: ("learned_P5_zeroshot.json", "MAPPO-s0"),
           16: ("learned_P16_zeroshot.json", "MAPPO-s0")}
LAYER = {9: ("ttc_t10.json", "t10"),
         5: ("final_P5_t10.json", "mappo-joint-tau1.0-P5"),
         16: ("final_P16_t10.json", "mappo-joint-tau1.0-P16")}
PEER = {9: ("ttc_t10p.json", "t10p"),
        5: None, 16: None}
SCRATCH = {16: ("p16_base.json", "MAPPO-P16")}
SCRATCH_LAYER = {16: ("p16tau_t025.json", "t025")}
# P3: the density-CURRICULUM policy (P=5 -> 11 -> 16 during training), evaluated
# at all three densities, with and without the affect-aware layer.
DENS_CURR = {5: ("dens_P5_base.json", "dens-P5"),
             9: ("dens_P9_base.json", "dens-P9"),
             16: ("dens_P16_base.json", "dens-P16")}
DENS_CURR_LAYER = {5: ("dens_P5_t05p.json", "dens-P5+layer"),
                   9: ("dens_P9_t05p.json", "dens-P9+layer"),
                   16: ("dens_P16_t05p.json", "dens-P16+layer")}
# P3: the RECOMMENDED configuration (difficulty-weighted budget) and the wide
# curriculum that covers P=20, with and without the safety layer.
WEIGHTED = {5: ("wtd_P5_base.json", "wtd-P5-base"),
            9: ("wtd_P9_base.json", "wtd-P9-base"),
            16: ("wtd_P16_base.json", "wtd-P16-base")}
WEIGHTED_LAYER = {5: ("wtd_P5_t075p.json", "wtd-P5-t075p"),
                  9: ("wtd_P9_t075p.json", "wtd-P9-t075p"),
                  16: ("wtd_P16_t075p.json", "wtd-P16-t075p")}
WIDE = {5: ("wide_P5_base.json", "wide-P5-base"),
        9: ("wide_P9_base.json", "wide-P9-base"),
        16: ("wide_P16_base.json", "wide-P16-base"),
        20: ("wide_P20_base.json", "wide-P20-base")}
CYC = {5: ("cyc_P5_base.json", "cyc-P5-base"),
       9: ("cyc_P9_base.json", "cyc-P9-base"),
       16: ("cyc_P16_base.json", "cyc-P16-base"),
       20: ("cyc_P20_base.json", "cyc-P20-base")}
CYC_LAYER = {5: ("cyc_P5_t075p.json", "cyc-P5-t075p"),
             9: ("cyc_P9_t075p.json", "cyc-P9-t075p"),
             16: ("cyc_P16_t075p.json", "cyc-P16-t075p"),
             20: ("cyc_P20_t075p.json", "cyc-P20-t075p")}
WIDE_LAYER = {5: ("wide_P5_t05p.json", "wide-P5-t05p"),
              9: ("wide_P9_t05p.json", "wide-P9-t05p"),
              16: ("wide_P16_t05p.json", "wide-P16-t05p"),
              20: ("wide_P20_t05p.json", "wide-P20-t05p")}
REF = {"beeline": ("scene_base.json", "scripted-beeline"),
       "detour": ("scene_base.json", "scripted-detour"),
       "ORCA": ("scene_base_orca.json", "ORCA-RVO2")}

# Per-density classical references.  Plotting these as CURVES (rather than the
# P=9 horizontal guides above) is the honest way to show density trends: ORCA at
# P=5 is 0.914/74.0% while at P=20 it is 0.563/21.4%, and a P=9 guide line hides
# exactly the comparison a reader wants to make.
ORCA_DENS = {P: (f"cls_orca_P{P}.json", "ORCA-RVO2") for P in (5, 9, 16, 20)}
DETOUR_DENS = {P: (f"cls_orca_P{P}.json", "scripted-detour")
               for P in (5, 9, 16, 20)}


def series(d):
    xs, s, soc = [], [], []
    for P, spec in sorted(d.items()):
        if spec is None:
            continue
        r = get(*spec)
        if r is None:
            continue
        xs.append(P); s.append(r["success"]); soc.append(r["social"])
    return xs, s, soc


def main():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d, style, name in ((LEARNED, "-o", "learned policy (no filter)"),
                           (LAYER, "-s", "+ affect-aware layer (tau=1.0)"),
                           (PEER, "--^", "+ layer + reciprocal peer"),
                           (SCRATCH, ":d", "trained at P (no filter)"),
                           (SCRATCH_LAYER, ":v", "trained at P + layer + peer"),
                           (DENS_CURR, "-P", "density curriculum 800 it (no filter)"),
                           (DENS_CURR_LAYER, "-*", "density curriculum 800 it + layer"),
                           (WEIGHTED, "->", "weighted 300/400/900 (recommended)"),
                           (WEIGHTED_LAYER, "-<", "weighted + layer (recommended)"),
                           (WIDE, ":+", "wide curriculum (covers P=20)"),
                           (WIDE_LAYER, ":x", "wide curriculum + layer"),
                           (CYC, "-.", "interleaved densities"),
                           (CYC_LAYER, ":*", "interleaved + layer"),
                           (ORCA_DENS, "-", "ORCA  (classical, per density)"),
                           (DETOUR_DENS, "--", "analytic detour (per density)")):
        xs, s, soc = series(d)
        if xs:
            ax[0].plot(xs, s, style, label=name)
            ax[1].plot(xs, soc, style, label=name)
    for name, spec in REF.items():
        r = get(*spec)
        if r is None:
            continue
        if name in ("ORCA", "detour"):
            # these two now have real PER-DENSITY curves (ORCA_DENS/DETOUR_DENS
            # above); a P=9 horizontal guide next to them only invites the
            # wrong-density comparison this project already made once.
            continue
        p9 = os.path.join("results/bench", spec[0])
        # reference curves are only available at the densities evaluated there;
        # draw them as horizontal guides at P=9 for orientation
        ax[0].axhline(r["success"], ls=":", lw=1, alpha=0.6,
                      label=f"{name} (P=9 ref {r['success']:.0f}%)")
        ax[1].axhline(r["social"], ls=":", lw=1, alpha=0.6,
                      label=f"{name} (P=9 ref {r['social']:.0f}%)")
    for a, t in zip(ax, ("reach success (%)", "social@0.55 (%)")):
        a.set_xlabel("pedestrians P"); a.set_ylabel(t); a.grid(alpha=0.3)
        a.set_xticks([5, 9, 12, 16, 20])
    ax[0].set_title("Route A: task success vs density")
    ax[1].set_title("Route A: social success vs density")
    fig.suptitle("Affect-aware safety layer across densities "
                 "(P=9 training unless marked 'trained at P')")
    # one shared legend under both panels: 14 series will not fit inside an axes
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.subplots_adjust(bottom=0.30, top=0.86)
    out = "results/bench/route_a_density.png"
    fig.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main()
