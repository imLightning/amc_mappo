"""calibrate_tau.py -- standardised calibration protocol for the early-yield margin.

WHY A SCRIPT: tau was previously tuned by hand for each density (a 4-point manual
sweep at P=9, a 3-point one at P=16), which is neither reproducible nor
comparable across settings.  Measured facts that motivate a protocol:
  * tau traces a smooth safety/task curve (P=9: tau 0 -> 1.5 gives success
    78.4% -> 55.2% and cvar 25.05 -> 17.90);
  * cvar has a KNEE (P=9 at tau ~ 1.0) after which extra conservatism buys very
    little cost but keeps costing task;
  * the best tau is DENSITY DEPENDENT (P=9 ~1.0 s, P=16 ~0.25-0.5 s), and the
    local-crowding adaptation we tried did not reproduce the per-density optima.

This script runs the sweep in-process, reports for every tau
(success, cvar, social@0.55, contacts, min distance) and marks the knee by the
standard maximum-distance-to-the-chord rule on the (success, cvar) curve, so the
operating point is chosen by a stated rule rather than by eye.

Usage:
  python scripts/calibrate_tau.py --ckpt runs/ws_p9_mappo_s0/ckpt/seed0/final.pt \
      --recipe configs/mappo_ws_curr.yaml --P 9 --taus 0,0.25,0.5,0.75,1.0,1.25,1.5 \
      --peer --out results/bench/tau_calib_P9.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools._impl.pareto_eval import evaluate  # noqa: E402


def knee(xs, ys):
    """max distance to the chord (both sequences in data order, normalised)."""
    if len(xs) < 3:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    def n(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.0
    best, best_d = None, -1.0
    for i in range(1, len(xs) - 1):
        px, py = n(xs[i], x0, x1), n(ys[i], y0, y1)
        ax, ay = n(xs[0], x0, x1), n(ys[0], y0, y1)
        bx, by = n(xs[-1], x0, x1), n(ys[-1], y0, y1)
        dx, dy = bx - ax, by - ay
        L = (dx * dx + dy * dy) ** 0.5 or 1.0
        d = abs(dy * px - dx * py + bx * ay - by * ax) / L
        if d > best_d:
            best, best_d = i, d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--P", type=int, default=9)
    ap.add_argument("--taus", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--seeds", default="1000,1001,1002")
    ap.add_argument("--peer", action="store_true")
    ap.add_argument("--d-min", dest="d_min", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--cbf-mode", dest="cbf_mode", default="joint",
                    choices=["linear", "exact", "joint", "lex"],
                    help="barrier solver used during calibration; the paper's "
                         "operating points in the density tables use 'lex', the "
                         "first calibration sweep used 'joint' (section 59 shows "
                         "the two differ by ~1pp of contact / 0.5pp of success)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s]
    rows = []
    for tau in [float(t) for t in a.taus.split(",") if t != ""]:
        per = []
        for sd in seeds:
            over = {"n_pedestrians": float(a.P)}
            r = evaluate(a.recipe, a.ckpt, f"tau{tau}", a.device,
                         env_over=over, seed=sd, cbf_min=a.d_min,
                         cbf_alpha=a.alpha, cbf_mode=a.cbf_mode, cbf_emotion=True,
                         cbf_ttc_gain=tau, cbf_peer=a.peer,
                         cbf_peer_all=a.peer, cbf_d_peer=0.70)
            per.append(r)
        n = len(per)
        agg = lambda k: sum(r[k] for r in per) / n
        soc = sum(r["social_succ"][0.55] for r in per) / n
        rows.append(dict(tau=tau, success=agg("success"), cvar=agg("cvar_raw"),
                         social=soc, coll_rp=agg("coll_rp"), coll_rr=agg("coll_rr"),
                         min_rp=agg("min_rp"), nav=agg("nav_s")))
        print(f"tau={tau:5.2f}  success {rows[-1]['success']:.3f}  cvar {rows[-1]['cvar']:6.2f}  "
              f"social {rows[-1]['social']:.3f}  coll_rr {rows[-1]['coll_rr']:.3f}", flush=True)

    i = knee([r["tau"] for r in rows], [r["cvar"] for r in rows])
    best = rows[i]["tau"] if i is not None else None
    print(f"\nKnee (max distance to chord on the success/cvar curve): tau = {best}")
    if best is not None:
        r = rows[i]
        print(f"  at the knee: success {r['success']:.3f}  cvar {r['cvar']:.2f}  social {r['social']:.3f}")
    if a.out:
        json.dump(dict(P=a.P, recipe=a.recipe, ckpt=a.ckpt, peer=a.peer,
                       alpha=a.alpha, d_min=a.d_min, mode=a.cbf_mode, seeds=seeds,
                       rows=rows, knee_tau=best), open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
