"""make_tau_figure.py -- route A main figure: the early-yield margin tau traces a
controllable safety/task curve.

Panel 1 : reach success and social@0.55 vs tau (the two task-side axes)
Panel 2 : cvar (affect cost) vs tau, with the analytic detour and beeline as
          horizontal reference lines, and the peer-barrier variant marked

Data: results/bench/ttc_t*.json (tau 0.5/1.0/1.5), tau_t075/t125.json,
joint_jnt_a05.json (tau=0), ttc_t10p.json / tau_t075p / tau_t125p (with peer).

Usage: python scripts/make_tau_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Data source priority: the THREE-POLICY-SEED files produced by
# ops/queue/queue_layer_3seed.sh (`layer3s_<tag>-s{k}.json`), which give a
# training-seed error band; fall back to the historical single-seed files so the
# figure still renders if the 3-seed sweep has not been run.
# (tau, 3-seed tag, fallback file, fallback label)
NO_PEER = [(0.0, "jt0.0", "joint_jnt_a05.json", None),
           (0.5, "jt0.5", "ttc_t05.json", None),
           (0.75, "jt0.75", "tau_t075.json", None),
           (1.0, "jt1.0", "ttc_t10.json", None),
           (1.25, "jt1.25", "tau_t125.json", None),
           (1.5, "jt1.5", "ttc_t15.json", None)]
PEER = [(0.75, "jt0.75p", "tau_t075p.json", None),
        (1.0, "jt1.0p", "ttc_t10p.json", None),
        (1.25, None, "tau_t125p.json", None)]
PRED = [(5.0, None, "pred_k05.json", None),
        (10.0, None, "pred_k10.json", None),
        (20.0, None, "pred_k20.json", None)]


def load(fname, label_match=None):
    p = os.path.join("results/bench", fname)
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    rows = {s["label"]: s for s in d["summary"]}
    ref = rows.get("scripted-detour")
    arm = None
    for lab, s in rows.items():
        if lab.startswith("scripted"):
            continue
        if label_match is None or label_match in lab:
            arm = s
    return arm, ref


def series3(tag):
    """Mean and range over the three POLICY seeds (each a 3-env-seed mean)."""
    got = []
    for sd in (0, 1, 2):
        p = os.path.join("results/bench", f"layer3s_{tag}-s{sd}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["summary"]:
            if r["label"] == f"{tag}-s{sd}":
                soc = r["social_succ"]["0.55"]
                got.append(dict(success=100 * r["success"][0], cvar=r["cvar_raw"][0],
                                social=100 * (soc[0] if isinstance(soc, list) else soc)))
    if not got:
        return None
    out = {}
    for k in ("success", "cvar", "social"):
        v = [g[k] for g in got]
        out[k] = (sum(v) / len(v), (max(v) - min(v)) / 2 if len(v) > 1 else 0.0)
    out["n"] = len(got)
    return out


def series(spec):
    """-> tau list, and per-metric (mean, half-range) lists."""
    xs, succ, cvar, soc = [], [], [], []
    for tau, tag, f, lm in spec:
        a = series3(tag) if tag else None
        if a is None:
            arm, _ = load(f, lm)
            if arm is None:
                continue
            a = dict(success=(100 * arm["success"][0], 0.0),
                     cvar=(arm["cvar_raw"][0], 0.0),
                     social=(100 * arm["social_succ"]["0.55"][0], 0.0))
        xs.append(tau)
        succ.append(a["success"]); cvar.append(a["cvar"]); soc.append(a["social"])
    return xs, succ, cvar, soc


def main():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for spec, style, name in ((NO_PEER, "-o", "no peer"),
                              (PEER, "--s", "+ reciprocal peer")):
        xs, succ, cvar, soc = series(spec)
        m, e = [v[0] for v in succ], [v[1] for v in succ]
        ax[0].plot(xs, m, style, label=f"reach success ({name}")
        ax[0].fill_between(xs, [a - b for a, b in zip(m, e)],
                           [a + b for a, b in zip(m, e)], alpha=0.18)
        m, e = [v[0] for v in soc], [v[1] for v in soc]
        ax[0].plot(xs, m, style, alpha=0.55, label=f"social@0.55 ({name}")
        ax[0].fill_between(xs, [a - b for a, b in zip(m, e)],
                           [a + b for a, b in zip(m, e)], alpha=0.12)
        m, e = [v[0] for v in cvar], [v[1] for v in cvar]
        ax[1].plot(xs, m, style, label=name)
        ax[1].fill_between(xs, [a - b for a, b in zip(m, e)],
                           [a + b for a, b in zip(m, e)], alpha=0.18)
    # predictive margin for reference (kappa rescaled onto the tau axis is not
    # meaningful, so plot it as markers with its own annotation)
    xs, succ, cvar, soc = series(PRED)
    ax[0].plot([], [], " ", label="")
    ax[1].scatter([1.0] * len(cvar), [v[0] for v in cvar], marker="x", s=45,
                  label="predictive affect margin (kappa 5/10/20)")

    _, ref = load("ttc_t10.json")
    if ref is not None:
        ax[1].axhline(ref["cvar_raw"][0], ls=":", color="tab:red",
                      label=f"analytic detour cvar={ref['cvar_raw'][0]:.1f}")
    beeline, _ = load("ttc_t10.json", "__none__")
    base = json.load(open("results/bench/scene_base.json"))
    b = next(s for s in base["summary"] if s["label"] == "scripted-beeline")
    ax[1].axhline(b["cvar_raw"][0], ls=":", color="tab:gray",
                  label=f"beeline cvar={b['cvar_raw'][0]:.1f}")

    ax[0].set_xlabel(r"early-yield tau $\tau$ (s)"); ax[0].set_ylabel("success rate (%)")
    ax[0].set_title("Task / social success vs early-yield margin")
    ax[1].set_xlabel(r"early-yield tau $\tau$ (s)"); ax[1].set_ylabel("cvar_raw ↓")
    ax[1].set_title("Affect cost vs early-yield margin (lower is better)")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Route A: anisotropic affect barrier + joint QP + early yielding "
                 "(P=9; band = range over 3 training seeds)")
    fig.tight_layout()
    out = "results/bench/tau_curve.png"
    fig.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main()
