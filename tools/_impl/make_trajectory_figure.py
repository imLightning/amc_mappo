"""make_trajectory_figure.py -- qualitative figure for route A.

Design (deliberately NOT a hand-picked single episode): single episodes are not
representative, and the old `traj_compare.py` docstring already recorded that the
min-distance ordering flips across scenarios.  So the figure has three panels:

  (a) illustrative trajectories on ONE scenario, chosen by a STATED RULE
      (the seed with the largest min-distance improvement among a fixed set of
      held-out seeds), with the rule printed in the caption so the reader knows
      it is illustrative, not cherry-picked by eye;
  (b) the time series of minimum robot-pedestrian distance for that same episode,
      no filter vs the recommended layer;
  (c) the PAIRED per-scenario distribution over all seeds in the set -- the
      honest summary.

Usage: python scripts/make_trajectory_figure.py [--seeds 2000:2016]
Writes: results/bench/trajectory_qualitative.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from config import load_cfg  # noqa: E402
from envs.cbf import CBFFilter  # noqa: E402
from envs.social_nav import SocialNavVecEnv  # noqa: E402
from tools._impl.eval_protocol import _make_trainer  # noqa: E402

RECIPE = "configs/mappo_ws_curr.yaml"
CKPT = "runs/ws_p9_mappo_s0/ckpt/seed0/final.pt"
LAYER = dict(d_min=0.6, alpha=0.5, mode="lex", emotion=True, ttc_gain=0.75,
             peer=True, d_peer=0.70, peer_front_only=False)
P = 9


def rollout(seed, use_layer):
    """one scenario, one environment; returns trajectories + min-distance series."""
    cfg = load_cfg(RECIPE)
    cfg.env["n_pedestrians"] = float(P)
    cfg.env.auto_reset = False
    env = SocialNavVecEnv(cfg, n_parallel_envs=1, device="cpu")
    env.reset(seed=seed)
    tr = _make_trainer(CKPT, env, dict(cfg.algo), "cpu")
    tr.load(CKPT)
    filt = CBFFilter(vmax=float(env.max_sp), amax=float(getattr(env, "max_accel", 0.0) or 0.0),
                     dt=float(env.action_dt), passes=3, **LAYER) if use_layer else None
    rp = [env.robo_pos[0].clone()]
    pp = [env.ped_pos[0].clone()]
    md = []
    for _ in range(env.horizon):
        act = tr.act_det(env.get_obs()).reshape(1, env.N, -1)
        if filt is not None:
            act = filt.project(env, act)
        env.step(act)
        rp.append(env.robo_pos[0].clone())
        pp.append(env.ped_pos[0].clone())
        rel = (env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)).norm(dim=-1)
        md.append(float(rel.min()))
    md = torch.tensor(md)
    # NOTE: the per-episode MINIMUM saturates at the contact distance whenever a
    # contact happens (`contact_stop` freezes the episode), so min_rp is a nearly
    # binary statistic across scenarios.  The informative per-episode summaries
    # are how MUCH of the episode is spent inside 0.55 m and how contacts there
    # were -- those are what panel (c) plots.
    return dict(rp=torch.stack(rp).numpy(), pp=torch.stack(pp).numpy(),
                md=md.numpy(), intrude=float((md < 0.55).float().mean()),
                mean_md=float(md.mean()), contact=int(bool((md < 0.55).any())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2000:2016")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split(":"))
    seeds = list(range(lo, hi))

    res = {}
    for s in seeds:
        res[s] = (rollout(s, False), rollout(s, True))
        print(f"seed {s}: min_rp base {min(res[s][0]['md']):.3f} -> layer "
              f"{min(res[s][1]['md']):.3f}", flush=True)

    gains = {s: res[s][0]["intrude"] - res[s][1]["intrude"] for s in seeds}
    best = max(gains, key=lambda s: gains[s])
    print(f"illustrative seed (largest drop in intruding-time share, stated rule): "
          f"{best} ({gains[best]:+.3f})")

    fig = plt.figure(figsize=(17.5, 4.6))
    ax0, ax1, ax2, ax3 = fig.subplots(1, 4)
    for ax, use_layer, title in ((ax0, False, "no filter"), (ax1, True, "recommended layer")):
        base, lay = res[best]
        r = (lay if use_layer else base)
        for n in range(r["pp"].shape[1]):
            ax.plot(r["pp"][:, n, 0], r["pp"][:, n, 1], lw=0.7, alpha=0.45,
                    color="tab:orange")
            ax.scatter(r["pp"][0, n, 0], r["pp"][0, n, 1], s=6, color="tab:orange")
        for n in range(r["rp"].shape[1]):
            ax.plot(r["rp"][:, n, 0], r["rp"][:, n, 1], lw=2.0, color="tab:blue")
            ax.scatter(*r["rp"][-1, n], marker="*", s=60, color="tab:green")
        ax.set_title(f"{title}\nillustrative episode (seed {best}); "
                     f"intruding {100*r['intrude']:.1f}% of steps", fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    b, l = res[best]
    ax1b = ax2
    ax1b.plot([i * 0.05 for i in range(len(b["md"]))], b["md"], color="tab:gray",
              label="no filter")
    ax1b.plot([i * 0.05 for i in range(len(l["md"]))], l["md"], color="tab:blue",
              label="recommended layer")
    ax1b.axhline(0.55, ls=":", color="tab:red", label="contact threshold 0.55 m")
    ax1b.set_xlabel("time (s)"); ax1b.set_ylabel("min robot–pedestrian distance (m)")
    ax1b.set_title("same episode, distance over time", fontsize=9)
    ax1b.grid(alpha=0.3); ax1b.legend(fontsize=8)

    # panel (d): the honest paired summary -- winner cannot be judged from (a)/(b)
    bi = [res[s][0]["intrude"] for s in seeds]
    li = [res[s][1]["intrude"] for s in seeds]
    mx = max(max(bi), max(li)) * 1.05 + 1e-6
    ax3.plot([0, mx], [0, mx], ls=":", color="tab:red", label="equal")
    ax3.scatter(bi, li, s=28, color="tab:blue", zorder=3)
    ax3.set_xlabel("no filter: share of steps inside 0.55 m")
    ax3.set_ylabel("recommended layer: same")
    ax3.set_title(f"paired over {len(seeds)} scenarios\n"
                  f"layer lower on {sum(1 for b, l in zip(bi, li) if l < b - 1e-9)}/{len(seeds)}, "
                  f"mean drop {100*sum(gains.values())/len(gains):.1f} pp", fontsize=9)
    ax3.grid(alpha=0.3); ax3.legend(fontsize=8)

    fig.suptitle("Route A: affect-aware safety layer, qualitative behaviour "
                 f"(P={P}; (a)(b) = one illustrative seed chosen by the stated rule "
                 "'largest drop in intruding-time share', (d) = the honest paired summary)",
                 fontsize=10)
    fig.tight_layout()
    out = "results/bench/trajectory_qualitative.png"
    fig.savefig(out, dpi=160)
    print("saved", out)

    # panel (c): paired per-scenario intruding-time share (the honest summary)
    ax3 = fig.add_subplot(1, 4, 4) if False else None
    import json
    json.dump({str(s): dict(base_intrude=res[s][0]["intrude"],
                           layer_intrude=res[s][1]["intrude"],
                           base_contact=res[s][0]["contact"],
                           layer_contact=res[s][1]["contact"],
                           base_min=float(min(res[s][0]["md"])),
                           layer_min=float(min(res[s][1]["md"])))
               for s in seeds} | {"illustrative_seed": best, "rule":
                                  "max drop in intruding-time share over the seed set"},
              open("results/bench/trajectory_qualitative.json", "w"), indent=1)
    n_better = sum(1 for s in seeds if gains[s] > 1e-9)
    print(f"paired summary: layer intrudes less on {n_better}/{len(seeds)} scenarios; "
          f"mean drop {sum(gains.values())/len(gains):+.4f} of episode time; "
          f"contacts {sum(res[s][0]['contact'] for s in seeds)} -> "
          f"{sum(res[s][1]['contact'] for s in seeds)}")


if __name__ == "__main__":
    main()
