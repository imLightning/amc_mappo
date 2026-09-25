"""diag_obs_noise.py -- dose-response of the learned policy to observation noise.

`queue_robustness.sh` found that obs_noise = 0.02 already drives success to 0 for
BOTH arms (every episode times out).  Before reporting "the policy is brittle to
sensor noise" that needs a mechanism, because two benign explanations exist:

  1. the env draws `torch.randn_like(obs)` from the GLOBAL RNG whenever
     obs_noise != 0, so switching noise on also shifts the environment's own
     random stream (tested separately by the 1e-6 cell in the robustness suite);
  2. the noise could simply be large *relative to the signal* in the sparse
     channels (observations are divided by `size`, so 0.02 ~ 0.1 m per step).

This script measures the dose-response in one place, on CPU, with the same
checkpoint: success, mean speed, net progress toward the goal, and how much the
policy's action direction wobbles.  If success collapses while the robot still
moves toward the goal, the failure is not "frozen"; if the action direction goes
random, the policy itself is the failure point.

Usage: python scripts/diag_obs_noise.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from config import load_cfg  # noqa: E402
from envs.social_nav import SocialNavVecEnv  # noqa: E402
from tools._impl.eval_protocol import _make_trainer  # noqa: E402

SIGMAS = [0.0, 1e-6, 0.002, 0.005, 0.01, 0.02, 0.05]
B = 16
RECIPE_DEFAULT = "configs/mappo_ws_curr.yaml"
CKPT_DEFAULT = "runs/ws_p9_mappo_s0/ckpt/seed0/final.pt"


def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigmas", default=",".join(str(s) for s in SIGMAS),
                    help="observation-noise levels to sweep (comma separated)")
    ap.add_argument("--batch", type=int, default=B, help="parallel environments")
    ap.add_argument("--recipe", default="configs/mappo_ws_curr.yaml")
    ap.add_argument("--ckpt", default="runs/ws_p9_mappo_s0/ckpt/seed0/final.pt")
    ap.add_argument("--peds", type=int, default=9)
    ap.add_argument("--out", default="results/bench/diag_obs_noise.json")
    return ap.parse_args(argv)


def run(sigma, seed=1000, recipe=RECIPE_DEFAULT, ckpt=CKPT_DEFAULT, peds=9, batch=B):
    cfg = load_cfg(recipe)
    cfg.env["n_pedestrians"] = float(peds)
    cfg.env["obs_noise"] = float(sigma)
    cfg.env.auto_reset = False
    env = SocialNavVecEnv(cfg, n_parallel_envs=batch, device="cpu")
    env.reset(seed=seed)
    tr = _make_trainer(ckpt, env, dict(cfg.algo), "cpu")
    tr.load(ckpt)
    d0 = (env.robo_pos - env.robo_goal).norm(dim=-1)          # (B,N)
    sp, cos, prog, prev_head = [], [], [], None
    for _ in range(env.horizon):
        act = tr.act_det(env.get_obs()).reshape(batch, env.N, -1)
        env.step(act)
        v = env.robo_vel
        sp.append(v.norm(dim=-1))
        g = env.robo_goal - env.robo_pos
        cos.append(torch.nn.functional.cosine_similarity(
            v, g, dim=-1, eps=1e-6))
    dT = (env.robo_pos - env.robo_goal).norm(dim=-1)
    succ = float((dT < float(cfg.env["goal_tolerance"])).float().mean())
    return dict(sigma=sigma, success=succ,
                mean_speed=float(torch.stack(sp).mean()),
                cos_vel_goal=float(torch.stack(cos).mean()),
                progress=d0.mean().item() - dT.mean().item())


def main(argv=None):
    a = parse_args(argv)
    sigmas = [float(x) for x in str(a.sigmas).split(",") if x != ""]
    out = []
    print(f"{'sigma':>8s} {'success':>8s} {'mean_speed':>11s} {'cos(v,goal)':>12s} {'progress(m)':>12s}")
    for s in sigmas:
        r = run(s, recipe=a.recipe, ckpt=a.ckpt, peds=a.peds, batch=a.batch)
        out.append(r)
        print(f"{r['sigma']:8.5f} {r['success']:8.3f} {r['mean_speed']:11.3f} "
              f"{r['cos_vel_goal']:12.3f} {r['progress']:12.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
