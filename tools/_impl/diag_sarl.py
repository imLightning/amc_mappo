"""diag_sarl.py -- why does the SARL port not converge?

Four SARL/LSTM runs all failed (success 0% / 13% / 0% / 0.3%).  Before spending
more GPU time on hyper-parameters this measures the FAILURE MODE of an existing
checkpoint, because the fixes are different:

  * if the argmax action is almost always the ZERO speed  -> the Q-net collapsed
    to "stand still" (a classic DQN local optimum when the step penalty/contact
    cost outweighs progress);
  * if speeds are used but the average is far below what the arena needs
    (10x10 m, T=10 s) -> the value function never learned that progress pays;
  * if headings are near-random relative to the goal -> the state encoder is not
    giving the policy the goal direction (or it is not normalised).

Usage: python scripts/diag_sarl.py --ckpt runs/sarl_lstm_s0/ckpt/seed0/final.pt \
           --attention 0 [--seeds 1000,1001]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from config import load_cfg  # noqa: E402
from envs.sarl import SARLPolicy, build_state  # noqa: E402
from envs.social_nav import SocialNavVecEnv  # noqa: E402

B = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--attention", type=int, default=0)
    ap.add_argument("--recipe", default="configs/mappo_ws_curr.yaml")
    ap.add_argument("--peds", type=int, default=9)
    ap.add_argument("--seeds", default="1000,1001")
    ap.add_argument("--head-span", dest="head_span", type=float, default=1.0,
                    help="heading span the CHECKPOINT WAS TRAINED WITH.  This only "
                         "affects the printed bin labels -- reporting the labels of "
                         "the wrong span mis-states the action histogram (this "
                         "happened once: a run trained with 0.3 was reported with "
                         "1.0 labels).")
    ap.add_argument("--out", default=None,
                    help="also write the diagnosis as JSON (traceability)")
    a = ap.parse_args()

    pol = SARLPolicy(ckpt=a.ckpt, use_attention=bool(a.attention),
                     head_span=a.head_span, device="cpu")
    rec = []
    print(f"checkpoint {a.ckpt}  actions {pol.n_speeds}x{pol.n_head} = "
          f"{pol.n_speeds * pol.n_head}")
    for seed in [int(x) for x in a.seeds.split(",")]:
        cfg = load_cfg(a.recipe)
        cfg.env["n_pedestrians"] = float(a.peds)
        cfg.env.auto_reset = False
        env = SocialNavVecEnv(cfg, n_parallel_envs=B, device="cpu")
        env.reset(seed=seed)
        vs = torch.linspace(0.0, 1.0, pol.n_speeds) * float(env.max_sp)
        d0 = (env.robo_pos - env.robo_goal).norm(dim=-1).mean().item()
        cnt = torch.zeros(pol.n_speeds, dtype=torch.long)
        hcnt = torch.zeros(pol.n_head, dtype=torch.long)
        sp, cos = [], []
        for _ in range(env.horizon):
            s, h, m = build_state(env)
            with torch.no_grad():
                q = pol.net(s, h, m)                       # (B,N,A)
            idx = q.argmax(-1)
            iv, ih = idx // pol.n_head, idx % pol.n_head
            cnt += torch.bincount(iv.reshape(-1), minlength=pol.n_speeds)
            hcnt += torch.bincount(ih.reshape(-1), minlength=pol.n_head)
            act = pol.act(env)
            env.step(act)
            v = env.robo_vel
            sp.append(v.norm(dim=-1))
            g = env.robo_goal - env.robo_pos
            cos.append(torch.nn.functional.cosine_similarity(v, g, dim=-1, eps=1e-6))
        dT = (env.robo_pos - env.robo_goal).norm(dim=-1)
        succ = float((dT < float(cfg.env["goal_tolerance"])).float().mean())
        tot = cnt.sum().item()
        print(f"\n--- seed {seed}: success {succ:.3f}  d_goal {d0:.2f} -> "
              f"{dT.mean().item():.2f} m  mean speed {torch.stack(sp).mean():.3f} m/s  "
              f"cos(v,goal) {torch.stack(cos).mean():+.3f}")
        print("    speed histogram over chosen actions: " + "  ".join(
            f"{float(vs[i]):.2f}m/s {100*cnt[i]/tot:5.1f}%" for i in range(pol.n_speeds)))
        print("    heading-offset histogram (deg from goal dir): " + "  ".join(
            f"{int(round(((-1 + 2*i/(pol.n_head-1)) * pol.head_span * 180)))}:"
            f"{100*hcnt[i]/tot:4.1f}%" for i in range(pol.n_head)))
        rec.append(dict(seed=seed, ckpt=a.ckpt, success=succ, d_goal_0=d0,
                        d_goal_T=float(dT.mean()), mean_speed=float(torch.stack(sp).mean()),
                        cos_vel_goal=float(torch.stack(cos).mean()),
                        speed_hist=[float(c) / tot for c in cnt],
                        head_hist=[float(c) / tot for c in hcnt],
                        speeds=[float(v) for v in vs]))
    if a.out:
        import json
        json.dump(rec, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
