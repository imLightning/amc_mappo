"""train_sarl.py -- DQN training for the SARL port (envs/sarl.py).

Discrete action set: n_speeds x n_head (heading offsets around the goal
direction).  Reward (SARL-style): goal progress + reach bonus - collision
penalty - discomfort (TTC-based).  Parameter sharing across the N robots and no
communication, i.e. a single-agent method applied to every robot.

Usage:
  python scripts/train_sarl.py --seed 0 --episodes 1500 --tag sarl_s0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.nn as nn

from config import load_cfg
from envs.social_nav import SocialNavVecEnv
from envs.sarl import SARLNet, build_state

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="configs/mappo_ws_curr.yaml")
    ap.add_argument("--n_pedestrians", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--n_env", type=int, default=64)
    ap.add_argument("--n_speeds", type=int, default=5)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--head_span", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.1)
    ap.add_argument("--target_every", type=int, default=500)
    ap.add_argument("--train_every", type=int, default=4)
    ap.add_argument("--w_progress", type=float, default=1.0)
    ap.add_argument("--w_reach", type=float, default=10.0)
    ap.add_argument("--w_coll", type=float, default=5.0)
    ap.add_argument("--w_discom", type=float, default=1.0)
    ap.add_argument("--double", type=int, default=0,
                    help="Double DQN: select the greedy action with the ONLINE net "
                         "and evaluate it with the TARGET net.  The port uses vanilla "
                         "DQN (same net for selection and evaluation), and in a "
                         "40-action discrete space with noisy estimates that "
                         "maximisation bias is a classic route to the degenerate "
                         "'run at max speed' attractor we measured.  Default 0 = "
                         "legacy behaviour.")
    ap.add_argument("--use_attention", type=int, default=1)
    ap.add_argument("--tag", default="sarl")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    cfg = load_cfg(os.path.join(PROJ, a.recipe))
    cfg.env["n_pedestrians"] = a.n_pedestrians
    cfg.env.auto_reset = False          # we drive episode boundaries ourselves
    env = SocialNavVecEnv(cfg, n_parallel_envs=a.n_env, device=a.device,
                          base_seed=a.seed)
    dev = env.device
    n_act = a.n_speeds * a.n_head
    net = SARLNet(n_act, hidden=a.hidden, use_attention=bool(a.use_attention)).to(dev)
    tgt = SARLNet(n_act, hidden=a.hidden, use_attention=bool(a.use_attention)).to(dev)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    vs = torch.linspace(0.0, 1.0, a.n_speeds, device=dev) * float(env.max_sp)
    hs = (torch.linspace(-1.0, 1.0, a.n_head, device=dev) * a.head_span * np.pi)

    buf = {k: [] for k in ("s", "h", "a", "r", "s2", "h2", "d")}
    n_samp = 0

    def act_batch(eps):
        s, h, m = build_state(env)
        q = net(s, h, m)
        if eps > 0:
            rand = torch.rand(env.B, env.N, device=dev) < eps
            idx = q.argmax(dim=-1)
            idx = torch.where(rand, torch.randint(0, n_act, idx.shape, device=dev), idx)
        else:
            idx = q.argmax(dim=-1)
        iv, ih = idx // a.n_head, idx % a.n_head
        g = env.robo_goal - env.robo_pos
        gth = torch.atan2(g[..., 1], g[..., 0])
        th = gth + hs[ih]
        v = vs[iv]
        return torch.stack([v * torch.cos(th), v * torch.sin(th)], dim=-1), idx, (s, h)

    def step_env(prev_d, prev_close):
        """advance one env step, return reward components"""
        new_d = env._dist_goal()
        rew = a.w_progress * (prev_d - new_d)
        newly = (new_d <= env.goal_tol) & (~env.reached) & (~env.crashed)
        rew = rew + a.w_reach * newly.float()
        if env.P:
            dd = (env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)).norm(dim=-1)
            close = dd.amin(dim=-1)                                  # B,N
            coll = (close < env.coll_rp).float()
            rew = rew - a.w_coll * coll
            # discomfort: closing speed while inside the personal radius
            rel = env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)
            vrel = env.robo_vel.unsqueeze(2) - env.ped_vel.unsqueeze(1)
            closing = (-(rel * vrel).sum(-1) / rel.norm(dim=-1).clamp_min(1e-3))
            inside = (dd < 1.2).float()
            rew = rew - a.w_discom * (closing.clamp_min(0) * inside).amax(dim=-1)
        return rew

    t0 = time.time()
    hist = []
    for ep in range(a.episodes):
        env.reset(seed=a.seed * 100000 + ep)
        eps = a.eps_start + (a.eps_end - a.eps_start) * min(1.0, ep / max(1, 0.6 * a.episodes))
        prev_d = env._dist_goal()
        for t in range(env.horizon):
            obs_act, idx, (s, h) = act_batch(eps)
            env.step(obs_act)
            rew = step_env(prev_d, None)
            s2, h2, m2 = build_state(env)
            done = (env.reached.all(-1) | env.crashed.all(-1)).unsqueeze(-1).expand_as(rew)
            for k, v in (("s", s), ("h", h), ("a", idx), ("r", rew),
                         ("s2", s2), ("h2", h2), ("d", done.float())):
                buf[k].append(v.detach())
            n_samp += rew.numel()
            prev_d = env._dist_goal()
            # ---- DQN update ----
            if n_samp >= a.batch and (n_samp // (env.B * env.N)) % a.train_every == 0:
                def cat(k):
                    return torch.cat(buf[k], dim=0).reshape(-1, *buf[k][0].shape[2:])
                S, H, A, R, S2, H2, D = (cat("s").reshape(-1, buf["s"][0].shape[-1]),
                                         cat("h").reshape(-1, *buf["h"][0].shape[2:]),
                                         cat("a").reshape(-1),
                                         cat("r").reshape(-1),
                                         cat("s2").reshape(-1, buf["s"][0].shape[-1]),
                                         cat("h2").reshape(-1, *buf["h"][0].shape[2:]),
                                         cat("d").reshape(-1))
                n = S.shape[0]
                if n > a.buffer:                     # keep the most recent window
                    for k in buf:
                        buf[k] = buf[k][-a.buffer // (env.B * env.N):]
                    continue
                perm = torch.randperm(n, device=dev)[:a.batch]
                q = net(S[perm], H[perm]) if H.shape[1] == 0 else net(S[perm], H[perm])
                qa = q.gather(1, A[perm].unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if a.double:
                        a_star = net(S2[perm], H2[perm]).argmax(dim=1, keepdim=True)
                        q2 = tgt(S2[perm], H2[perm]).gather(1, a_star).squeeze(1)
                    else:
                        q2 = tgt(S2[perm], H2[perm]).max(dim=1).values
                    y = R[perm] + a.gamma * (1.0 - D[perm]) * q2
                loss = nn.functional.smooth_l1_loss(qa, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                if (n_samp // a.batch) % a.target_every == 0:
                    tgt.load_state_dict(net.state_dict())
        if (ep + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[{a.tag}] ep {ep+1}/{a.episodes} samples {n_samp} eps {eps:.2f} "
                  f"({el/60:.1f} min)", flush=True)
            hist.append(dict(ep=ep + 1, samples=n_samp, eps=eps))

    outdir = os.path.join(PROJ, "runs", a.tag, "ckpt", f"seed{a.seed}")
    os.makedirs(outdir, exist_ok=True)
    torch.save({"net": net.state_dict()}, os.path.join(outdir, "final.pt"))
    json.dump(dict(args=vars(a), hist=hist),
              open(os.path.join(PROJ, "runs", a.tag, f"seed{a.seed}.json"), "w"), indent=1)
    print("done ->", outdir)


if __name__ == "__main__":
    main()
