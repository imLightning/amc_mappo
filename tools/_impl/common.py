"""common.py -- shared building blocks for train / eval / plot.

Contains:  make_cfg (yaml or python override), env constructor wrapper,
deterministic evaluator (independent of internal auto-reset), and trajectory
recording used for .gif animation generation.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from config import load_cfg, default
from envs.social_nav import SocialNavVecEnv
from envs.metrics import Outcome


def load_recipe_cfg(yaml_path=None, overrides=None):
    return load_cfg(yaml_path, overrides)


def make_env(cfg, seed=0, device=None, n=None):
    device = device or cfg.vec.device
    if n is None and hasattr(cfg.vec, "n_parallel_envs"):
        n = cfg.vec.n_parallel_envs
    return SocialNavVecEnv(cfg, n_parallel_envs=n, device=device,
                           base_seed=seed)


@torch.no_grad()
def evaluate_deterministic(trainer, env, total_steps=6000,
                           max_len=240, seed=1):
    """run closed-loop deterministic evaluation on freshly seeded env.

    Returns (Outcome.summary dict, avg_reward).  episode bounds parsed off
    env 'done'/'trunc' flags while internal auto-reset keeps rollouts moving.
    """
    import torch as t
    # work on one vector-size small to simplify
    b = env.B
    alive = np.ones(b, bool)          # currently in an episode
    steps = np.zeros(b)
    succ_flag = None
    outcome = Outcome(n_agents=env.N)
    total_r = 0.0
    n_steps = 0
    obs, state = env.reset()
    act_dim = 2

    coll_rr = t.zeros(b)
    coll_rp = t.zeros(b)
    min_rr = t.full((b,), float("inf"))
    min_rp = t.full((b,), float("inf"))
    # week-3 social-cost bookkeeping (per *step* so independent of auto-reset)
    near_t = env.r + env.ped_r + 0.45          # "too close" distance to human
    soc_steps = 0
    viol_steps = 0.0
    n_socsteps = 0

    while True:
        act = trainer.act_det(obs)
        nobs, nstate, rew, done, trunc, info = env.step(act)
        rew = rew.cpu()
        total_r += float(rew[:, 0].mean())  # same env reward
        n_steps += 1

        # social proximity metric computed fresh (reward-agnostic: same gauge)
        if env.P:
            dp_ = (env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)
                   ).norm(dim=-1)
            mind_ = dp_.amin(dim=(1, 2))            # (B,)
            viol = (mind_ < near_t)
            under_p = (mind_ < (env.r + env.ped_r + 0.15))
            soc_steps += int(under_p.sum())
            viol_steps += int(viol.sum())
            n_socsteps += b

        done_i = done.detach().cpu().numpy().astype(bool)
        trunc_i = trunc.detach().cpu().numpy().astype(bool)
        # `info["succ"]` is snapshotted INSIDE the env's step(), before the
        # auto-reset.  Reading env.reached/env.crashed here instead is wrong
        # whenever the env auto-resets (the recipe default): step() replays the
        # finished envs before returning, which clears those flags, so the test
        # `reached.all() and not crashed.any()` below always came out False.
        # Measured symptom: the in-training EVAL line printed
        # ``success=0.000 collrp=0.851`` for a checkpoint the independent
        # evaluator scores at 0.930 success -- see
        # scripts/diag_eval_consistency.py.
        succ_i = info["succ"].detach().cpu().numpy().astype(bool)

        # capture current per-env accumulated metrics (before they reset)
        cur_rr = info["rr_min"].cpu()
        cur_rp = info["rp_min"].cpu()
        cur_rrc = info["rr_coll"].cpu()
        cur_rpc = info["rp_coll"].cpu()
        newly_finished = done_i  # episode boundary recorded this event

        for idx in np.nonzero(newly_finished)[0]:
            # BUG FIX (2026-09-21): success was `not trunc`, i.e. any episode that
            # ended for a non-timeout reason counted as a success.  With
            # collision_terminal=True a COLLISION ends the episode, so the
            # in-training evaluator reported success ~0.99 for a policy that was
            # in fact colliding constantly (its own training-time succ_ema was
            # 0.026 at the same moment).  Ask the env directly instead --
            # via the PRE-reset snapshot (see the note above).
            success = bool(succ_i[idx]) and not trunc_i[idx]
            length = steps[idx] + 1
            finish_rr = min(min_rr[idx].item(), cur_rr[idx].item())
            mr1 = finish_rr if np.isfinite(finish_rr) else None
            finish_rp = min(min_rp[idx].item(), cur_rp[idx].item())
            mr2 = finish_rp if np.isfinite(finish_rp) else None
            crr = max(coll_rr[idx].item(), float(cur_rrc[idx]))
            crp = max(coll_rp[idx].item(), float(cur_rpc[idx]))
            outcome.add_episode(success, trunc_i[idx], int(length),
                                mr1, mr2, crr, crp)
            # reset per-env counters for new auto-spawned episode
            steps[idx] = 0
            min_rr[idx] = float("inf")
            min_rp[idx] = float("inf")
            coll_rr[idx] = 0.
            coll_rp[idx] = 0.

        # track in-episode min distances before current snapshot resets
        steps += 1
        min_rr = torch.minimum(min_rr, cur_rr)
        min_rp = torch.minimum(min_rp, cur_rp)
        coll_rr = torch.maximum(coll_rr, cur_rrc)
        coll_rp = torch.maximum(coll_rp, cur_rpc)

        if n_steps >= total_steps:
            break
        obs, state = nobs, torch.as_tensor(nstate, device=env.device)
    summary = outcome.summary()
    summary["avg_reward"] = total_r / max(n_steps, 1)
    if n_socsteps:
        summary["soc_personal_viol_rate"] = soc_steps / n_socsteps
        summary["soc_comfort_viol_rate"] = viol_steps / n_socsteps
    return summary, None


def capture_trajectory(trainer, cfg, seed=0, n_agents=None,
                       steps=300, device="cpu"):
    """record pos/vel/goals for a single episode (for animation), returning np."""
    env = make_env(cfg, seed=seed, device=device, n=1)
    obs, _ = env.reset()
    N = env.N
    P = env.P
    traj = {"robo": [], "robo_goal": None, "rost": str(cfg.env.n_robots) if False else None,
            "ped": [], "size": float(cfg.env.size),
            "N": N, "P": P, "robot_r": cfg.env.robot_radius,
            "ped_r": cfg.env.ped_radius if P else None,
            "O": env.O, "ob_r": cfg.env.obstacle_radius if env.O else None,
            "goals": None}
    traj["robo_goal"] = np.array(env.robo_goal[0].cpu())
    traj["goals"] = traj["robo_goal"]
    obs = torch.as_tensor(obs, device=device)
    for _ in range(steps):
        act = trainer.act_det(obs)
        nobs, *_ = env.step(act)
        traj["robo"].append(np.array(env.robo_pos[0].detach().cpu()))
        if P:
            traj["ped"].append(np.array(env.ped_pos[0].detach().cpu()))
        obs = torch.as_tensor(nobs, device=device)
    traj["robo"] = np.stack(traj["robo"]) if traj["robo"] else \
        np.zeros((0, N, 2))
    if P:
        traj["ped"] = np.stack(traj["ped"]) if traj["ped"] else \
            np.zeros((0, P, 2))
    return traj
