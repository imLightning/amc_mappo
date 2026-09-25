"""
eval_protocol.py -- strict train/test evaluation wrapper.

A single trained policy (ckpt) is evaluated under (a) the same recipe and (b)
a few *observation-dimension-preserving* perturbations (human-speed scale,
world size, seed), each for a requested number of finished episodes, with env
auto-reset disabled so every returned episode is metric-clean.

Honest scope note: changing #robots or #pedestrians changes the observation
dimension of our current policy, so generalization across densities needs an
obs-capacity (padding) representation fix first — not part of this wrapper.

Usage:
  python scripts/eval_protocol.py --recipe configs/amc_mid.yaml \
      --ckpt runs/s2_amc_s0/ckpt/seed0/final.pt --tag proto --n 220
"""
import argparse, copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools._impl.results_paths import RP  # noqa: E402
import numpy as np
import torch

from config import load_cfg
from envs.social_nav import SocialNavVecEnv
from envs.metrics import Outcome
from algorithms.mappo import Trainer
from algorithms.ra_cmappo import Trainer as RATrainer


def _make_trainer(ckpt, env, cfg_algo, device):
    """choose mappo vs ra_cmappo trainer from checkpoint metadata.

    Also warns loudly when the checkpoint lacks the observation-normaliser
    statistics: act_det() normalises with rms_obs, so a checkpoint saved
    without them yields mis-scaled inputs and near-zero actions.  Silence here
    previously produced plausible-looking but invalid numbers.
    """
    ck = {}
    try:
        ck = torch.load(ckpt, map_location="cpu")
    except Exception:
        ck = {}
    kind = ck.get("__kind__")
    if "rms_mean" not in ck:
        print(f"[eval_protocol][WARN] {os.path.basename(ckpt)} has no "
              f"rms_mean/rms_var: observation normalisation falls back to "
              f"mean=0/var=1, which mis-scales inputs and can zero out the "
              f"actions. Re-fit the normaliser or re-save the checkpoint.",
              file=sys.stderr)
    if kind == "ra_cmappo":
        return RATrainer(env, dict(cfg_algo), device=device)
    # default (also 'mappo' / legacy)
    return Trainer(env, dict(cfg_algo), device=device)

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_once(base, ckpt, B=256, n_ep=300, device=None,
             speed_scale=None, size=None, actor=None):
    cfg = copy.deepcopy(base)
    # honour the algorithm kind stored in checkpoint so critics line up
    try:
        meta = torch.load(ckpt, map_location="cpu")
        kind = meta.get("__kind__")
        if kind:
            cfg.algo.kind = kind
    except Exception:
        pass
    cfg.env.auto_reset = False
    if speed_scale is not None:
        cfg.env.pedestrian_max_speed = round(
            float(cfg.env.pedestrian_max_speed) * speed_scale, 3)
    if size is not None:
        cfg.env.size = float(size)
    device = device or str(cfg.vec.device)
    env = SocialNavVecEnv(cfg, n_parallel_envs=B, device=device, base_seed=1)
    trainer = None
    if actor is None:
        trainer = _make_trainer(ckpt, env, dict(cfg.algo), device)
        trainer.load(ckpt)
    out = Outcome(n_agents=env.N)
    elapsed = np.zeros(B, int)
    near_t = float(env.r + env.ped_r + 0.45)
    pv_cnt = np.zeros(B)
    ttc1_cnt = np.zeros(B)
    ttc2_cnt = np.zeros(B)
    ped_dist_sum = np.zeros(B)
    ped_step = np.zeros(B)
    PTR = [np.zeros((0, env.N, 2)) for _ in range(B)]   # pos trace per env
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, device=device).float()
    guard = 0
    while out.n_eps < n_ep and guard < n_ep * max(env.horizon, 1) * 20:
        guard += 1
        if actor is not None:
            act = actor(obs, env)
        else:
            act = trainer.act_det(obs)
        o2, st, rew, done, trunc, info = env.step(act)
        done = done.cpu().numpy().astype(bool)
        trunc = trunc.cpu().numpy().astype(bool)
        pos_now = env.robo_pos.detach().cpu().numpy()   # (B,N,2)
        nd = np.nonzero(~done)[0]
        if env.P:
            dpv = (env.robo_pos.unsqueeze(2) -
                   env.ped_pos.unsqueeze(1)).norm(dim=-1).amin(dim=(1, 2)).cpu().numpy()
            pv_cnt[nd] += (dpv[nd] < near_t)
            # TTC thresholds from risk signal (per env: any robot)
            sig = env.risk_signal()
            mt = sig["min_ttc"].detach().cpu().numpy()       # B,N
            t1 = (mt < 1.0).any(axis=1)
            t2 = (mt < 2.0).any(axis=1)
            ttc1_cnt[nd] += t1[nd].astype(int)
            ttc2_cnt[nd] += t2[nd].astype(int)
            # pedestrian disturbance proxy: robot-induced speed reduction
            nom = float(env.ped_sp)
            psp = env.ped_vel.norm(dim=-1).detach().cpu().numpy()   # B,P
            reduce_ = np.clip(nom - psp, 0.0, None).mean(axis=1)     # B
            ped_dist_sum[nd] += reduce_[nd]
            ped_step[nd] += 1
        for e in range(B):                              # append step to trace
            PTR[e] = np.concatenate([PTR[e], pos_now[None, e]], 0)
        for e in np.nonzero(done)[0]:
            steps = int(elapsed[e] + 1)
            P = PTR[e]
            trace = path_stats(P, env.robo_goal[e].detach().cpu().numpy(),
                               steps, env)
            if env.P:
                dcur = (env.robo_pos[e][:, None] - env.ped_pos[e][None]).norm(-1)
                if bool((dcur < near_t).any().item()):
                    pv_cnt[e] += 1
            rr_ = float(info["rr_min"][e]); rp_ = float(info["rp_min"][e])
            pv = float(pv_cnt[e]) / max(steps, 1)
            # deadlock: timed out AND the robot actually stopped making progress.
            # Use NET displacement (start->end), not mean per-step speed: a robot
            # stuck oscillating in a jam has high step-to-step movement but zero
            # net progress.  Discriminates deadlock from plain "too slow".
            deadlock = 0
            if trunc[e] and P.shape[0] >= 2:
                net = np.linalg.norm(P[-1] - P[0], axis=1)          # (N,)
                if float(net.mean()) < 0.5:      # <0.5 m net travel in a whole ep
                    deadlock = 1
            trace.update({"personal_viol": pv, "discomfort": pv,
                          "nav_time": steps * env.action_dt,
                          "ttc_low": float(ttc1_cnt[e]) / max(steps, 1),
                          "ttc_low1": float(ttc1_cnt[e]) / max(steps, 1),
                          "ttc_low2": float(ttc2_cnt[e]) / max(steps, 1),
                          "deadlock": deadlock,
                          "ped_disturb": float(ped_dist_sum[e]) / max(ped_step[e], 1),
                          # affective metrics: eval_protocol previously omitted
                          # these, so Outcome recorded None and every arm
                          # reported mpi/wmp/ari = 0.0 -- i.e. the emotion
                          # comparison was silently empty.
                          "mpi": float(info["mpi"][e]) if "mpi" in info else None,
                          "wmp": float(info["wmp"][e]) if "wmp" in info else None,
                          "ari": float(info["ari"][e]) if "ari" in info else None,
                          "aid": float(info["aid"][e]) if "aid" in info else None,
                          "wid": float(info["wid"][e]) if "wid" in info else None})
            out.add_episode(not trunc[e], trunc[e], steps,
                            rr_ if np.isfinite(rr_) else None,
                            rp_ if np.isfinite(rp_) else None,
                            float(info["rr_coll"][e]),
                            float(info["rp_coll"][e]), trace)
            elapsed[e] = 0; pv_cnt[e] = 0
            ttc1_cnt[e] = 0; ttc2_cnt[e] = 0
            ped_dist_sum[e] = 0; ped_step[e] = 0
            PTR[e] = PTR[e][0:0].reshape(0, env.N, 2)
        elapsed[~done] += 1
        if done.any():
            obs, _ = env.respawn_done_envs(torch.from_numpy(done))
        else:
            obs = o2
        obs = torch.as_tensor(obs, device=device).float()
    return out.summary(env.action_dt)


def path_stats(P, goal, steps, env):
    """per-trace path metrics from recorded positions P (T,N,2)."""
    if P.shape[0] < 2 or P.shape[1] == 0:
        return {"path_len": 0.0, "avg_speed": 0.0, "spl": 0.0,
                "oscillation": 0.0}
    dt = env.action_dt
    d = np.linalg.norm(np.diff(P, axis=0), axis=2)        # (T-1,N)
    plen = d.sum(0)
    sp = plen / (d.shape[0] * dt)
    g = np.asarray(goal)
    direct = np.linalg.norm(P[0] - g, axis=1).clip(1e-3)
    endd = np.linalg.norm(P[-1] - g, axis=1)
    spl = np.clip(direct / np.maximum(direct, endd), 0.0, 1.0)
    v = d / dt
    osc = float(np.mean(np.abs(np.diff(v, axis=0)))) if v.shape[0] >= 2 else 0.0
    return {"path_len": float(plen.mean()),
            "avg_speed": float(sp.mean()),
            "spl": float(spl.mean()),
            "oscillation": osc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="proto")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--speeds", default="1.0,1.35")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--profiles", default="",
                    help="comma profile names from scenario_generator.TestProfiles")
    a = ap.parse_args()
    base = load_cfg(os.path.join(PROJ, a.recipe))
    ck = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(PROJ, a.ckpt)
    sums = {"in-dist": run_once(base, ck, 256, a.n, a.device)}
    for k in [float(x) for x in a.speeds.split(",") if x][1:]:
        sums[f"speed_{k:g}"] = run_once(base, ck, 256, a.n, a.device,
                                        speed_scale=k)
    for s in [float(x) for x in a.sizes.split(",") if x]:
        sums[f"size_{s:g}"] = run_once(base, ck, 256, a.n, a.device, size=s)
    if a.profiles:
        from envs.scenario_generator import profile
        for name in a.profiles.split(","):
            name = name.strip()
            if not name:
                continue
            pr = profile(name)
            cfgp = copy.deepcopy(base)
            cfgp.env.n_pedestrians = pr["n_pedestrians"]
            cfgp.env.pedestrian_max_speed = pr["pedestrian_max_speed"]
            cfgp.env.max_robot_speed = pr["max_robot_speed"]
            sums[name] = run_once(cfgp, ck, 256, a.n, a.device)
    print(json.dumps(sums, indent=2, default=str))
    with open(RP("eval", f"evalproto_{a.tag}.json"), "w") as f:
        json.dump(sums, f, indent=2, default=str)


if __name__ == "__main__":
    main()
