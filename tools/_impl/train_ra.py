"""
train_ra.py -- long-training entry for RA-CMAPPO and its ablated predecessors.

Runs the constrained dual-critic trainer (algorithms.ra_cmappo.Trainer) on a
SocialNav recipe, with a mode switch that collapses to:
    MAPPO          risk_mode='none'
    MAPPO-fixed    risk_mode='fixed'
    MAPPO+Lagrangian  risk_mode='adaptive'   (anisotropic=False)
    RA-CMAPPO      risk_mode='adaptive'      (anisotropic=True -> risk field)

One seed per process by default; loop seeds via shell or --seeds. Saves
checkpoint with metadata ('__kind__': 'ra_cmappo', risk_mode, cost_target,
lambda) so results/eval can reload deterministically.

Usage:
  # single seed
  python scripts/train_ra.py --recipe configs/amc_mid.yaml --mode RA-CMAPPO \
       --seed 0 --steps 6000000 --tag s2_amc --eval 40
  # 5 seeds in one process
  python scripts/train_ra.py --recipe configs/amc_mid.yaml --mode RA-CMAPPO \
       --seeds 0,1,2,3,4 --steps 6000000 --tag s2_amc --eval 40
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from config import load_cfg
from envs.social_nav import SocialNavVecEnv
from algorithms.ra_cmappo import Trainer
from algorithms.baselines import BASELINE_MODES, MODE_ENV

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODE_ALIASES = {
    "MAPPO": "MAPPO", "MAPPO_fixed": "MAPPO_fixed",
    "MAPPO-Lagrangian": "MAPPO_Lagrangian", "MAPPO_Lagrangian": "MAPPO_Lagrangian",
    "RA-CMAPPO": "RA_CMAPPO", "RA_CMAPPO": "RA_CMAPPO",
    # Mood-Shaping: affect enters the REWARD instead of the constraint
    # (present in BASELINE_MODES but missing from this alias table, so
    # every --mode Mood_Shaping run died with a KeyError).
    "Mood_Shaping": "Mood_Shaping", "Mood-Shaping": "Mood_Shaping",
}


def _parse_density_stages(spec):
    """'0:5,200:11,400:16' -> [(0, 5), (200, 11), (400, 16)] (sorted, validated)."""
    stages = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        it, _, p = part.partition(":")
        stages.append((int(it), int(p)))
    stages.sort()
    return stages


def _weighted_interleave(pairs, n_stages):
    """spread densities proportionally to their weights (largest-remainder style).

    `pairs` is a list of (P, weight).  The result is a sequence of densities in
    which each density appears about w/sum(w) of the time, distributed EVENLY
    rather than clustered -- which is what a domain-randomisation schedule wants.
    """
    total = sum(w for _, w in pairs) or 1.0
    assigned = [0] * len(pairs)
    out = []
    for s in range(n_stages):
        best, best_val = 0, -1e18
        for i, (_, w) in enumerate(pairs):
            val = w * (s + 1) / total - assigned[i]
            if val > best_val:
                best, best_val = i, val
        assigned[best] += 1
        out.append(pairs[best][0])
    return out


def _density_schedule(a, steps_total, rollout_len):
    """explicit 'iter:P' list, a cycled density list, or a WEIGHTED interleave."""
    if getattr(a, "density_cycle", ""):
        items = [x for x in str(a.density_cycle).split(",") if x.strip()]
        if not items:
            return ()
        n_it = int(steps_total // max(rollout_len, 1))
        L = max(int(a.density_stage_len), 1)
        n_stages = n_it // L + 1
        if any(":" in x for x in items):        # weighted: 'P:w,P:w,...'
            pairs = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in items]
            seq = _weighted_interleave(pairs, n_stages)
        else:                                    # uniform cycling
            dens = [int(x) for x in items]
            seq = [dens[i % len(dens)] for i in range(n_stages)]
        return [(i * L, seq[i]) for i in range(n_stages)]
    return _parse_density_stages(a.density_stages)


def run_seed(cfg, n_env, device, steps_total, seed, out_root, eval_every,
             ckpt_every=50, init_ckpt=None, density_stages=()):
    env = SocialNavVecEnv(cfg, n_parallel_envs=n_env, device=device,
                          base_seed=seed)
    trainer = Trainer(env, dict(cfg.algo), device=device)
    if init_ckpt:
        # Warm start the ACTOR from a checkpoint (typically a trained
        # unconstrained MAPPO baseline); the cost critics stay fresh.
        # Motivation: with the constraint engaged from scratch, lambda pushes
        # the policy into the degenerate "stand still = no intrusion" basin
        # (measured: success 0.000 once lambda reached ~6).  A hand-written
        # detour controller shows the intended solution costs only 3.6%
        # success for a 26% CVaR reduction -- the behaviour exists, the
        # optimiser just has to start near it.  load_state uses strict=False,
        # so only the matching actor.* tensors are copied.
        sd = torch.load(init_ckpt, map_location=device)
        before = {k: v.clone() for k, v in trainer.net.actor.state_dict().items()}
        trainer.load_state(sd)
        moved = sum(1 for k, v in trainer.net.actor.state_dict().items()
                    if not torch.equal(v, before[k]))
        print(f"[init] warm-started actor from {init_ckpt} "
              f"({moved}/{len(before)} tensors changed)", flush=True)
    rollout = int(cfg.train.rollout_len)
    hist = {"iter": [], "loss": [], "rew": [], "cost": [], "lambda": [],
            "samples": [], "eval": []}
    it = 0
    ckptdir = os.path.join(out_root, "ckpt", f"seed{seed}")
    os.makedirs(ckptdir, exist_ok=True)
    t0 = time.time()
    cur_P = int(getattr(cfg.env, "n_pedestrians", 0) or 0)
    while trainer.rollout_steps < steps_total:
        it += 1
        # ---- density curriculum: swap the environment when the stage changes
        if density_stages:
            want = cur_P
            for start, p in density_stages:
                if it >= start:
                    want = p
            if want != cur_P:
                cur_P = want
                cfg.env.n_pedestrians = want
                env = SocialNavVecEnv(cfg, n_parallel_envs=n_env, device=device,
                                      base_seed=seed)
                trainer.env = env
                print(f"[density] iter {it}: switching to P={want}", flush=True)
        data, stats = trainer.collect(rollout)
        m = trainer.update(data)
        hist["iter"].append(it)
        hist["loss"].append(m["loss"])
        hist["rew"].append(stats["rew_mean"])
        hist["cost"].append(m["cost_mean"])
        hist["lambda"].append(m["lambda"])
        hist["samples"].append(trainer.rollout_steps)
        got = trainer.rollout_steps
        elapsed = max(time.time()-t0, 1e-6)
        eta = max(steps_total-got, 0)/max(got/elapsed, 1e-9)
        extra = ""
        if "viol" in m:
            # The trainer's OWN violation and CVaR, in ITS units.  Calibrating
            # the budget from an offline episode-level analogue is wrong: the
            # trainer computes the CVaR over per-sample cost returns inside a
            # rollout window, which has a different scale (this is why a
            # "violating" budget produced lambda=0 for a whole run).
            extra = (f" viol {m['viol']:+.4f} cvar {m.get('cvar_cost', float('nan')):.3f}"
                     + (" WARMUP" if m.get("warmup") else "")
                     + (" GATED" if m.get("gated") else ""))
        if "succ_ema" in stats:
            extra += f" succ_ema {stats['succ_ema']:.3f}"
        # exploration scale: with an unbounded action head the raw std is
        # meaningless on its own (std 7.4 against |a| ~47 is only 9 degrees of
        # directional jitter), so print the std the policy is actually using --
        # it is the quantity the bounded-mean experiment is about.
        if hasattr(trainer, "net") and hasattr(trainer.net, "actor"):
            _std = float(torch.exp(
                trainer.net.actor.log_std.clamp(-5, 2)).mean())
            extra += f" std {_std:.2f}"
        print(f"[{cfg.train.exp_name}]{trainer.risk_mode} seed {seed} iter {it} "
              f"samples {got}/{steps_total} loss {m['loss']:.3f} "
              f"rew {stats['rew_mean']:.3f} cost {m['cost_mean']:.3f} "
              f"lam {m['lambda']:.3f}{extra} eta {eta/60:.0f}min", flush=True)
        if eval_every > 0 and it % eval_every == 0:
            # real task-success check (deterministic closed-loop), so we can tell
            # whether the run is actually improving -- not just training reward.
            try:
                from tools._impl.common import evaluate_deterministic
                ev_env = SocialNavVecEnv(cfg, n_parallel_envs=min(n_env, 64),
                                         device=device, base_seed=9000 + seed)
                ev, _ = evaluate_deterministic(trainer, ev_env,
                                               total_steps=1500)
                succ = ev.get("success_rate")
                rec = {"iter": it, "cost": m["cost_mean"], "lambda": m["lambda"],
                       "rew": stats["rew_mean"], "success_rate": succ,
                       "deadlock_rate": ev.get("deadlock_rate"),
                       "collrp_rate": ev.get("collrp_rate")}
                hist["eval"].append(rec)
                print(f"   EVAL iter {it}: success={succ:.3f} "
                      f"deadlock={ev.get('deadlock_rate'):.3f} "
                      f"collrp={ev.get('collrp_rate'):.3f}", flush=True)
            except Exception as ex:      # never let logging kill a long run
                print(f"   EVAL failed at iter {it}: {ex}", flush=True)
        # Periodically persist a checkpoint.  Previously the ONLY checkpoint
        # was written after the final iteration, so killing a 3-hour run (or a
        # crash) discarded the entire run.  Writing every `ckpt_every`
        # iterations costs ~1s and makes a run interruptible and resumable,
        # and lets us evaluate an intermediate policy without waiting for the
        # full schedule.
        if ckpt_every > 0 and it % ckpt_every == 0 and it > 0:
            try:
                trainer.save(os.path.join(ckptdir, f"iter{it}.pt"))
            except Exception as ex:
                print(f"   ckpt failed at iter {it}: {ex}", flush=True)
    ckpt = os.path.join(ckptdir, "final.pt")
    trainer.save(ckpt)
    res = {"history": hist, "checkpoint": ckpt, "kind": trainer.kind,
           "risk_mode": trainer.risk_mode, "cost_target": trainer.cost_target}
    fn = os.path.join(out_root, f"seed{seed}.json")
    with open(fn, "w") as f:
        json.dump(_jsonify(res), f, indent=2)
    return res


def _jsonify(o):
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(x) for x in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if hasattr(o, "item") and not isinstance(o, (str, bytes, bool)):
        try:
            return o.item()
        except Exception:
            return str(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="configs/amc_mid.yaml")
    ap.add_argument("--mode", default="RA-CMAPPO")
    ap.add_argument("--seeds", default=None, help="comma list or single int")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_env", type=int, default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--eval", type=int, default=0)
    ap.add_argument("--cost_target", type=float, default=None)
    ap.add_argument("--lambda_init", type=float, default=None)
    ap.add_argument("--lambda_lr", type=float, default=None)
    ap.add_argument("--lambda_max", type=float, default=None)
    ap.add_argument("--lambda_warmup_steps", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--entropy_coef", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--budget_ramp_steps", type=int, default=None)
    ap.add_argument("--budget_start_mult", type=float, default=None)
    ap.add_argument("--task_gate_threshold", type=float, default=None,
                    help="switch the constraint OFF while the windowed "
                         "success rate is below this value (0 = off)")
    ap.add_argument("--ckpt_every", type=int, default=50,
                    help="write an intermediate checkpoint every N iterations")
    ap.add_argument("--init_ckpt", default=None,
                    help="warm-start the actor from this checkpoint (cost "
                         "critics stay fresh); used to fine-tune a constrained "
                         "policy from an unconstrained baseline")
    ap.add_argument("--emo_reward_weight", type=float, default=None,
                    help="Mood-Shaping arm: coefficient of the affect penalty "
                         "in the REWARD (0 = off)")
    ap.add_argument("--emo_shaping_mode", default=None,
                    choices=["mean", "tail"],
                    help="which affect signal the shaping penalty uses: "
                         "'mean' = legacy /P-diluted distress, 'tail' = the "
                         "same per-robot cost the CVaR constraint sees")
    ap.add_argument("--deterministic", action="store_true",
                    help="force deterministic CUDA kernels (the env's overlap "
                         "resolution uses scatter reductions; two identical runs "
                         "of the same arm diverged to 0.646 vs 0.470 at iter 250)")
    ap.add_argument("--algo", nargs="*", default=[],
                    help="algo-key overrides, e.g. cost_adv_zscore=False "
                         "cvar_select=rollout (both default to the legacy "
                         "constrained surrogate)")
    ap.add_argument("--reward", nargs="*", default=[],
                    help="reward-weight overrides, e.g. collision_robot_ped=0.3; "
                         "needed because --env only reaches cfg.env, while the "
                         "collision weights live in cfg.reward")
    ap.add_argument("--timeout_penalty", type=float, default=None,
                    help="reward-side failure penalty (cfg.reward.timeout_penalty); "
                         "0 = legacy.  With --env collision_terminal=True this is "
                         "the penalty the agent receives when a contact ends the "
                         "episode.")
    ap.add_argument("--env", nargs="*", default=[],
                    help="generic env overrides, e.g. detour_prior_gain=1.0 "
                         "(recorded in the run's config.json)")
    ap.add_argument("--density-cycle", dest="density_cycle", default="",
                    help="ALTERNATIVE to --density-stages: a comma list of "
                         "densities cycled every --density-stage-len iterations, "
                         "e.g. '5,9,12,16,20'.  Interleaving exposes the policy to "
                         "the high densities EARLY and repeatedly instead of only "
                         "at the end of a monotone ramp, which is the usual "
                         "domain-randomisation argument.");
    ap.add_argument("--density-stage-len", dest="density_stage_len", type=int,
                    default=100);
    ap.add_argument("--density-stages", dest="density_stages", default="",
                    help="CROSS-DENSITY training schedule, e.g. '0:5,200:11,400:16' "
                         "(from iteration N, train with P pedestrians).  The "
                         "observation dimension is fixed by obs_capacity.peds=24, "
                         "so swapping the environment mid-run is safe, and because "
                         "the pedestrian slots are no longer constant the running "
                         "normaliser stops having var~0 dimensions -- which is "
                         "exactly the artefact that made P=16 zero-shot emit "
                         "anti-goal actions (NIGHT_REPORT section 49).")
    a = ap.parse_args()

    if a.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = load_cfg(os.path.join(PROJ, a.recipe))
    mode = MODE_ALIASES[a.mode]
    cfg.algo.update(BASELINE_MODES[mode])
    cfg.env.update(MODE_ENV.get(mode, {}))
    if a.emo_reward_weight is not None:
        cfg.env["emotion_reward_weight"] = a.emo_reward_weight
    if a.emo_shaping_mode is not None:
        cfg.env["emotion_shaping_mode"] = a.emo_shaping_mode
    for kv in a.algo:
        k, _, v = kv.partition("=")
        if v.lower() in ("true", "false"):
            cfg.algo[k] = (v.lower() == "true")
        else:
            try:
                cfg.algo[k] = float(v)
            except ValueError:
                cfg.algo[k] = v
    for kv in a.reward:
        k, _, v = kv.partition("=")
        try:
            v = float(v)
        except ValueError:
            pass
        cfg.reward[k] = v
    if a.timeout_penalty is not None:
        cfg.reward["timeout_penalty"] = a.timeout_penalty
    for kv in a.env:
        k, _, v = kv.partition("=")
        try:
            v = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            pass
        cfg.env[k] = v
    if a.cost_target is not None:
        cfg.algo["cost_target"] = a.cost_target
    if a.lambda_init is not None:
        cfg.algo["lambda_init"] = a.lambda_init
    if a.lambda_lr is not None:
        cfg.algo["lambda_lr"] = a.lambda_lr
    if a.lambda_max is not None:
        cfg.algo["lambda_max"] = a.lambda_max
    if a.lambda_warmup_steps is not None:
        cfg.algo["lambda_warmup_steps"] = a.lambda_warmup_steps
    if a.task_gate_threshold is not None:
        cfg.algo["task_gate_threshold"] = a.task_gate_threshold
    if a.budget_ramp_steps is not None:
        cfg.algo["budget_ramp_steps"] = a.budget_ramp_steps
    if a.budget_start_mult is not None:
        cfg.algo["budget_start_mult"] = a.budget_start_mult
    for k in ("gamma", "hidden", "entropy_coef", "lr"):
        v = getattr(a, k)
        if v is not None:
            cfg.algo[k] = v
    cfg.train.exp_name = f"week3_{mode}"

    device = a.device or cfg.vec.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    cfg.vec.device = device
    n_env = a.n_env or int(cfg.vec.n_parallel_envs)
    steps_total = a.steps or int(cfg.train.total_steps)

    if a.seed is not None:
        seeds = [a.seed]
    elif a.seeds is not None:
        seeds = [int(x) for x in str(a.seeds).split(",")]
    else:
        seeds = list(range(int(getattr(cfg.train, "num_seeds", 3))))

    out_root = os.path.join(PROJ, "runs", a.tag)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "config.json"), "w") as f:
        json.dump(_jsonify(cfg), f, indent=2, default=str)

    for sd in seeds:
        run_seed(cfg, n_env, device, steps_total, sd, out_root, a.eval,
                 ckpt_every=a.ckpt_every, init_ckpt=a.init_ckpt,
                 density_stages=_density_schedule(a, steps_total,
                                                  int(cfg.train.rollout_len)))
    print("done ->", out_root)


if __name__ == "__main__":
    main()
