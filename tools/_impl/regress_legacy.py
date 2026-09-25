"""regress_legacy.py -- bit-exact guard for the LEGACY behaviour of the env.

Why this exists: every new mechanism added in this project is supposed to
default to OFF and leave the historical results reproducible.  That claim was
checked by hand many times during the 2026-09-20/21 session (the recorded
values were ``success=1.0, wid=0.3105840510246344, nav=1.8789999999999998`` on
``runs/s2_amc_s0`` + ``configs/amc_mid.yaml``), but the harness that produced
them has since changed, so a permanent, self-contained check is better.

This script runs a *fixed* deterministic rollout (scripted actions, fixed
seeds, CPU) on the legacy mid recipe and hashes the per-step reward, cost and
robot-position tensors.  Any change that alters legacy behaviour -- even in the
last decimal -- changes the hash.  New mechanisms must therefore be added with
an explicit OFF switch and a test showing the hash is unchanged.

Usage:
    python scripts/regress_legacy.py            # check against the constant
    python scripts/regress_legacy.py --show     # print the current hashes
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from config import load_cfg
from envs.social_nav import SocialNavVecEnv

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Recorded 2026-09-21 06:0x with all 2026-09-20/21 mechanisms present but OFF.
# If a legitimate behaviour change is intended, update these deliberately and
# say so in the commit/notes -- do not silently re-baseline them.
# ---------------------------------------------------------------------------
# TWO baselines are guarded, because they serve two different purposes.
#
# (1) LEGACY: the pre-fix recipe `configs/amc_mid.yaml`.  Its NUMBERS are
#     obsolete as evidence (that scene spawned pedestrians on the robots' goals,
#     which polluted every social metric), but the code path is still the
#     regression test that proves a new mechanism is inert when switched off.
#
# (2) PAPER: the recipe the paper's baseline is actually trained with,
#     `configs/mappo_ws_curr.yaml` (the fixed 10x10 m scene).  THIS is the one
#     that protects the paper: whenever a new knob is added, this hash must not
#     move unless the knob was switched on deliberately.
# ---------------------------------------------------------------------------
EXPECTED = {
    # Re-baselined 2026-09-21: the per-environment collision flags
    # (_overlaps_rp/_overlaps_ob) used `.norm(-1)`, which collapsed the
    # distance tensor to a scalar, so the robot-pedestrian and robot-obstacle
    # reward penalties were a CONSTANT applied everywhere instead of a
    # per-environment signal.  Fixing that intentionally changes the reward
    # (this is a deliberate, documented behaviour change -- see NIGHT_REPORT
    # section 15), hence the new hash.
    "reward":  "c2f6ed884697bacdf54eb894f74444eff74158cd9c6d993607da31831d68d3a3",
    # NOTE: the COST hash is sensitive to the CPU thread count (torch CPU
    # reductions are not associative), so this baseline is only valid with
    # --threads 1, which is now the default.  The reward/robot hashes are
    # insensitive (in the legacy scene the scripted robots walk straight
    # regardless of the crowd).  Re-baselined 2026-09-21 under --threads 1.
    "cost":    "aa6114005b1978ab6b6116bb024db8d04679b82724c72765640cbb7ecbf92adf",
    "robot":   "3e688db4d850346b8fb16b99a149f1981e17f516d3302103530d17e71c960bfb",
}


# recorded 2026-09-21 with every new mechanism present but OFF
# re-baselined 2026-09-21 AFTER the ped_circulate reproducibility fix (that
# path used the global torch RNG, so the paper scene was not reproducible and
# this hash moved between processes -- the guard is what exposed it).
EXPECTED_PAPER = {
    # re-baselined for the same collision-flag fix (see the note on EXPECTED)
    "reward":  "814a0a7179661da86828cc37b23dece0b0c9898652a94fbf77c10f32844632df",
    "cost":    "39464cc1798df598272938325b1c32a740bad6202d8c9f99338256ee32c63e76",
    "robot":   "e0988ac4856eb3da1a0234394d1d2261cda43cfa66fda3df5a8006e8245e55d7",
}


def hashes(recipe="configs/amc_mid.yaml", n_env=16, steps=24, seed=1234):
    cfg = load_cfg(os.path.join(PROJ, recipe))
    cfg.env.auto_reset = False
    env = SocialNavVecEnv(cfg, n_parallel_envs=n_env, device="cpu")
    env.reset(seed=seed)
    h = {k: hashlib.sha256() for k in ("reward", "cost", "robot")}
    for t in range(steps):
        d = env.robo_goal - env.robo_pos
        act = d / (d.norm(dim=-1, keepdim=True) + 1e-6) * env.max_sp
        _, _, rew, _, _, _ = env.step(act)
        # the constrained cost is queried exactly as the trainer does
        cost = env.ped_mood_per_robot()
        for k, v in (("reward", rew), ("cost", cost), ("robot", env.robo_pos)):
            h[k].update(v.detach().to(torch.float64).numpy().tobytes())
    return {k: v.hexdigest() for k, v in h.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1,
                    help="torch CPU threads for the hash rollout.  The PAPER "
                         "scene is chaotic (16 pedestrians, geometric overlap "
                         "resolution), so with a variable thread count the same "
                         "rollout is NOT bit-reproducible across processes -- "
                         "measured: two runs of this guard gave different "
                         "hashes for configs/mappo_ws_curr.yaml.  Pinning to a "
                         "single thread is what makes the guard meaningful.")
    ap.add_argument("--recipe", default="configs/amc_mid.yaml")
    ap.add_argument("--n_env", type=int, default=16)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--scene", default="both", choices=["legacy", "paper", "both"],
                    help="which guarded baseline to check (default: both)")
    a = ap.parse_args()
    import torch as _t
    _t.set_num_threads(int(a.threads))
    todo = []
    if a.scene in ("legacy", "both"):
        todo.append(("legacy", EXPECTED, "configs/amc_mid.yaml"))
    if a.scene in ("paper", "both"):
        todo.append(("paper ", EXPECTED_PAPER, "configs/mappo_ws_curr.yaml"))
    failed = []
    for name, exp, recipe in todo:
        recipe = a.recipe if a.recipe != "configs/amc_mid.yaml" else recipe
        got = hashes(recipe, a.n_env, a.steps)
        for k, v in got.items():
            print(f"[{name}] {recipe:28s} {k:7s} {v}")
        if a.show:
            continue
        bad = [k for k, v in exp.items() if v != "PLACEHOLDER" and v != got[k]]
        if bad or any(v == "PLACEHOLDER" for v in exp.values()):
            failed.append((name, recipe, bad))
    if a.show:
        return
    if failed:
        for name, recipe, bad in failed:
            print(f"\nREGRESSION FAILED ({name}) {recipe}: {', '.join(bad) or '(empty baseline)'}")
        print("If the change is intentional, re-baseline EXPECTED/EXPECTED_PAPER "
              "explicitly and say so in the notes -- never silently.")
        raise SystemExit(1)
    print("\nboth guarded behaviours unchanged (bit-exact) OK")


if __name__ == "__main__":
    main()
