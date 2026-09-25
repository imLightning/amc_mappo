"""
pareto_eval.py -- one consistent protocol for the task/emotion trade-off.

Motivation: this project has repeatedly compared numbers that were measured
with different protocols (trainer-normalised CVaR vs raw episode CVaR vs the
in-training evaluator with auto_reset=True).  Every comparison in the paper
must come from ONE script, so this is it.

For each arm (a checkpoint, or a scripted reference policy) it reports:
  success   -- finished-episode success rate (auto_reset=False)
  timeout   -- fraction that ran out of time
  wid       -- worst-decile cumulative pedestrian affective dose
  min_rp    -- closest robot-pedestrian approach
  nav_s     -- mean navigation time of successful episodes
  cvar_raw  -- CVaR_0.2 of the RAW discounted episode cost (plan_emo's C,
               including the terminal failure penalty when configured)
  yield     -- mean robot speed when a pedestrian is closing on the robot's
               own goal, vs otherwise (does it give way?)

Scripted references (beeline / detour) are included so the learned policy can
be placed against an achievable reference rather than only against itself.

Usage:
  python scripts/pareto_eval.py --arms configs/mappo_ws.yaml:runs/x/final.pt:MAPPO \
      --out results/bench/pareto.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from config import load_cfg
from envs.social_nav import SocialNavVecEnv

GAM = 0.99


def _parse_override(v):
    """Parse one ``--env k=v`` value.

    Scalars keep the legacy behaviour (int/float when numeric, else the raw
    string, which is what string knobs such as ``resp_mode=distance`` need).
    Bracketed values become lists, so list-valued knobs are reachable from the
    CLI: ``--env 'ped_yield_levels=[0.0,0.0,0.0]'`` (before this, the literal
    string "[0.0,0.0,0.0]" was written into the config and the env's
    ``if self.ped_yield_levels:`` branch then crashed on np.asarray(str)).
    """
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        body = v[1:-1].strip()
        if not body:
            return []
        out = []
        for item in body.split(","):
            item = item.strip()
            if item.lower() in ("true", "false"):
                out.append(item.lower() == "true")
            elif item.replace(".", "", 1).replace("-", "", 1).isdigit():
                out.append(float(item))
            else:
                out.append(item)
        return out
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v


def scripted(env, kind):
    """reference controller: 'beeline' or 'detour' (gain 1.0, no slowdown)."""
    d = env.robo_goal - env.robo_pos
    des = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
    if kind == "detour" and env.P:
        rel = env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)
        dist = rel.norm(dim=-1).clamp_min(1e-6)
        u = rel / dist.unsqueeze(-1)
        w = (1.5 - dist).clamp(0, 1).unsqueeze(-1)
        des = des + (u * w).sum(dim=2)
        des = des / (des.norm(dim=-1, keepdim=True) + 1e-6)
    elif kind == "stop":
        return torch.zeros(env.B, env.N, 2, device=env.device)
    elif kind == "creep":
        # "polite stop": freeze whenever a pedestrian is within 1.2 m.  This is
        # the action that MOST reduces the instantaneous affect cost, i.e. what
        # a per-step gradient on that cost points at locally.  Including it as a
        # reference separates "the objective is wrong" from "the optimiser
        # cannot find the good behaviour".
        if env.P:
            dist = (env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)).norm(dim=-1)
            gate = (dist.min(dim=-1).values > 1.2).float().unsqueeze(-1)
            return des * env.max_sp * gate
        return des * env.max_sp
    return des * env.max_sp


@torch.no_grad()
def evaluate(recipe, ckpt, label, device, kind=None, B=128, env_over=None,
             seed=0, cbf_min=None, cbf_alpha=0.5, cbf_mode="linear",
             cbf_also_scripted=False, classical_policy=None, sfm_params=None,
             sarl_ckpt=None, sarl_attention=1, cbf_emotion=False,
             cbf_mood_margin=0.3, cbf_q_gain=0.0, cbf_peer=False,
             cbf_d_peer=0.75, cbf_peer_all=False, cbf_ttc_gain=0.0,
             cbf_pred_gain=0.0, cbf_pred_horizon=1.0, cbf_ttc_adapt=0.0,
             cbf_ttc_near=1.5, cbf_shape=None):
    cfg = load_cfg(recipe)
    for k, v in (env_over or {}).items():
        cfg.env[k] = v
    cfg.env.auto_reset = False
    env = SocialNavVecEnv(cfg, n_parallel_envs=B, device=device)
    env.reset(seed=seed)
    start_pos = env.robo_pos.clone()      # for the SPL metric
    tr = None
    if ckpt:
        from tools._impl.eval_protocol import _make_trainer
        tr = _make_trainer(ckpt, env, dict(cfg.algo), device)
        tr.load(ckpt)
    cost = torch.zeros(B, env.N, device=device)
    disc, slow, fast, wids = 1.0, [], [], []
    # episode-level collision rates (CEMRRL-style table 3 columns): an episode
    # counts as a collision if ANY robot-pedestrian distance ever dropped below
    # the environment's own contact distance, at any step.
    classical = None
    if classical_policy and not ckpt:
        # LOUD guard instead of a silent wrong label.  A classical/separately
        # trained controller is attached to the LEARNED arm (its actions are
        # overridden), so without a ckpt `tr` stays None and the loop below
        # falls through to `scripted(env, kind)` -- which produced a "LSTM-RL"
        # row that was bit-identical to scripted-beeline.  Never again: pass a
        # real ckpt (any matching ckpt works; its actions are not used).
        raise SystemExit(
            f"FATAL: --policy {classical_policy} requires a learned arm "
            "(ckpt) to attach to; with ckpt='-' the controller is ignored and "
            "the row would silently be a scripted reference.")
    if classical_policy == "orca":
        from envs.orcasfm import ORCAPolicy
        classical = ORCAPolicy()
    elif classical_policy == "sfm":
        from envs.orcasfm import SFMPolicy
        classical = SFMPolicy(**sfm_params)
    elif classical_policy == "sarl":
        from envs.sarl import SARLPolicy
        # ---- ACTION-DECODING GUARD -----------------------------------------
        # A SARL checkpoint only means what it was trained to mean: `head_span`
        # (and the action-grid shape) are part of the ACTION MAPPING, not of the
        # network.  Evaluating a span-0.3 checkpoint with the default span 1.0
        # therefore does not error -- it silently decodes every chosen index into
        # the wrong velocity, which we did once and read as "the policy has 0%
        # success".  `train_sarl.py` saves its args next to the checkpoint, so
        # read them and refuse to guess.
        import json as _json
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(sarl_ckpt)))
        seed_tag = os.path.basename(os.path.dirname(os.path.abspath(sarl_ckpt)))
        args_path = os.path.join(os.path.dirname(os.path.abspath(sarl_ckpt)), "..", "..",
                                 f"{seed_tag}.json")
        args_path = os.path.normpath(args_path)
        saved = {}
        if os.path.exists(args_path):
            try:
                saved = _json.load(open(args_path)).get("args", {}) or {}
            except Exception:
                saved = {}
        else:
            print(f"[warn] no saved training args at {args_path}; using CLI values "
                  "for the SARL action grid (a mismatch silently changes the "
                  "decoded velocities)")
        span = float(sfm_params.get("head_span", saved.get("head_span", 1.0)))
        n_speeds = int(sfm_params.get("n_speeds", saved.get("n_speeds", 5)))
        n_head = int(sfm_params.get("n_head", saved.get("n_head", 8)))
        att = bool(sarl_attention) if "use_attention" not in saved \
            else bool(saved["use_attention"])
        if saved:
            if abs(span - float(saved.get("head_span", span))) > 1e-9:
                raise SystemExit(
                    f"FATAL: SARL head_span {span} != the checkpoint's training "
                    f"value {saved.get('head_span')}; the action decoding would be "
                    "silently wrong.")
            if att != bool(saved.get("use_attention", att)):
                raise SystemExit(
                    f"FATAL: SARL use_attention {att} != the checkpoint's training "
                    f"value {saved.get('use_attention')}.")
            print(f"[sarl] using the checkpoint's training action grid: "
                  f"head_span={span} n_speeds={n_speeds} n_head={n_head} "
                  f"attention={att}")
        classical = SARLPolicy(ckpt=sarl_ckpt, hidden=int(sfm_params.get("hidden", 128)),
                               n_speeds=n_speeds, n_head=n_head,
                               use_attention=att, head_span=span,
                               device=("cuda" if torch.cuda.is_available() else "cpu"))
    elif classical_policy == "dwa":
        from envs.orcasfm import DWAPolicy
        classical = DWAPolicy(**{k: v for k, v in sfm_params.items()
                                 if k in ("alpha", "beta", "gamma", "aw",
                                          "wmax", "nv", "nw", "horizon")})
    filt = None
    if cbf_min is not None:
        from envs.cbf import CBFFilter
        filt = CBFFilter(d_min=cbf_min, alpha=cbf_alpha,
                         vmax=float(getattr(env, "max_sp", 1.3)),
                         amax=float(getattr(env, "max_accel", 0.0) or 0.0),
                         dt=float(env.action_dt), passes=3, mode=cbf_mode,
                         emotion=bool(cbf_emotion),
                         mood_margin=float(cbf_mood_margin),
                         q_gain=float(cbf_q_gain),
                         ttc_gain=float(cbf_ttc_gain),
                         ttc_adapt=float(cbf_ttc_adapt),
                         ttc_near=float(cbf_ttc_near), shape=cbf_shape,
                         pred_gain=float(cbf_pred_gain),
                         pred_horizon=float(cbf_pred_horizon),
                         peer=bool(cbf_peer), d_peer=float(cbf_d_peer),
                         peer_front_only=not bool(cbf_peer_all))
    cbf_hmin = torch.full((B,), float("inf"), device=device)
    cbf_viol = torch.zeros(B, device=device)
    hit_rp = torch.zeros(B, device=device, dtype=torch.bool)
    hit_rr = torch.zeros(B, device=device, dtype=torch.bool)
    min_rp = torch.full((B,), float("inf"), device=device)
    min_rr = torch.full((B,), float("inf"), device=device)
    # ---- social-success criterion -------------------------------------------
    # Standard social-navigation convention (CEMRRL and the ORCA/SARL line):
    # an episode counts as a success only if the robots REACH their goals
    # WITHOUT intruding on anyone.  The whole threshold family is reported
    # instead of one hand-picked radius, so the claim cannot be attacked as
    # threshold shopping.  0.55 m is this environment's own contact distance.
    # ---- comfort / legibility / proactivity bookkeeping -------------------
    # All of these are computed from the SAME rollout, so every arm gets them
    # for free.  Definitions (also in MAIN_TABLE_TEMPLATE.md §A):
    #   jerk        : mean ||a_t - a_{t-1}|| / dt      (m/s^3, lower better)
    #   decel_p95   : 95th percentile of the deceleration along the direction of
    #                 motion (m/s^2, lower better)
    #   head_rate   : mean |d theta| / dt while moving (rad/s, lower better)
    #   spl          : success-weighted path-length ratio (1.0 = straight line)
    #   reveal       : fraction of the episode before the velocity direction
    #                 stays within 20 deg of the goal direction (legibility
    #                 proxy: how early the intent becomes unambiguous)
    #   yield_lead   : time between the first sustained slowdown and the closest
    #                 approach to any pedestrian (s, higher = more proactive)
    prev_v = env.robo_vel.clone()
    prev_a = None
    prev_head = torch.atan2(env.robo_vel[..., 1], env.robo_vel[..., 0])
    prev_pos = env.robo_pos.clone()
    path_len = torch.zeros(B, env.N, device=device)
    jerk_sum = torch.zeros((), device=device)
    head_sum = torch.zeros((), device=device)
    head_cnt = torch.zeros((), device=device)
    decel_hist = []
    dist_hist = []          # per step (B,N) min distance to any pedestrian
    speed_hist = []         # per step (B,N) speed
    goal_dir_ok = []        # per step (B,N) bool: velocity within 20 deg of goal
    VIOL_T = [0.55, 0.8, 1.0, 1.2]
    ever = {t: torch.zeros(B, device=device, dtype=torch.bool) for t in VIOL_T}
    cnt = {t: torch.zeros(B, device=device) for t in VIOL_T}
    # navigation time of successful episodes: the step at which EVERY robot of
    # the environment had arrived.  This used to be a hard-coded 0.0, which
    # made the nav column of every comparison table silently meaningless.
    t_done = torch.full((B,), float(env.horizon), device=device)
    # discounted episode RETURN under the recipe's own reward (including any
    # affect shaping): lets us compare what each arm was actually optimising,
    # so "the optimiser failed" can be told apart from "the objective is bad".
    ret = torch.zeros(B, device=device)
    for t in range(env.horizon):
        # classical controllers replace the LEARNED arm only; the scripted rows
        # must stay analytic references (otherwise the table would show ORCA
        # twice and hide the beeline/detour anchors)
        if classical is not None and tr is not None:
            act = classical.act(env)
        elif tr is not None:
            act = tr.act_det(env.get_obs()).reshape(B, env.N, -1)
        else:
            act = scripted(env, kind)
        # By default the filter is applied ONLY to learned arms: the scripted
        # arms are analytic references, and filtering them silently turns the
        # "beeline" row into "beeline+CBF", which is a different method.
        if filt is not None and (tr is not None or cbf_also_scripted):
            act = filt.project(env, act)
        _, _, rew_t, _, _, _ = env.step(act)
        ret += disc * rew_t.sum(dim=-1)
        # ---- comfort / legibility / proactivity accumulation -----------------
        dt = float(env.action_dt)
        v = env.robo_vel
        a = (v - prev_v) / dt
        if prev_a is not None:
            jerk_sum = jerk_sum + (a - prev_a).norm(dim=-1).sum()
        prev_a = a
        decel = -(a * v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)).sum(-1)
        decel_hist.append(decel.clamp_min(0.0).detach())
        speed_hist.append(v.norm(dim=-1).detach())
        sp = v.norm(dim=-1)
        head = torch.atan2(v[..., 1], v[..., 0])
        dh = torch.atan2(torch.sin(head - prev_head), torch.cos(head - prev_head))
        moving = sp > 0.1
        head_sum = head_sum + ((dh.abs() / dt) * moving).sum()
        head_cnt = head_cnt + moving.sum()
        prev_head = head
        path_len = path_len + (env.robo_pos - prev_pos).norm(dim=-1)
        prev_pos = env.robo_pos.clone()
        prev_v = v.clone()
        if env.P:
            dd = (env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)).norm(dim=-1)
            dist_hist.append(dd.amin(dim=-1).detach())          # (B,N)
        gd = (env.robo_goal - env.robo_pos)
        cosang = (gd * v).sum(-1) / (gd.norm(dim=-1) * sp).clamp_min(1e-6)
        goal_dir_ok.append((cosang > 0.94).detach())            # ~20 deg
        if cbf_min is not None and env.P:
            h = (env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)).norm(dim=-1)
            h = (h.amin(dim=(1, 2)) ** 2) - cbf_min ** 2
            cbf_hmin = torch.minimum(cbf_hmin, h)
            cbf_viol += (h < 0).float()
        if env.P:
            d_rp = (env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)).norm(dim=-1)
            m = d_rp.amin(dim=(1, 2))
            min_rp = torch.minimum(min_rp, m)
            hit_rp |= (d_rp < env.coll_rp).any(dim=-1).any(dim=-1)
            # NOTE: the loop variable must NOT be `t` -- it would clobber the
            # outer step index and silently corrupt nav_s (measured: nav_s
            # collapsed from 5.7 s to 0.11 s while this bug was live).
            for _th in VIOL_T:
                v = m < _th
                ever[_th] |= v
                cnt[_th] += v.float()
        if env.N > 1:
            d_rr = torch.cdist(env.robo_pos, env.robo_pos)
            d_rr = d_rr + torch.eye(env.N, device=device) * 1e3
            m = d_rr.amin(dim=(1, 2))
            min_rr = torch.minimum(min_rr, m)
            hit_rr |= (d_rr < env.coll_rr).any(dim=-1).any(dim=-1)
        fin = env.reached.all(-1) & ~env.crashed.any(-1)
        t_done = torch.where(fin & (t_done > (t + 1)),
                             torch.full_like(t_done, float(t + 1)), t_done)
        cost += disc * env.ped_mood_per_robot()
        disc *= GAM
        if env.P and env.N:
            dg = (env.ped_pos.unsqueeze(1) - env.robo_goal.unsqueeze(2)).norm(dim=-1)
            tw = ((env.robo_goal.unsqueeze(2) - env.ped_pos.unsqueeze(1))
                  * env.ped_vel.unsqueeze(1)).sum(-1) > 0.1
            near = ((dg < 1.5) & tw).any(-1)
            sp = env.robo_vel.norm(dim=-1)
            if bool(near.any()):
                slow.append(float(sp[near].mean()))
            if bool((~near).any()):
                fast.append(float(sp[~near].mean()))
    fin = env.reached.all(-1) & ~env.crashed.any(-1)
    succ = float(fin.float().mean())
    # ---- comfort / legibility / proactivity aggregates --------------------
    steps_run = max(len(decel_hist), 1)
    jerk = float(jerk_sum / (B * env.N * steps_run))
    dec = torch.stack(decel_hist).flatten() if decel_hist else torch.zeros(1, device=device)
    decel_p95 = float(torch.quantile(dec, 0.95)) if dec.numel() > 1 else 0.0
    head_rate = float(head_sum / head_cnt.clamp_min(1.0))
    d0 = (env.robo_goal - start_pos).norm(dim=-1)
    spl = float(((d0 / path_len.clamp_min(1e-6)).clamp(max=1.0))[fin].mean()) \
        if bool(fin.any()) else float("nan")
    # legibility proxy: last step at which the intent was still ambiguous
    # "reveal" = first step from which at least 80% of the remaining steps are
    # goal-aligned.  (A "stays aligned forever" rule gave 1.0 for every arm,
    # because the final approach always turns toward the goal.)
    ok = torch.stack(goal_dir_ok).float()              # (T,B,N)
    T = ok.shape[0]
    frac = ok.flip(0).cumsum(0).flip(0) / torch.arange(
        T, 0, -1, device=device).float().view(T, 1, 1)
    committed = frac >= 0.8
    first = torch.argmax(committed.float(), dim=0).float()   # 0 if never
    never = committed.sum(dim=0) == 0
    reveal = torch.where(never, torch.ones_like(first), first / T)
    reveal = float(reveal[fin].mean()) if bool(fin.any()) else float("nan")
    # proactivity: first sustained slowdown (3 steps below half speed) before the
    # closest approach to any pedestrian
    yield_lead = float("nan")
    if dist_hist:
        D = torch.stack(dist_hist)                     # (T,B,N)
        S = torch.stack(speed_hist)
        t_close = D.argmin(dim=0)                      # (B,N)
        # NOTE: must NOT be called `slow` -- that name already holds the list of
        # yield speeds used later, and shadowing it broke `if slow:` at the end.
        # slowdown must be RELATIVE to the speed reached so far; otherwise the
        # acceleration from rest at the start of the episode counts as yielding
        # (measured: every arm got ~3.6-3.8 s, i.e. the whole episode).
        run_max = S.cummax(dim=0).values
        slow_mask = ((S < 0.5 * run_max) & (run_max > 0.6 * float(env.max_sp))).float()
        sust = ((slow_mask.cumsum(dim=0) > 0) & (slow_mask.roll(1, 0) > 0)
                & (slow_mask.roll(2, 0) > 0))
        leads = []
        for b in range(B):
            for n in range(env.N):
                cand = torch.nonzero(sust[:t_close[b, n], b, n], as_tuple=False)
                if cand.numel():
                    leads.append(float(t_close[b, n] - cand[0, 0]) * dt)
        yield_lead = float(np.mean(leads)) if leads else float("nan")
    social = {t: float((fin & ~ever[t]).float().mean()) for t in VIOL_T}
    vtime = {t: float((cnt[t] / float(env.horizon)).mean()) for t in VIOL_T}
    vever = {t: float(ever[t].float().mean()) for t in VIOL_T}
    nav_s = float(t_done[fin].mean()) * float(getattr(env, "action_dt", 0.1)) \
        if bool(fin.any()) else float("nan")
    timeout = float((~env.reached.all(-1)).float().mean())   # not finished in time
    rf = cost.flatten()
    k = max(int(0.2 * rf.numel()), 1)
    cvar = float(torch.topk(rf, k, largest=True).values.mean())
    # worst-decile cumulative dose per pedestrian (the paper's `wid`)
    dose = env.ped_dose_impact.reshape(-1) if env.P else torch.zeros(1, device=device)
    if dose.numel() > 1:
        kk = max(int(0.1 * dose.numel()), 1)
        wids = float(torch.topk(dose, kk, largest=True).values.mean())
    return dict(label=label, seed=seed, success=round(succ, 4), timeout=round(timeout, 4),
                cvar_raw=round(cvar, 3), wid=round(wids, 4),
                nav_s=round(nav_s, 3), ret=round(float(ret.mean()), 3),
                coll_rp=round(float(hit_rp.float().mean()), 4),
                coll_rr=round(float(hit_rr.float().mean()), 4),
                min_rp=round(float(min_rp.mean()), 3),
                min_rr=round(float(min_rr.mean()), 3),
                social_succ={t: round(social[t], 4) for t in VIOL_T},
                cbf_hmin=(None if cbf_min is None else round(float(cbf_hmin.mean()), 4)),
                cbf_viol=(None if cbf_min is None
                          else round(float(cbf_viol.sum() / (B * env.horizon)), 5)),
                cbf_min=cbf_min, cbf_alpha=(cbf_alpha if cbf_min is not None else None),
                jerk=round(jerk, 3), decel_p95=round(decel_p95, 3),
                head_rate=round(head_rate, 3), spl=round(spl, 3) if spl == spl else None,
                reveal=round(reveal, 3) if reveal == reveal else None,
                yield_lead=round(yield_lead, 3) if yield_lead == yield_lead else None,
                viol_time={t: round(vtime[t], 4) for t in VIOL_T},
                viol_ever={t: round(vever[t], 4) for t in VIOL_T},
                yield_near=round(float(np.mean(slow)), 3) if slow else None,
                yield_far=round(float(np.mean(fast)), 3) if fast else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="recipe:ckpt:label  (ckpt may be '-' for scripted)")
    ap.add_argument("--scripted", nargs="*", default=[],
                    help="reference policies: beeline detour stop")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--policy", default=None,
                    choices=[None, "orca", "sfm", "dwa", "sarl"],
                    help="replace the learned action with a non-learned or "
                         "separately-trained controller: 'orca' = real ORCA via "
                         "RVO2 (reciprocal, all robots+pedestrians in one "
                         "simulator), 'sfm' = social force model, 'dwa' = "
                         "dynamic window, 'sarl' = ported social-DRL policy "
                         "(needs --sarl-ckpt)")
    ap.add_argument("--sarl-ckpt", default=None, help="checkpoint for --policy sarl")
    ap.add_argument("--sarl-attention", type=int, default=1,
                    help="1 = SARL (attention), 0 = LSTM-RL ablation")
    ap.add_argument("--sfm-params", nargs="*", default=[],
                    help="SFM overrides, e.g. A_ped=12 B_ped=0.5")
    ap.add_argument("--cbf-emotion", action="store_true",
                    help="affect-aware barrier: d_eff = max(d_soc(bearing), "
                         "d_min) + mood_margin*relu(-mood) (default OFF)")
    ap.add_argument("--cbf-peer", action="store_true",
                    help="add a RECIPROCAL robot-robot barrier (all robots run "
                         "the same filter, so peer avoidance is mutual)")
    ap.add_argument("--cbf-peer-all", action="store_true",
                    help="apply the peer barrier to ALL peers (not only those "
                         "ahead): measured to halve R-R contacts at a real task "
                         "cost, see NIGHT_REPORT section 44")
    ap.add_argument("--cbf-d-peer", dest="cbf_d_peer", type=float, default=0.75,
                    help="peer-robot exclusion distance for --cbf-peer (m)")
    ap.add_argument("--cbf-ttc-adapt", dest="cbf_ttc_adapt", type=float, default=0.0,
                    help="density-adaptive early yield: tau_eff = tau*(1+a)/"
                         "(1+a*n_near) with n_near pedestrians within "
                         "--cbf-ttc-near metres")
    ap.add_argument("--cbf-ttc-near", dest="cbf_ttc_near", type=float, default=1.5)
    ap.add_argument("--cbf-pred-gain", dest="cbf_pred_gain", type=float, default=0.0,
                    help="AFFECT-PREDICTIVE margin: radius += gain * predicted "
                         "mood loss over --cbf-pred-horizon (m per mood unit)")
    ap.add_argument("--cbf-pred-horizon", dest="cbf_pred_horizon", type=float,
                    default=1.0, help="prediction horizon in seconds")
    ap.add_argument("--cbf-ttc-gain", dest="cbf_ttc_gain", type=float, default=0.0,
                    help="early-yield margin: exclusion radius += tau * closing "
                         "speed (seconds of look-ahead); needs --cbf-emotion")
    ap.add_argument("--cbf-q-gain", dest="cbf_q_gain", type=float, default=0.0,
                    help="extra exclusion radius per unit of the three-layer "
                         "emotional impact q_t (needs --cbf-emotion)")
    ap.add_argument("--cbf-mood-margin", dest="cbf_mood_margin", type=float,
                    default=0.3,
                    help="extra exclusion radius (m) per unit of pedestrian "
                         "distress, used with --cbf-emotion")
    ap.add_argument("--cbf-min", type=float, default=None,
                    help="enable the discrete-time CBF velocity filter with this "
                         "d_min (m); the realised barrier is reported for EVERY "
                         "arm as cbf_hmin / cbf_viol, filtered or not")
    ap.add_argument("--cbf-alpha", type=float, default=0.5,
                    help="class-K gain of the discrete CBF condition")
    ap.add_argument("--cbf-shape", default=None,
                    help="explicit personal-space shape the BARRIER assumes, "
                         "'front,side,back' (e.g. '0.4,0.6,1.0' = the affect "
                         "shape reversed).  Default: read the shape from the "
                         "environment, i.e. the same geometry the affect COST "
                         "uses.  Set this to de-confound 'is the emotion-derived "
                         "shape the reason?': the cost geometry stays fixed while "
                         "only the controller's assumed shape changes.")
    ap.add_argument("--cbf-also-scripted", action="store_true",
                    help="also filter the scripted reference arms (off by default: "
                         "filtering 'beeline' makes it 'beeline+CBF', a different "
                         "method, so the reference rows would be mislabelled)")
    ap.add_argument("--cbf-mode", default="linear",
                    choices=["linear", "exact", "joint", "lex"],
                    help="'linear' = minimal-norm correction of the first-order "
                         "barrier (usable); 'exact' = projection onto the disk "
                         "exterior (measured to be ineffective here: when the "
                         "robot is already inside, c<0 makes the constraint "
                         "vacuous, so contacts barely improve)")
    ap.add_argument("--env", nargs="*", default=[],
                    help="env overrides for every arm, e.g. detour_prior_gain=1.0")
    ap.add_argument("--deterministic", action="store_true",
                    help="force deterministic CUDA kernels and seed torch/numpy: "
                         "the env's overlap resolution uses scatter reductions, so "
                         "identical scripted arms differed by up to 0.07 success "
                         "across processes (measured 0.805 vs 0.859 at env seed 0)")
    ap.add_argument("--seeds", default="1000,1001,1002",
                    help="comma list of ENVIRONMENT seeds; with more than one, "
                         "reports mean+-std over seeds (each seed re-samples the "
                         "crowd, so this is the reference-point error bar).  "
                         "DEFAULT IS THE HELD-OUT SET 1000/1001/1002: env.reset("
                         "seed=k) sets base_seed=k and the crowd RNG is "
                         "default_rng(base_seed + 7777733*(n_resets%%4096)), so "
                         "evaluating on seeds 0/1/2 reproduces the FIRST training "
                         "scene of the run with the same seed -- a train/test "
                         "leak.  Pass --seeds 0,1,2 explicitly only to reproduce "
                         "old numbers.")
    a = ap.parse_args()
    if a.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(0)
        np.random.seed(0)
    seeds = [int(x) for x in str(a.seeds).split(",") if x != ""]
    cbf_shape = None
    if a.cbf_shape:
        parts = [x for x in str(a.cbf_shape).replace(" ", "").split(",") if x]
        if len(parts) != 3:
            raise SystemExit("--cbf-shape needs exactly 3 values: front,side,back")
        cbf_shape = tuple(float(x) for x in parts)
    sfm_params = {}
    for kv in a.sfm_params:
        k, _, v = kv.partition("=")
        sfm_params[k] = float(v)
    env_over = {}
    for kv in a.env:
        k, _, v = kv.partition("=")
        env_over[k] = _parse_override(v)
    rows = []
    ref_recipe = a.arms[0].split(":")[0]
    for seed in seeds:
        for kind in a.scripted:
            rows.append(evaluate(ref_recipe, None, f"scripted-{kind}", a.device,
                                 kind=kind, env_over=env_over, seed=seed,
                                 cbf_min=a.cbf_min, cbf_alpha=a.cbf_alpha,
                                 cbf_mode=a.cbf_mode,
                                 cbf_also_scripted=a.cbf_also_scripted,
                                 cbf_emotion=a.cbf_emotion,
                                 cbf_mood_margin=a.cbf_mood_margin,
                                 cbf_q_gain=a.cbf_q_gain,
                                 cbf_peer=a.cbf_peer, cbf_d_peer=a.cbf_d_peer,
                                 cbf_peer_all=a.cbf_peer_all,
                                 cbf_ttc_gain=a.cbf_ttc_gain,
                                 cbf_pred_gain=a.cbf_pred_gain,
                                 cbf_pred_horizon=a.cbf_pred_horizon,
                                 cbf_ttc_adapt=a.cbf_ttc_adapt,
                                 cbf_ttc_near=a.cbf_ttc_near,
                                 cbf_shape=cbf_shape))
        for spec in a.arms:
            recipe, ckpt, label = spec.split(":")
            rows.append(evaluate(recipe, None if ckpt == "-" else ckpt, label,
                                 a.device, env_over=env_over, seed=seed,
                                 cbf_min=a.cbf_min, cbf_alpha=a.cbf_alpha,
                                 cbf_mode=a.cbf_mode,
                                 classical_policy=a.policy,
                                 sfm_params=sfm_params,
                                 sarl_ckpt=a.sarl_ckpt,
                                 sarl_attention=a.sarl_attention,
                                 cbf_emotion=a.cbf_emotion,
                                 cbf_mood_margin=a.cbf_mood_margin,
                                 cbf_q_gain=a.cbf_q_gain,
                                 cbf_peer=a.cbf_peer, cbf_d_peer=a.cbf_d_peer,
                                 cbf_peer_all=a.cbf_peer_all,
                                 cbf_ttc_gain=a.cbf_ttc_gain,
                                 cbf_pred_gain=a.cbf_pred_gain,
                                 cbf_pred_horizon=a.cbf_pred_horizon,
                                 cbf_ttc_adapt=a.cbf_ttc_adapt,
                                 cbf_ttc_near=a.cbf_ttc_near,
                                 cbf_shape=cbf_shape))
    print(f"\n{'arm':22s} {'success':>8s} {'timeout':>8s} {'cvar_raw':>9s} "
          f"{'wid':>7s} {'nav_s':>7s} {'coll_rp':>8s} {'coll_rr':>8s} "
          f"{'ret':>8s} {'yield near/far':>16s}")
    for r in rows:
        y = (f"{r['yield_near']}/{r['yield_far']}"
             if r["yield_near"] is not None else "-")
        print(f"{r['label']:22s} {r['success']:>8.3f} {r['timeout']:>8.3f} "
              f"{r['cvar_raw']:>9.2f} {r['wid']:>7.3f} {r['nav_s']:>7.2f} "
              f"{r['coll_rp']:>8.3f} {r['coll_rr']:>8.3f} "
              f"{r['ret']:>8.2f} {y:>16s}")
    print("\n=== SOCIAL SUCCESS: reach the goal AND never intrude past r ===")
    print(f"{'arm':22s} " + " ".join(f"{'succ@'+str(t):>9s}" for t in (0.55, 0.8, 1.0, 1.2)))
    for r in rows:
        ss = r.get("social_succ") or {}
        if not ss:
            continue
        def _g(d, t):
            return d[t] if t in d else d[str(t)]
        print(f"{r['label']:22s} " + " ".join(f"{_g(ss, t):>9.3f}" for t in (0.55, 0.8, 1.0, 1.2)))

    if len(seeds) > 1:
        import collections
        agg = collections.OrderedDict()
        for r in rows:
            agg.setdefault(r["label"], []).append(r)
        print(f"\n=== mean +- std over {len(seeds)} env seeds ===")
        print(f"{'arm':22s} {'success':>16s} {'cvar_raw':>16s} {'wid':>16s}")
        SOT = ("0.55", "0.8", "1.0", "1.2")
        summary = []
        for lab, rs in agg.items():
            line = {}
            ssv = {}
            for t in SOT:
                # keys are floats in-process and strings after a json round-trip
                vals = []
                for x in rs:
                    d = x.get("social_succ") or {}
                    if t in d:
                        vals.append(d[t])
                    elif float(t) in d:
                        vals.append(d[float(t)])
                    elif str(t) in d:
                        vals.append(d[str(t)])
                if vals:
                    ssv[t] = (float(np.mean(vals)), float(np.std(vals)))
            line["social_succ"] = ssv
            for src, key in (("viol_time", "viol_time"), ("viol_ever", "viol_ever")):
                dv = {}
                for t in SOT:
                    vals = []
                    for x in rs:
                        d = x.get(src) or {}
                        for kk in (t, float(t), str(t)):
                            if kk in d:
                                vals.append(d[kk])
                                break
                    if vals:
                        dv[t] = (float(np.mean(vals)), float(np.std(vals)))
                line[key] = dv
            for k in ("success", "cvar_raw", "wid", "coll_rp", "coll_rr", "nav_s",
                      "min_rp", "min_rr", "cbf_hmin", "cbf_viol", "jerk",
                      "decel_p95", "head_rate", "spl", "reveal", "yield_lead"):
                v = [x[k] for x in rs if x[k] is not None]
                m = float(np.mean(v)) if v else float("nan")
                sd = float(np.std(v)) if v else float("nan")
                line[k] = (m, sd)
            summary.append(dict(label=lab, n=len(rs),
                                **{k: line[k] for k in
                                   ("success", "cvar_raw", "wid", "coll_rp",
                                    "coll_rr", "nav_s", "min_rp", "min_rr",
                                    "cbf_hmin", "cbf_viol", "jerk", "decel_p95",
                                    "head_rate", "spl", "reveal", "yield_lead")},
                                social_succ=line["social_succ"],
                                viol_time=line["viol_time"],
                                viol_ever=line["viol_ever"]))
            print(f"{lab:22s} {line['success'][0]:>8.3f}+-{line['success'][1]:<6.3f}"
                  f" {line['cvar_raw'][0]:>8.2f}+-{line['cvar_raw'][1]:<6.2f}"
                  f" {line['wid'][0]:>8.3f}+-{line['wid'][1]:<6.3f}"
                  f" {line['nav_s'][0]:>7.2f} {100*line['coll_rp'][0]:>7.1f}%"
                  f" {100*line['coll_rr'][0]:>6.1f}%"
                  + "".join(f" {100*line['social_succ'][t][0]:>7.1f}%"
                            for t in SOT if t in line["social_succ"]))
        if a.out:
            pth = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), a.out)
            json.dump({"per_seed": rows, "summary": summary}, open(pth, "w"), indent=1)
            print("\nsaved", a.out)
        return
    if a.out:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(rows, open(p, "w"), indent=1)
        print("\nsaved", a.out)


if __name__ == "__main__":
    main()
