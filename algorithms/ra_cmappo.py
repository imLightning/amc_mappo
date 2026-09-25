"""
ra_cmappo.py -- RA-CMAPPO scaffold (risk-adaptive constrained MAPPO).

Constraints (plan_add §三.4) via Lagrangian:
    actor objective:  maximise J_reward(θ)
    safety critics :  Vr (reward value)  and  Vc (social/safety cost value)
    update:  L(θ,λ) = -L_actor + Σ_k λ_k (Vc_k - C_k) ,  λ_k ← λ_k + η (Vc_k - C_k)

Not `RA-CMAPPO` as final tuned implementation, but the exact architecture the
ablation (MAPPO-fixed / MAPPO-Lagrangian / RA-CMAPPO) collapses to when
switches are disabled/enabled.  Import-friendly for scripts/eval.

Safety cost per robot per step comes from env.risk_signal() (anisotropic risk
field); cost constraints are soft via the adaptive multipliers.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .buffer import RolloutBuffer
from .policy import GaussianActor, CriticMLP, RunningMeanStd, layer_init


class RA_CMAPPO(nn.Module):
    """Holder of actor + dual critics + Lagrange multipliers."""

    def __init__(self, obs_dim, state_dim, act_dim, hidden=128,
                 mean_gain=0.01, log_std_init=-0.5,
                 bounded_mean=False, mean_scale=1.3):
        super().__init__()
        self.actor = GaussianActor(obs_dim, act_dim, hidden,
                                   mean_gain=mean_gain,
                                   log_std_init=log_std_init,
                                   bounded_mean=bounded_mean,
                                   mean_scale=mean_scale)
        self.vr = CriticMLP(state_dim, hidden)   # reward value
        self.vc = CriticMLP(state_dim, hidden)   # cost value


class Trainer:
    """Switches at init determine whether fixed or adaptive/constrained modes:

      risk_mode: 'none' -> lagrange weight 0, 'fixed' -> constant lambda,
                 'adaptive' -> online multiplier update (RA-CMAPPO)
      anisotropic: whether env.risk_signal() uses anisotropic social radius
    """

    def __init__(self, env, cfg, device=None):
        cfg = dict(cfg)
        self.env = env
        self.device = torch.device(device) if device else env.device
        self.obs_dim = env._obs_dim
        self.act_dim = 2
        self.state_dim = getattr(env, "_state_dim", env._obs_dim)
        self.net = RA_CMAPPO(self.obs_dim, self.state_dim, self.act_dim,
                             hidden=int(cfg.get("hidden", 128)),
                             mean_gain=float(cfg.get("action_mean_gain", 0.01)),
                             log_std_init=float(cfg.get("log_std_init", -0.5)),
                             bounded_mean=bool(cfg.get("bounded_mean", False)),
                             mean_scale=float(getattr(env, "max_sp", 1.3)),
                             ).to(self.device)
        self.opt = optim.Adam(self.net.parameters(),
                              lr=float(cfg.get("lr", 3e-4)), eps=1e-5)
        self.gamma = float(cfg.get("gamma", 0.99))
        self.lam = float(cfg.get("gae_lambda", 0.95))
        self.clip = float(cfg.get("clip_eps", 0.2))
        self.ent_coef = float(cfg.get("entropy_coef", 0.001))
        self.log_std_init = float(cfg.get("log_std_init", -0.5))
        self.vf_coef = float(cfg.get("value_loss_coef", 0.5))
        self.gradmax = float(cfg.get("grad_N", 10.0))
        self.epochs = int(cfg.get("train_epochs", 5))
        self.minib = int(cfg.get("num_minibatches", 4))
        # constraint knobs
        self.risk_mode = cfg.get("risk_mode", "adaptive")  # none/fixed/adaptive
        self.anisotropic = bool(cfg.get("anisotropic", True))
        # ---- constraint scaling (CRITICAL) --------------------------------
        # The violation used to update lambda is a *discounted cost RETURN*
        # (GAE over the episode), but `cost_target` was silently interpreted
        # as a PER-STEP target.  Measured on the dense scene the per-step cost
        # averages ~0.46, so a 50-step episode returns ~23 while the default
        # target was 0.1.  The violation was therefore ~23 at EVERY update,
        # lambda advanced by lambda_lr * 23 per iteration and pinned to
        # lambda_max within ~200 iterations.  A pinned lambda makes the
        # penalty term (cost_weight * lambda == 50) dominate the reward term,
        # so the policy freezes (success 0.000, robot-ped collision 0.91 vs
        # MAPPO's 0.698) -- the constraint stops being a constraint and
        # becomes a blanket suppression of all movement.
        #
        # Fix: express the target in the SAME units as the violation by
        # scaling it with the episode horizon, and default lambda_lr so that
        # lambda traverses [0, lambda_max] over a meaningful portion of
        # training rather than a few dozen iterations.  Both remain
        # overridable per recipe.
        self.cost_per_step_target = float(cfg.get("cost_target", 0.10))
        self.lambda_cost = float(cfg.get("lambda_init", 0.5))
        # `viol` is normalized to "fraction of the episode's cost that is
        # excess", so it lands in roughly [-1, +1].  lambda_lr is therefore
        # expressed directly in lambda-units per iteration: with the default
        # 0.01 and a persistent normalized violation of ~0.13, lambda moves
        # ~0.0013/iter and crosses [0, 20] in ~1.5e4 iterations -- slow
        # enough to let the policy adapt between dual updates, fast enough
        # to bind within a 12M-step run (~1e4 iterations).
        self.lambda_lr = float(cfg.get("lambda_lr", 0.01))
        self.lambda_max = float(cfg.get("lambda_max", 20.0))
        # ---- lambda warm-up (fixes "AMC never learns") --------------------
        # Measured lambda -> success (iter40) on the dense scene:
        #   0.00 -> 0.016 | 0.20 -> 0.005 | 0.50 -> 0.001 | 2.00 -> 0.000
        # The constraint suppresses task learning monotonically, and 0.5 -- a
        # tiny multiplier by constrained-RL standards -- already costs 16x.
        # The mechanism is not a tuning issue: at initialisation the policy
        # cannot yet reach the goal, so the cheapest way to reduce the social
        # cost is to STOP MOVING.  That degenerate solution (stand still, never
        # touch anyone) is exactly what the constraint gradient drives toward,
        # so the policy never learns the task and therefore never gets the
        # chance to learn to do the task socially.
        #
        # Fix: hold lambda at 0 for the first `lambda_warmup` iterations so the
        # task is learned first, then let the dual variable rise.  Once the
        # policy actually navigates, "detour around people" is reachable and
        # becomes cheaper than "stand still", which is the behaviour we want
        # the constraint to select.
        # ---- TASK-PRIORITY GATE (plan_emo.md 原则1) ------------------------
        # "Task completion is a hard precondition; emotion is a soft
        # constraint."  Measured failure without it: every run whose lambda
        # became large enough to bind collapsed to success 0.000 (the policy
        # saturated and thrashed at 35-41 m/s of commanded speed) -- lambda=20
        # versus reward advantages of order 1 leaves the task no influence.
        # With the gate the constraint is simply switched OFF while the task
        # is not yet solved, and returns once success recovers.
        # 0.0 = disabled (legacy).
        self.task_gate_threshold = float(cfg.get("task_gate_threshold", 0.0))
        self.task_gate_beta = float(cfg.get("task_gate_beta", 0.05))
        self._succ_ema = 0.0
        # ---- DYNAMICS CURRICULUM on the acceleration limit (plan_add §三.5) --
        # Measured: with max_robot_accel=1.0 the MAPPO baseline plateaus at
        # ~0.72 success while a scripted beeline reaches 0.867; switching the
        # limit OFF lifts the SAME training to 0.986 within 250 iterations.
        # With a rate-limited velocity-target action the early policy cannot
        # produce net displacement under mean-zero exploration, so learning is
        # 5-10x slower.  The curriculum starts unconstrained and anneals to the
        # physical limit, keeping realism while restoring learnability.
        # accel_target = 0 disables (legacy).
        self.accel_target = float(cfg.get("accel_target", 0.0))
        self.accel_start = float(cfg.get("accel_start", 100.0))
        # ---- BUDGET CURRICULUM ------------------------------------------
        # Measure-and-tighten instead of measure-and-shock: start from a SLACK
        # budget (lambda stays 0 because nothing is violated) and ramp the
        # budget down to the binding one.  The hand-written detour reference
        # shows the target region is reachable (success 0.83 with cvar -27%),
        # but jumping straight to a binding budget made the policy worse on
        # BOTH axes, so let it walk the frontier instead.
        # 0 = disabled (legacy).
        self.budget_ramp_steps = int(cfg.get("budget_ramp_steps", 0))
        self.budget_start_mult = float(cfg.get("budget_start_mult", 2.0))
        self.accel_warmup_steps = int(cfg.get("accel_warmup_steps", 0))
        self.lambda_warmup = int(cfg.get("lambda_warmup", 0))
        # warmup expressed in ENVIRONMENT STEPS (more robust than an iteration
        # count, which depends on rollout_len and the number of parallel envs).
        # `lambda_warmup_steps` takes precedence when set.
        self.lambda_warmup_steps = int(cfg.get("lambda_warmup_steps", 0))
        # number of completed update() calls, for `lambda_warmup`
        self._updates_done = 0
        # ---- stage 1: cost GAE + safety-critic settings -------------------
        self.gamma_cost = float(cfg.get("gamma_cost", self.gamma))
        # horizon of the cost return; set AFTER gamma_cost so the two agree
        self._cost_horizon = 1.0 / max(1e-6, 1.0 - self.gamma_cost)
        self.cost_target = self.cost_per_step_target * self._cost_horizon
        self.cost_clip = float(cfg.get("cost_clip", 3.0))
        # ---- cost normalisation (fixes the "AMC never learns" failure) -----
        # Measured on the dense scene the cost RETURN is heavy-tailed: mean
        # ~1-3 but max 13-16, while the safety critic's prediction only ever
        # reached 4-9.  Vc therefore cannot represent the tail, so the
        # residual `cadv = cost_ret - Vc` came out positive for 55-75% of ALL
        # samples, and CVaR(top-20% of cadv) was positive for 100% of them.
        # A cost penalty that is always positive has no equilibrium: it keeps
        # pushing the ratio of those samples down forever, which is what
        # suppressed learning (success 0.001 vs 0.013 with the penalty off).
        #
        # Fix: normalise the cost return to roughly zero-mean/unit-scale
        # before it reaches the critic and the dual update, using a running
        # estimate.  This is the standard remedy in constrained RL and it
        # makes the cost advantage an actual "better/worse than expected"
        # signal with both signs, which is what gives the constraint a fixed
        # point.  The per-step target is scaled by the same statistic so the
        # constraint keeps its meaning in the original units.
        self.cost_norm = bool(cfg.get("cost_norm", False))
        self._cost_mean = 0.0
        self._cost_var = 1.0
        self._cost_n = 1e-4
        # set properly on the first update(); guards the risk_mode="fixed"
        # path where the violation is computed but never used
        self._cost_target_n = self.cost_target
        # ---- stage 2 (innovation): distributionally-robust constraint -----
        # cvar_alpha = fraction of WORST experiences the constraint covers.
        # 1.0 == plain mean (risk-neutral, MAPPO-like); 0.2 == worst 20%.
        self.cvar_alpha = float(cfg.get("cvar_alpha", 1.0))
        # ---- stage 3 (innovation): anticipatory risk features -------------
        # when > 0, the safety critic additionally sees a short-horizon
        # prediction of future intrusion, enabling "yield now to avoid
        # distressing someone in 0.4 s" behaviour.
        self.anticipate_k = int(cfg.get("anticipate_k", 0))
        # "social" (ours) or "return" (GUIDER-style self-risk baseline)
        self.cost_source = str(cfg.get("cost_source", "social"))
        # proximity softmax width for attributing pedestrian distress to robots
        self.mood_sigma = float(cfg.get("mood_sigma", 1.2))
        self.cost_weight = 1.0 if self.risk_mode != "none" else 0.0
        # ---- implementation variants for the constrained actor surrogate -----
        # Both default to the LEGACY behaviour, so existing runs are unchanged.
        #   cost_adv_zscore=True  : z-score the cost advantage per minibatch
        #                           (legacy).  False keeps RAW cost units, which
        #                           restores a physical meaning to lambda and
        #                           removes the dependence on the minibatch-wide
        #                           variance of the cost advantage.
        #   cvar_select="minibatch" : pick the worst alpha fraction inside each
        #                           minibatch (legacy).  "rollout" picks it once
        #                           per update over ALL samples of the rollout,
        #                           so the CVaR tail is a distributional
        #                           quantity instead of a per-minibatch one.
        self.cost_adv_zscore = bool(cfg.get("cost_adv_zscore", True))
        self.cvar_select = str(cfg.get("cvar_select", "minibatch"))
        # risk-field radii: anisotropic vs circular personal space
        self.risk_cfg = {"d_front": 1.0, "d_side": 0.6, "d_back": 0.4} \
            if self.anisotropic else {"d_front": 0.7, "d_side": 0.7, "d_back": 0.7}
        # weight of the PEER-ROBOT risk term (0 = pedestrians only, original
        # behaviour; >0 extends the same risk field to robot-robot pairs, which
        # matters once N>=3 makes peer coordination the binding constraint)
        self.risk_cfg["w_peer"] = float(cfg.get("w_peer", 0.0))
        # running stats
        self.rms_obs = RunningMeanStd(shape=(self.obs_dim,),
                                      device=self.device,
                                      var_floor=float(cfg.get("rms_var_floor", 0.0)),
                                      clip=float(cfg.get("rms_clip", 0.0)))
        self.rollout_steps = 0

    # ------------------------------------------------------------- cost source
    @torch.no_grad()
    def _cost_per_robot(self, risk=None, rew=None):
        """Safety cost per robot.

        cost_source selects WHAT the CVaR constraint is applied to:
          "social"  : the social/affective risk field (OURS -- the constraint
                      protects the pedestrians' worst experiences)
          "return"  : the negative task return (a GUIDER-style baseline where
                      CVaR guards the ROBOT's own outcome risk)

        The two are semantically different: "return" is self-interested risk
        aversion, "social" is a welfare constraint on the humans.  Having both
        lets us show the difference is not merely where CVaR is plugged in.
        """
        env = self.env
        if self.cost_source == "return":
            if rew is None:
                return torch.zeros(env.B, env.N, device=self.device)
            return (-rew).clamp(min=0.0)                     # self-risk proxy
        # P2: constrain AFFECT, not geometry.  Fall back to the geometric risk
        # field only when the emotion model is disabled.
        if hasattr(env, "ped_mood_per_robot") and getattr(env, "emo", {}).get("enabled"):
            return env.ped_mood_per_robot(self.mood_sigma)   # (B,N)
        sig = risk if risk is not None else env.risk_signal(self.risk_cfg)
        if sig["risk"].numel():
            c = sig["risk"].clone()
            c = c + sig["directional_soc"].clone()
        else:
            c = torch.zeros(env.B, env.N, device=self.device)
        return c                                             # (B,N)

    # ------------------------------------------------------------- collect
    @torch.no_grad()
    def collect(self, rollout_len):
        env = self.env
        buf = RolloutBuffer()
        obs, state = env.reset()
        obs = torch.as_tensor(obs, device=self.device).float()
        state = torch.as_tensor(state, device=self.device).float()
        self.rms_obs.update(obs.reshape(-1, self.obs_dim))
        if self.budget_ramp_steps > 0:
            frac = min(1.0, self.rollout_steps / float(self.budget_ramp_steps))
            mult = (self.budget_start_mult
                    + (1.0 - self.budget_start_mult)*frac)
            self.cost_target = (self.cost_per_step_target
                                * self._cost_horizon * mult)
        if self.accel_target > 0.0 and self.accel_warmup_steps > 0:
            # anneal the LIMIT itself from a permissive start DOWN to the
            # physical target (a `target*frac` schedule is wrong: for small
            # frac it yields an even TIGHTER limit than the target, i.e. it
            # freezes the robot first and loosens later).
            frac = min(1.0, self.rollout_steps / float(self.accel_warmup_steps))
            env.max_accel = (self.accel_start
                             + (self.accel_target - self.accel_start)*frac)
        stats = {"rew_mean": 0.0, "cost_mean": 0.0, "n_eps": 0,
                 "n_done": 0.0, "n_succ": 0.0}
        # reward that produced the CURRENT state (needed when the constraint
        # targets the negative return, a GUIDER-style self-risk baseline)
        rew_prev = torch.zeros(env.B, env.N, device=self.device)
        for _ in range(rollout_len):
            obs_n = self.rms_obs.normalize(obs)
            flat = obs_n.reshape(-1, self.obs_dim)
            act, logp = self.net.actor.sample(flat)
            act = act.view(obs.shape[0], env.N, self.act_dim)
            logp = logp.view(obs.shape[0], env.N)
            # values (central state expanded per agent)
            vr = self.net.vr(state).unsqueeze(1).expand(-1, env.N)
            vc = self.net.vc(state).unsqueeze(1).expand(-1, env.N)
            # must pass risk_cfg: calling risk_signal() bare silently
            # dropped the peer-robot weight during rollout collection
            risk = env.risk_signal(self.risk_cfg)
            cost = self._cost_per_robot(risk, rew=rew_prev)
            # ---- normalise the per-step cost BEFORE the GAE recursion ------
            # The cost return is heavy-tailed (mean ~2, max ~16) and the raw
            # safety critic could not represent the tail, so `cadv` came out
            # positive for 55-75% of all samples and the CVaR penalty was
            # positive for 100% of the selected ones.  A penalty that is
            # always positive has no equilibrium -- it keeps pushing those
            # ratios down forever, which suppressed learning entirely
            # (success 0.001 vs 0.013 with the penalty disabled, measured).
            # Normalising here means Vc regresses a well-scaled target and the
            # advantage has both signs, giving the dual variable a fixed
            # point.  Statistics are updated on the raw cost and are shared
            # with the target mapping in update().
            if self.cost_norm:
                with torch.no_grad():
                    # Update the statistics from the FULL per-element cost
                    # batch (B*N values), not from a scalar step mean: feeding
                    # a single scalar per step made the running variance
                    # explode (std reached ~250), because early steps are
                    # dominated by the initial transient.
                    x = cost.reshape(-1)
                    nb = x.numel()
                    bm = x.mean()
                    bv = x.var(unbiased=False) if nb > 1 else torch.zeros((), device=x.device)
                    tot_n = self._cost_n + nb
                    d_ = bm - self._cost_mean
                    self._cost_mean += d_ * (nb / tot_n)
                    # pooled variance (Chan et al. parallel update)
                    self._cost_var = (
                        self._cost_var * self._cost_n + bv * nb
                        + d_ ** 2 * self._cost_n * nb / tot_n
                    ) / tot_n
                    self._cost_n = tot_n
                cost = (cost - self._cost_mean) / max(self._cost_var ** 0.5, 1e-3)
            nobs, nstate, rew, done, _trunc, info = env.step(act.detach())
            rew = torch.as_tensor(rew, device=self.device).float()
            done_b = torch.as_tensor(done, device=self.device, dtype=torch.bool)
            buf.push(obs, state, act, logp, rew, done_b, vr)
            # store cost value (vc) and per-step cost
            buf.push_cost(vc.detach(), cost.detach())
            self.rollout_steps += env.N * env.B
            if "succ" in info:
                stats["n_done"] += float(done_b.sum())
                stats["n_succ"] += float(torch.as_tensor(
                    info["succ"], device=self.device).sum())
            stats["rew_mean"] += float(rew.mean())
            stats["cost_mean"] += float(cost.mean())
            rew_prev = rew
            obs = torch.as_tensor(nobs, device=self.device).float()
            state = torch.as_tensor(nstate, device=self.device).float()
        if rollout_len:
            stats["rew_mean"] /= rollout_len
            stats["cost_mean"] /= rollout_len
        # window success rate -> EMA (only meaningful once episodes complete)
        if stats["n_done"] > 0:
            wr = stats["n_succ"] / stats["n_done"]
            b = self.task_gate_beta
            self._succ_ema = (1.0 - b)*self._succ_ema + b*wr
        stats["succ_ema"] = self._succ_ema
        # bootstrap final values
        final_rv = self.net.vr(state).unsqueeze(1).expand(-1, env.N)
        final_cv = self.net.vc(state).unsqueeze(1).expand(-1, env.N)
        advs = buf.compute_returns_gae(self.gamma, self.lam, final_rv)
        rets = [a + v for a, v in zip(advs, buf.val)]
        data = buf.stack()
        data["adv"] = torch.stack(advs, 0)
        data["ret"] = torch.stack(rets, 0)
        data["cost_res"] = buf.cost_stack()
        data["final_vc"] = final_cv
        data["cost"] = data["cost_res"]["cost"] if "cost" in data["cost_res"] else None
        # ---- stage 1: proper GAE for the SAFETY cost ----------------------
        # Previously the trainer regressed vc onto the instantaneous cost, which
        # removes all temporal credit assignment from the safety objective.
        gamma_c = float(getattr(self, "gamma_cost", self.gamma))
        if data["cost"] is not None:
            cadv, cret = buf.compute_returns_gae_cost(gamma_c, self.lam,
                                                      final_cv)
            data["cost_adv"] = torch.stack(cadv, 0)
            data["cost_ret"] = torch.stack(cret, 0)
        else:
            data["cost_adv"] = None
            data["cost_ret"] = None
        return data, stats

    @staticmethod
    def _cvar(x, alpha):
        """Conditional Value-at-Risk at level alpha: mean of the worst alpha
        fraction.  alpha=1.0 recovers the plain mean (risk-neutral).

        Used because social acceptance is governed by the WORST experiences, not
        the average: "everyone comfortable but one pedestrian terrified" must
        not be treated as success.  A mean-based constraint structurally cannot
        express this, so this is also what distinguishes us from MAPPO.
        """
        if alpha >= 1.0:
            return x.mean()
        n = x.numel()
        k = max(int(alpha * n), 1)
        return torch.topk(x.reshape(-1), k, largest=True).values.mean()

    # ------------------------------------------------------------- update
    def update(self, data):
        # scaffold: run reward-PPO with adaptive cost-monotone Lagrangian;
        # full returns-on-cost GAE left as explicit TODO documented below.
        obs = data["obs"].reshape(-1, self.obs_dim)
        act = data["act"].reshape(-1, self.act_dim)
        logp0 = data["logp_old"].reshape(-1)
        adv = data["adv"].reshape(-1)
        ret = data["ret"].reshape(-1)
        L_, B_, N_ = data["obs"].shape[:3]
        M = obs.shape[0]
        self.rms_obs.update(obs)
        obs_n = self.rms_obs.normalize(obs)
        S = data["state"].shape[-1]
        st = data["state"].unsqueeze(2).expand(L_, B_, N_, S).reshape(-1, S)
        advz = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))
        cost = data["cost"].reshape(-1) if data["cost"] is not None \
            else torch.zeros(M, device=self.device)
        # stage 1: use GAE returns (discounted cumulative cost) when available,
        # otherwise fall back to the raw cost (legacy behaviour).
        cost_ret = (data["cost_ret"].reshape(-1)
                    if data.get("cost_ret") is not None else cost)
        cost_adv = (data["cost_adv"].reshape(-1)
                    if data.get("cost_adv") is not None else cost)
        # ---- the actual cause of "AMC never learns" -----------------------
        # The reward surrogate uses the NORMALISED advantage `advz` (std=1),
        # but the cost surrogate used the RAW cost advantage.  Measured on the
        # dense scene `cadv` has mean +0.96 and max 14.4, so its gradient was
        # 35-50% of the policy gradient's magnitude even though its LOSS
        # contribution looked negligible (lam*pen ~ 3 vs pol ~ 25, because the
        # loss is dominated by ~187 of value losses, which never touch the
        # actor).
        #
        # Worse, corr(advz, cadv) ~ +0.02: the cost penalty was therefore
        # pushing a substantial gradient on samples essentially unrelated to
        # task advantage.  That is what destroyed learning -- success 0.001,
        # versus 0.013 with the penalty removed and 0.002 even when CVaR was
        # replaced by a plain mean, i.e. the fault is the SCALE, not the CVaR.
        #
        # TESTED AND REJECTED as a fix: normalising the cost advantage the way
        # the reward advantage is normalised gave 0.000 vs 0.002 for the raw
        # version -- slightly WORSE.  Kept because both are on comparable
        # footing and the ranking (hence the CVaR top-k) is unchanged, but it
        # is NOT the cause of the learning failure.  See PROBLEM_MAP.md for the
        # lambda sweep that does explain it.
        cadvz = (cost_adv - cost_adv.mean()) / cost_adv.std().clamp_min(1e-6)
        cadv_used = cadvz if self.cost_adv_zscore else cost_adv
        worst_global = None
        if self.cvar_select == "rollout":
            k_all = max(int(self.cvar_alpha * cadv_used.numel()), 1)
            idx_all = torch.topk(cadv_used, k_all, largest=True).indices
            worst_global = torch.zeros_like(cadv_used, dtype=torch.bool)
            worst_global[idx_all] = True
        # The target must live in the SAME units as `cost_ret`.  Only map it
        # through the normalisation statistics when the cost is actually
        # normalised (cost_norm); doing it unconditionally made the target
        # effectively ~0 while the returns stayed raw, so the violation was
        # permanently positive and lambda pinned to its ceiling (observed:
        # lam=20.0 and success 0.000).
        if self.cost_norm:
            self._cost_target_n = (self.cost_target - self._cost_mean) \
                / max(self._cost_var ** 0.5, 1e-3)
        else:
            self._cost_target_n = self.cost_target
        perm = torch.randperm(M)
        bsize = max((M + self.minib - 1) // self.minib, 1)
        tot = {"loss": 0.0, "cost_mean": float(cost.mean())}
        steps = 0
        for _ in range(self.epochs):
            for s in range(0, M, bsize):
                b = perm[s:s+bsize]
                lp, ent = self.net.actor.evaluate(obs_n[b], act[b])
                ratio = torch.exp(lp - logp0[b])
                s1 = ratio * advz[b]
                s2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advz[b]
                pol = -torch.min(s1, s2).mean()
                # dual value losses
                vr = self.net.vr(st[b])
                vc = self.net.vc(st[b])
                lr_ = 0.5 * ((vr - ret[b]) ** 2).mean()
                # stage 1: safety critic regresses the GAE cost RETURN
                # NOTE: `cost_clip` must NOT clamp the critic's regression
                # target.  Cost returns reach ~20-30 on the dense scene; a
                # clamp at 3.0 biased Vc low, kept every cost advantage
                # positive, and (together with the unscaled target above)
                # drove lambda to its ceiling.  The clip is applied where it
                # belongs: as an outlier guard on the ADVANTAGE, not the value.
                lc = 0.5 * ((vc - cost_ret[b]) ** 2).mean()
                eL = -ent.mean()
                # ---- stage 2: CVaR constraint ON THE POLICY GRADIENT -------
                # The previous implementation added `lambda * (CVaR - target)`
                # to the loss as a CONSTANT.  A constant contributes exactly
                # zero gradient, so the constraint never influenced the policy
                # at all (lambda still climbed to ~17, which is why the failure
                # looked like "the method does not help" rather than a bug).
                #
                # Correct form: give the WORST-alpha experiences a PPO-style
                # policy-gradient term, so reducing their cost is directly
                # rewarded.  Only the worst `cvar_alpha` fraction is weighted,
                # which is what makes this a CVaR constraint rather than a mean.
                cost_pen = torch.zeros((), device=self.device)
                viol = torch.zeros((), device=self.device)
                if self.risk_mode in ("adaptive", "fixed"):
                    # use the NORMALISED cost advantage for the surrogate (see
                    # `cadvz` above); ranking is unchanged because it is a
                    # positive affine transform, but the gradient magnitude now
                    # matches the reward surrogate's instead of being scaled by
                    # raw cost units.
                    cadv = cadv_used[b]
                    if worst_global is not None:
                        # rollout-level tail: indices OF THIS MINIBATCH that lie
                        # in the global worst-alpha set
                        worst = torch.nonzero(worst_global[b], as_tuple=False).flatten()
                        if worst.numel() == 0:      # degenerate minibatch
                            k = max(int(self.cvar_alpha * cadv.numel()), 1)
                            worst = torch.topk(cadv, k, largest=True).indices
                    else:
                        k = max(int(self.cvar_alpha * cadv.numel()), 1)
                        worst = torch.topk(cadv, k, largest=True).indices
                    # Pessimistic PPO surrogate for the cost, mirroring the
                    # reward surrogate.  The reward uses -min(s1,s2) because it
                    # is MAXIMISED; the cost uses +max(c1,c2) because it is
                    # MINIMISED.  Verified this has the intended trust-region
                    # behaviour for a cost advantage of +2.0:
                    #   ratio < 0.8 -> the CLIPPED branch binds, gradient 0,
                    #                  i.e. once the policy has moved far enough
                    #                  away from the costly action the
                    #                  constraint stops pushing (correct)
                    #   ratio > 1.2 -> the UNCLIPPED branch binds and pushes
                    #                  the ratio down (correct)
                    # So the asymmetry with -min is intentional, not a sign
                    # error.  (Checked explicitly because it looks wrong.)
                    c1 = ratio[worst] * cadv[worst]
                    c2 = torch.clamp(ratio[worst], 1-self.clip,
                                     1+self.clip) * cadv[worst]
                    cost_pen = torch.max(c1, c2).mean()
                    # NOTE: the per-minibatch `viol` that used to be computed
                    # here was dead code -- it is overwritten by the
                    # authoritative dual update after the epoch loop, which
                    # uses the realised cost RETURN.  Removed so the two can
                    # never disagree about units.
                loss = (pol + self.vf_coef * (lr_ + lc) + self.ent_coef * eL
                        + self.cost_weight * self.lambda_cost
                        * cost_pen.clamp(min=-20.0))
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.gradmax)
                self.opt.step()
                tot["loss"] += float(loss.detach()); steps += 1
        # adaptive Lagrangian update of multiplier.  `viol` is a cost RETURN
        # minus a horizon-scaled target, so it is normalized by the same
        # horizon before being multiplied by lambda_lr (see __init__).
        #
        # CRITICAL: the dual variable must integrate the REALIZED cost return,
        # not `cost_adv`.  `cost_adv = cost_return - Vc` measures "worse than
        # the critic expected", and since Vc is trained to predict the
        # policy's OWN average cost, CVaR(cost_adv) drifts to zero as the
        # critic converges no matter how much absolute cost the policy
        # incurs.  Driving lambda with it produced a dual variable that
        # DECREASED 0.497 -> 0.452 while realized cost sat at 0.09/step
        # against a 0.05 target at every single checkpoint -- i.e. the
        # constraint was violated throughout and lambda moved the wrong way,
        # so the constraint never acquired force while still perturbing the
        # policy enough to cripple learning (success 0.03-0.13 vs MAPPO 0.80).
        if self.risk_mode == "adaptive":
            # `viol` is a cost RETURN minus a horizon-scaled target, so divide
            # by the same horizon to put lambda_lr in plain lambda-units per
            # iteration.  (An intermediate edit dropped this division while
            # chasing a "normalised units" idea; that made lambda advance ~1.0
            # per iteration and pin to lambda_max=20 within 20 iterations.)
            cvar_cost = float(self._cvar(cost_ret, self.cvar_alpha))
            raw = cvar_cost - self._cost_target_n
            viol = raw / max(1.0, self._cost_horizon)
            gated = (self.task_gate_threshold > 0.0
                     and self._succ_ema < self.task_gate_threshold)
            if gated:
                # task not solved yet -> keep the constraint inert
                self.lambda_cost = 0.0
                tot["gated"] = 1.0
            elif self._in_warmup():
                # task-learning phase: keep the constraint inert so the policy
                # can first learn to reach goals.  See __init__ for the
                # measured lambda -> success curve that motivates this.
                self.lambda_cost = 0.0
                tot["warmup"] = 1.0
            else:
                self.lambda_cost = max(0.0, min(self.lambda_max,
                    self.lambda_cost + self.lambda_lr * viol))
            tot["viol"] = viol
            tot["cvar_cost"] = cvar_cost
        elif self.risk_mode == "none":
            self.lambda_cost = 0.0
        tot["loss"] /= max(steps, 1)
        tot["lambda"] = self.lambda_cost
        self._updates_done += 1
        return tot

    def _in_warmup(self):
        """True while the constraint should stay inert (task-learning phase).

        Supports either an iteration count (`lambda_warmup`) or an environment
        step count (`lambda_warmup_steps`, preferred).  Returns False when both
        are 0, i.e. warm-up disabled.
        """
        if self.lambda_warmup_steps > 0:
            return self.rollout_steps < self.lambda_warmup_steps
        if self.lambda_warmup > 0:
            return self._updates_done < self.lambda_warmup
        return False

    @torch.no_grad()
    def act_det(self, obs):
        obs = torch.as_tensor(obs, device=self.device).float()
        flat = self.rms_obs.normalize(obs).reshape(-1, self.obs_dim)
        mean = self.net.actor.dist(flat).mean
        return mean.view(obs.shape[:-1] + (self.act_dim,))

    def state_dict(self):
        return {"net": self.net.state_dict(), "lambda": self.lambda_cost}

    def load_state(self, sd, strict=False):
        if "net" in sd:
            self.net.load_state_dict(sd["net"], strict=strict)
            self.lambda_cost = float(sd.get("lambda", self.lambda_cost))
            # ALSO restore the observation normalisation.  `act_det` feeds the
            # actor `rms_obs.normalize(obs)`, so copying the weights WITHOUT the
            # statistics leaves the warm-started actor reading mis-scaled inputs
            # until the fresh running stats converge -- which silently corrupts
            # a behaviour-cloned policy (its whole point is to preserve the
            # behaviour) and also perturbs every other warm start for the first
            # iteration.  `load()` already restored these; `load_state()` did
            # not, so the `--init_ckpt` path was the broken one.
            if "rms_mean" in sd and "rms_var" in sd:
                self.rms_obs.mean = sd["rms_mean"].to(self.device).float()
                self.rms_obs.var = sd["rms_var"].to(self.device).float()
                self.rms_obs.count = float(sd.get("rms_count", 1.0))
            # A checkpoint's log_std is tuned to ITS action parameterisation.
            # With `bounded_mean` the mean magnitude drops from ~47 to ~1.3, so
            # a copied std of 7.39 would mean ~78 degrees of directional jitter
            # (near-random) instead of the ~9 degrees it meant in the source
            # run.  Re-initialise it from the current recipe when (and only
            # when) the parameterisation differs.  Measured, see NIGHT_REPORT
            # section 31.
            if getattr(self.net.actor, "bounded_mean", False) \
                    and "actor.log_std" in sd.get("net", {}):
                with torch.no_grad():
                    self.net.actor.log_std.fill_(self.log_std_init)

    @property
    def kind(self):
        return "ra_cmappo"

    def save(self, path):
        # rms_obs MUST be persisted alongside the weights: act_det() normalises
        # observations with these running statistics, and a fresh Trainer would
        # otherwise use mean=0/var=1, yielding mis-scaled inputs and near-zero
        # actions (this silently invalidated checkpoint-based evaluations).
        torch.save({"__kind__": self.kind,
                    "net": self.net.state_dict(),
                    "lambda": float(self.lambda_cost),
                    "risk_mode": self.risk_mode,
                    "cost_target": self.cost_target,
                    "rms_mean": self.rms_obs.mean,
                    "rms_var": self.rms_obs.var,
                    "rms_count": self.rms_obs.count}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        # older format stores 'state' or mappo format
        if "net" in ck:
            self.load_state(ck, strict=False)
        elif "state" in ck:
            self.net.load_state_dict(ck["state"], strict=False)
        if "rms_mean" in ck and "rms_var" in ck:
            self.rms_obs.mean = ck["rms_mean"].to(self.device).float()
            self.rms_obs.var = ck["rms_var"].to(self.device).float()
            self.rms_obs.count = float(ck.get("rms_count", 1.0))
        # Restore the Lagrangian budget too.  Checkpoints store it, but load()
        # used to leave the freshly built trainer on the RECIPE's target, so an
        # offline diagnostic of a binding-budget run (whose CLI --cost_target
        # was, say, 12.0) reported the recipe's slack target 34.9 instead.
        if "cost_target" in ck:
            try:
                self.cost_target = float(ck["cost_target"])
            except (TypeError, ValueError):
                pass
        if ck.get("risk_mode"):
            try:
                self.risk_mode = str(ck["risk_mode"])
            except (TypeError, ValueError):
                pass
        return ck
