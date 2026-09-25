"""social_nav.py

Vectorised 2D continuous multi-robot social-navigation environment.
All simulation is performed on one torch device, vectorised over
B = ``n_parallel_envs`` independent episodes (shape prefix B).

Robot actuator: continuous desired 2-D world-frame velocity per control step.
Robots are homogeneous circles each with its own start/goal.

Pedestrians (P) use a simple deterministic "rule / social-force" walker with
goal-pull and robot & pedestrian separation.  Static circular obstacles (O)
provide basic environment clutter.  Weeks 2/3 extend the reward; week 1 keeps
the kinematic-terms-only baseline.

Boundary contract (torch on self.device):
    reset()                    ->  (obs, global_state)
    step(actions (B,N,2))      ->  (obs, global_state, rew(B,N),
                                    done(B), trunc(B), info)

Upon terminal (done or trunc) episodes are internally re-spawned with a fresh
layout so a vectorised runner never stalls; done/trunc remain exposed so
PPO/GAE can bootstrap correctly at the boundary.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class SocialNavVecEnv:

    def __init__(self, cfg, n_parallel_envs=None, device=None, base_seed=0):
        e = cfg.env
        self.cfg = cfg
        self.N = int(e.n_agents)
        self.P = int(e.n_pedestrians)
        self.O = int(e.n_obstacles)
        self.size = float(e.size)
        self.r = float(e.robot_radius)
        self.ped_r = float(e.ped_radius)
        self.ob_r = float(e.obstacle_radius)
        self.max_sp = float(e.max_robot_speed)
        self.ped_sp = float(e.pedestrian_max_speed)
        # --- emotion model config (E-SocialNav) --------------------------
        # alpha: mood decay back to neutral; beta: intrusion sensitivity;
        # delta: courtesy (robot yielding) sensitivity.  The anisotropic radii
        # reuse the RA risk-field geometry so the two stay consistent.
        self.emo = {
            "enabled": bool(getattr(e, "emotion_enabled", False)),
            "alpha": float(getattr(e, "mood_alpha", 0.35)),
            "beta": float(getattr(e, "mood_beta", 1.8)),
            "delta": float(getattr(e, "mood_delta", 0.6)),
            "d_front": float(getattr(e, "mood_d_front", 1.0)),
            "d_side": float(getattr(e, "mood_d_side", 0.6)),
            "d_back": float(getattr(e, "mood_d_back", 0.4)),
            "feedback": bool(getattr(e, "emotion_feedback", True)),
            "avoid_gain": float(getattr(e, "mood_avoid_gain", 2.0)),
            # ---- three-layer impact model (see _update_mood) --------------
            # q_t = w_I*intrusion + w_V*closing + w_M*relu(-mood)
            "impact_on": bool(getattr(e, "impact_on", True)),
            "w_I": float(getattr(e, "impact_w_intrusion", 1.0)),
            "w_V": float(getattr(e, "impact_w_closing", 0.25)),
            "w_M": float(getattr(e, "impact_w_mood", 0.5)),
            # responsibility-attribution temperatures
            "sigma_d": float(getattr(e, "resp_sigma_dist", 0.5)),
            "sigma_v": float(getattr(e, "resp_sigma_vel", 0.5)),
            "sigma_i": float(getattr(e, "resp_sigma_intr", 0.5)),
        }
        # "distance" (baseline) or "dynamic" (distance + closing + intrusion)
        self.resp_mode = str(getattr(e, "resp_mode", "dynamic"))
        self.w_emo = float(getattr(e, "emotion_reward_weight", 0.0))
        # "mean" (legacy, /P-diluted) or "tail" (same per-robot cost the CVaR
        # constraint uses) -- see config.py for why this exists.
        self.emo_shaping_mode = str(getattr(e, "emotion_shaping_mode", "mean"))
        # 0.0 == pedestrians never yield to robots (hardest, most reactive crowd)
        self.ped_avoid_robot = float(getattr(e, "ped_avoid_robot", 2.2))
        # P3: how strongly affect feeds back into observable behaviour
        self.mood_slow = float(getattr(e, "mood_slow", 0.5))    # speed reduction
        self.mood_dodge = float(getattr(e, "mood_dodge", 1.0))  # dodge force gain
        self.mood_turn = float(getattr(e, "mood_turn", 2.5))    # turn-to-face rate
        # P5: lower, data-driven distress threshold (mood only reached ~-0.1)
        self.wmp_thresh = float(getattr(e, "wmp_thresh", -0.2))
        # P4: slower mood decay so affect can accumulate within an episode
        # (alpha=0.35 -> tau~2.9 s, shorter than a 5 s episode)
        self.emo["alpha"] = float(getattr(e, "mood_alpha", 0.06))
        # P6: pedestrian robot-avoidance gain (2.2 == 4.4x walking speed, which
        # made the crowd dodge chaotically and drowned out robot behaviour)
        self.ped_avoid_robot = float(getattr(e, "ped_avoid_robot", 0.0))
        # social-cost formulation used by the CVaR constraint:
        #   "encroachment" (default) severity of the CURRENT personal-space
        #                  violation -- gives a genuine safety/efficiency
        #                  trade-off (detouring costs 0.021 vs 0.187 for
        #                  going straight through the crowd)
        #   "deficit"      legacy accumulated-affect cost, retained so the
        #                  two can be compared in an ablation
        self.cost_mode = str(getattr(e, "cost_mode", "encroachment"))
        # ---- analytic yielding prior (default 0 = OFF, legacy) ------------
        # Adds the scripted controller's pedestrian-repulsion term to the
        # robot's commanded velocity, so the learned policy acts as a RESIDUAL
        # on top of a behaviour that provably reaches the task/affect frontier
        # (scripted detour: success -3.6%, CVaR -27%).
        self.detour_prior_gain = float(getattr(e, "detour_prior_gain", 0.0))
        self.detour_prior_radius = float(getattr(e, "detour_prior_radius", 1.5))
        self.goal_tol = float(e.goal_tolerance)
        self.coll_rr = float(e.collision_dist_robot_robot)
        self.coll_rp = float(e.collision_dist_robot_ped)
        self.action_dt = float(e.action_dt)
        self.horizon = int(round(float(e.max_episode_seconds)/self.action_dt))

        # optional deterministic scene: "cross" uses a fixed crossing layout
        self.scene = getattr(e, "scene", None)
        # auto-reset on terminal (train). Evaluation keeps episodes open so each
        # episode can be measured cleanly.
        self.auto_reset = bool(getattr(e, "auto_reset", True))

        # --- evaluation hooks (all default OFF; existing behaviour unchanged) ---
        # when a tensor is supplied via set_ped_override(), pedestrians move with
        # that velocity instead of the built-in social-force walker.  This lets an
        # external planner (e.g. true reciprocal ORCA) drive the whole crowd.
        self._ped_override = None
        # observation noise std (added to get_obs output) and control delay steps
        self.obs_noise = float(getattr(e, "obs_noise", 0.0))
        self.control_delay = int(getattr(e, "control_delay", 0))
        self._act_queue = []

        # --- realism knobs (plan_realism.md) ------------------------------
        # ALL of these default to the legacy behaviour, so any existing recipe
        # reproduces its numbers bit-for-bit until it opts in.
        #
        # (1) per-pedestrian free walking speed.  Legacy: one scalar for the
        #     whole crowd (pedestrian_max_speed), which is not a crowd -- real
        #     pedestrian streams have a speed distribution (mean free walking
        #     speed ~1.2-1.34 m/s, spread ~0.2-0.3).  Set ped_speed_std > 0 to
        #     sample each pedestrian per episode from N(mean, std) clipped to
        #     [ped_speed_min, ped_speed_max].
        h_mean = getattr(e, "ped_speed_mean", None)
        h_std = float(getattr(e, "ped_speed_std", 0.0))
        self.ped_sp_std = h_std
        self.ped_sp_mean = float(h_mean) if h_mean is not None else self.ped_sp
        lo = getattr(e, "ped_speed_min", None)
        hi = getattr(e, "ped_speed_max", None)
        self.ped_sp_lo = float(lo) if lo is not None else max(0.2, self.ped_sp_mean - 3*h_std)
        self.ped_sp_hi = float(hi) if hi is not None else max(self.ped_sp_mean + 3*h_std,
                                                              self.ped_sp)
        # (2) willingness to yield to robots, sampled per pedestrian.  Legacy:
        #     one scalar ped_avoid_robot for everyone (amc_mid uses 0.0, i.e.
        #     nobody ever yields -- the hardest, least realistic case).
        #     Real crowds mix oblivious / partial / fully-yielding people.
        self.ped_yield_levels = [float(v) for v in
                                 (getattr(e, "ped_yield_levels", None) or [])]
        self.ped_yield_probs = [float(v) for v in
                                (getattr(e, "ped_yield_probs", None) or [])]
        # (3) stationary workers vs circulating walkers.  Legacy: everybody
        #     walks to one goal and then stops there forever.
        #     ped_worker_frac = fraction that stays at its workstation all
        #     episode; ped_circulate = walkers pick a NEW goal on arrival.
        self.ped_worker_frac = float(getattr(e, "ped_worker_frac", 0.0))
        self.ped_circulate = bool(getattr(e, "ped_circulate", False))
        # how far a workstation occupant may be nudged before it braces
        self.ped_worker_radius = float(getattr(e, "ped_worker_radius", 0.35))
        # keep pedestrian spawns clear of the robots' GOALS (see _spawn_peds).
        # Default False = legacy behaviour, so historical runs stay reproducible.
        self.ped_clear_goals = bool(getattr(e, "ped_clear_goals", False))
        # --- physical plausibility (see scripts/check_physics.py) ----------
        # hard_separation: geometrically resolve overlaps after each step
        #   instead of only penalising them.  Legacy False = agents may
        #   interpenetrate (a policy could drive through a person).
        self.hard_separation = bool(getattr(e, "hard_separation", False))
        # collision_terminal: a robot-pedestrian / robot-robot / robot-obstacle
        # CONTACT ends the episode as a failure.  Default False = legacy (in this
        # project nothing ever set `crashed`, so contacts were free and the
        # oblivious "walk straight at full speed" controller had the HIGHEST
        # success rate -- measured 0.815 vs 0.812 for MAPPO).  With this on,
        # avoiding people becomes necessary for the task metric itself, which is
        # the standard convention in social-navigation papers.
        self.collision_terminal = bool(getattr(e, "collision_terminal", False))
        # ped_steer_mode: "add" (legacy) adds an UNBOUNDED repulsion vector to
        #   the walking velocity -- at close range neighbours cancel the desired
        #   direction and the crowd jitters (measured: 49% of moving steps had a
        #   >90 deg heading flip and 49% a speed reversal).  "blend" instead
        #   blends directions and re-normalises, so the speed can never exceed
        #   the pedestrian's own walking speed and the heading changes smoothly.
        self.ped_steer_mode = str(getattr(e, "ped_steer_mode", "add"))
        # ped_worker_activity: amplitude (m) of a slow local motion for the
        #   stationary workers, so they are not frozen statues. 0 = legacy.
        self.ped_worker_activity = float(getattr(e, "ped_worker_activity", 0.0))
        self.ped_worker_period = float(getattr(e, "ped_worker_period", 12.0))
        # --- perception into the ACTOR observation ------------------------
        # The plain observation is [relative position, relative velocity, mood]
        # per pedestrian, which cannot express "someone else is heading for my
        # destination".  Measured consequence: a robot approaching its goal did
        # not slow down at all when a pedestrian was closing on the same point
        # (0.438 m/s vs 0.400 m/s otherwise -- it was FASTER).  Both features
        # below are computed from observable state only (linear prediction), so
        # the actor gets no oracle intent.
        self.anticipate_in_obs = bool(getattr(e, "anticipate_in_obs", False))
        self.contention_in_obs = bool(getattr(e, "contention_in_obs", False))
        self.perception_k = int(getattr(e, "perception_k", 16) or 16)
        # --- velocity-level non-penetration at contacts --------------------
        # Position projection alone leaves the agent pushing into whatever it
        # touched (measured: 68.5% of contact steps still drove INTO the
        # obstacle), which renders as grinding past instead of stopping or
        # going around.
        self.contact_stop = bool(getattr(e, "contact_stop", False))
        # ---- failure-augmented cost / reward (plan_emo.md 模块2) -----------
        # kappa_f: added to the CONSTRAINED cost once, at the step a robot
        # fails (timeout or collision).  Without it the constraint's optimum is
        # "never move" (D=0) and lambda drives the policy there.
        # rew_timeout: the matching reward-side penalty.
        self.cost_fail_penalty = float(getattr(e, "cost_fail_penalty", 0.0))
        self.rew_timeout = float(getattr(cfg.reward, "timeout_penalty", 0.0))
        # (4) acceleration limit.  Legacy: the commanded velocity is assigned
        #     directly (infinite acceleration), so "sudden stop / oscillation"
        #     metrics have no physical meaning.  max_robot_accel in m/s^2.
        self.max_accel = float(getattr(e, "max_robot_accel", 0.0))
        # (5) ERM: robot affect (plan_realism.md §7).  Legacy: disabled.
        #     a_i in [-1,1] (+1 calm, -1 stressed), driven by blocking, space
        #     pressure, courtesy received and time urgency -- the same
        #     construction as the pedestrian mood model, but the pressure and
        #     courtesy terms are summed over BOTH pedestrians and co-robots, so
        #     robot affects are coupled to each other (affect contagion).
        self.robot_emo = {
            "enabled": bool(getattr(e, "robot_emotion_enabled", False)),
            "alpha": float(getattr(e, "robot_mood_alpha", 1.0)),    # recovery (tau~1 s)
            "beta": float(getattr(e, "robot_mood_beta", 1.0)),      # blocking
            "gamma": float(getattr(e, "robot_mood_gamma", 0.3)),    # pressure (sums over agents)
            "delta": float(getattr(e, "robot_mood_delta", 0.5)),    # courtesy
            "eta": float(getattr(e, "robot_mood_eta", 0.3)),        # urgency
            "in_obs": bool(getattr(e, "robot_mood_in_obs", True)),
            "pressure_r": float(getattr(e, "robot_pressure_radius", 0.8)),
        }

        rw = cfg.reward
        self.w_pot = float(rw.progress_omega)
        self.rew_reach = float(rw.reach_bonus)
        self.rew_step = -float(rw.step_penalty)
        self.rew_rr = float(rw.collision_robot_robot)
        self.rew_rp = float(rw.collision_robot_ped)
        self.rew_ob = float(rw.collide_with_obstacle)
        # week-3 social reward knobs (kept 0 -> term disabled)
        self.w_discom = float(rw.get("discomfort_weight", 0.0))
        self.d_soc = float(rw.get("discom_sigma", 0.45))     # comfort radius m
        self.smooth = float(rw.get("discom_smooth", 1.0))
        self.w_ttc = float(rw.get("ttc_weight", 0.0))
        self.ttc_tau = float(rw.get("ttc_tau", 0.8))
        self.near_t = float(rw.get("near_dist", self.r+self.ped_r+0.45))
        # soft-close distance where discomfort becomes maximal
        self.d_close = max(self.d_soc, self.r + self.ped_r + 0.15)

        dev = torch.device(device if device else cfg.vec.device)
        if str(dev.type).startswith("cuda") and not torch.cuda.is_available():
            dev = torch.device("cpu")
        self.device = dev
        self.B = int(cfg.vec.n_parallel_envs) if n_parallel_envs is None \
            else int(n_parallel_envs)
        self.base_seed = int(base_seed)
        self._rng = np.random.default_rng(self.base_seed)
        self._n_resets = 0

        # capacity scheme (opt-in): fixed observation/state size independent of
        # scene cardinality (n_robots/n_peds/n_obs) so one network can eval a
        # range of densities/robot counts. Unset -> exact scene dims.
        # anticipatory safety features (innovation 3) add 2 values per robot
        # slot to the CENTRAL STATE only (the critic sees them; the actor does
        # not, preserving decentralised execution).
        self._anticipate_k = int(getattr(e, "anticipate_k", 0) or 0)
        # --- P1: expose pedestrian mood in the robot's observation ----------
        # Without this the policy cannot perceive affect at all, so any claim
        # of "emotion-aware" decision making would be unsupported: the previous
        # observation contained position/velocity only.
        self._mood_obs = bool(getattr(e, "emotion_enabled", False)
                              and getattr(e, "mood_in_obs", True))
        self._ped_slot = 5 if self._mood_obs else 4
        # ABLATION: keep the mood slots in the observation LAYOUT but always
        # feed 0.  Why not just set mood_in_obs=False: that changes the input
        # dimension, so an actor trained with mood cannot be warm-started
        # (load_state(strict=False) would silently leave the first layer
        # random, turning a 700-iteration ablation into a from-scratch run).
        # Default False = legacy behaviour, bit-for-bit.
        self.mood_obs_zero = bool(getattr(e, "mood_obs_zero", False))
        # ABLATION switches (default False = legacy, layout preserved so a
        # trained actor stays loadable; see mood_obs_zero for the rationale).
        #   ped_vel_zero : the pedestrian's relative velocity channel -> 0
        #   rmood_obs_zero: the robot's own ERM affect channel -> 0
        self.ped_vel_zero = bool(getattr(e, "ped_vel_zero", False))
        self.rmood_obs_zero = bool(getattr(e, "rmood_obs_zero", False))
        # ERM: the robot's own affect is part of its observation (information
        # parity across arms -- the constraint, not the information, is what
        # the method adds).
        self._rmood_slot = 1 if (self.robot_emo["enabled"]
                                 and self.robot_emo["in_obs"]) else 0
        self._own_dim = (5 + self._rmood_slot
                         + (2 if self.anticipate_in_obs else 0)
                         + (2 if self.contention_in_obs else 0))
        cap = getattr(e, "obs_capacity", None) or {}
        self.cap_r = int(cap.get("robots", 0) or 0)
        self.cap_p = int(cap.get("peds", 0) or 0)
        self.cap_o = int(cap.get("obstacles", 0) or 0)
        if (self.cap_r or self.cap_p or self.cap_o):
            self._cap_mode = True
            nr = max(self.N, self.cap_r)
            np_ = max(self.P, self.cap_p)
            no_ = max(self.O, self.cap_o)
            self._obs_dim = (self._own_dim + (nr - 1 + no_) * 4
                             + np_ * self._ped_slot)
            self._state_dim = (6 * nr + 4 * np_ + 2 * no_
                               + (2 * self.N if self._anticipate_k else 0)
                               + (self.N if self.robot_emo["enabled"] else 0))
        else:
            self._cap_mode = False
            self._obs_dim = (self._own_dim + (self.N - 1 + self.O) * 4
                             + self.P * self._ped_slot)
            self._state_dim = (6 * self.N + 4 * self.P + 2 * self.O
                               + (2 * self.N if self._anticipate_k else 0)
                               + (self.N if self.robot_emo["enabled"] else 0))
        self._alloc()

    # ------------------------------------------------------------------ alloc
    def _alloc(self):
        z = torch.zeros
        d = self.device
        b, n, p, o = self.B, self.N, self.P, self.O
        self.robo_pos = z(b, n, 2, device=d)
        self.robo_vel = z(b, n, 2, device=d)
        self.robo_goal = z(b, n, 2, device=d)
        self.ped_pos = z(b, p, 2, device=d)
        self.ped_vel = z(b, p, 2, device=d)
        self.ped_goal = z(b, p, 2, device=d)
        self.ped_heading = z(b, p, device=d)
        # --- realism traits (per-episode; see _sample_ped_traits) ----------
        # Initialised to the nominal/legacy values so that a build without an
        # explicit sampling call behaves exactly like the old constant-crowd
        # environment (speed = pedestrian_max_speed, yield = ped_avoid_robot).
        self.ped_sp_vec = torch.full((b, p), float(self.ped_sp), device=d)
        self.ped_yield_vec = torch.full((b, p), float(self.ped_avoid_robot),
                                       device=d)
        self.ped_is_worker = torch.zeros(b, p, dtype=torch.bool, device=d)
        # workstation anchor + phase for the workers' local activity
        self.ped_home = z(b, p, 2, device=d)
        self.ped_worker_phase = z(b, p, device=d)
        # --- ERM: robot affect state (plan_realism.md §7) ------------------
        self.robo_mood = z(b, n, device=d)          # a_i in [-1, 1]
        self.robo_stress = z(b, n, device=d)        # accumulated dose S
        self._rm_prev_dist = None                   # progress reference
        self._rm_prev_rr = None                     # robot-robot distances
        self._rm_prev_rp = None                     # robot-pedestrian distances
        self._rm_last = {}                          # driver decomposition
        # --- emotion state (E-SocialNav) ---------------------------------
        # per-pedestrian mood in [-1, 1]: +1 comfortable, -1 distressed.
        # It is a low-pass filtered integral of the social intrusion robots
        # inflict, minus a natural decay back to neutral.  See
        # results/EMOTION_TASK_DESIGN.md.
        self.ped_mood = z(b, p, device=d)
        self.ped_mood_sum = z(b, p, device=d)      # time-integral (for MPI)
        self.ped_mood_n = z(b, p, device=d)        # sample count
        self.ped_mood_min = z(b, p, device=d)      # worst mood seen
        # P5: cumulative affective dose -- threshold-free, so it stays sensitive
        # even when nobody crosses an arbitrary cutoff (the old WMP at -0.5 was
        # effectively binary because mood only ever reached ~-0.1).
        self.ped_mood_dose = z(b, p, device=d)
        # three-layer decomposition (see _update_mood): `ped_mood_dose` above
        # is the STATE layer; these two are the IMPACT (constraint cost) and
        # EXPOSURE layers.  Kept separate so the double-integration artefact
        # can never silently become the training objective again.
        self.ped_dose_impact = z(b, p, device=d)    # sum q dt  <- constraint
        self.ped_dose_exposure = z(b, p, device=d)  # sum I dt  <- explanatory
        self.ped_q = z(b, p, device=d)              # instantaneous harm q_t
        self.ped_mood_bad = z(b, dtype=torch.long, device=d)  # steps spent < -0.5
        self.ped_mood_rec = z(b, p, device=d)      # accumulated recovery time
        self.ob_pos = z(b, o, 2, device=d)
        self.reached = z(b, n, dtype=torch.bool, device=d)
        self.crashed = z(b, n, dtype=torch.bool, device=d)
        # failure-augmented cost bookkeeping (plan_emo.md 模块2)
        self._cost_terminal = z(b, n, device=d)
        self._fail_counted = z(b, n, dtype=torch.bool, device=d)
        self.step_i = z(b, dtype=torch.long, device=d)
        self._clear_metrics()

    def _clear_metrics(self):
        d = self.device
        inf = float("inf")
        self.rr_coll = torch.zeros(self.B, device=d)
        self.rp_coll = torch.zeros(self.B, device=d)
        self.ob_coll = torch.zeros(self.B, device=d)
        self.rr_min = torch.full((self.B,), inf, device=d)
        self.rp_min = torch.full((self.B,), inf, device=d)
        self.ep_rew = torch.zeros(self.B, device=d)
        # week-3 social book-keeping (per parallel env for the running episode)
        self.ep_discom_sum = torch.zeros(self.B, device=d)
        self.ep_discom_steps = torch.zeros(self.B, device=d)
        self.ep_near = torch.zeros(self.B, device=d)     # close encounters steps
        self.ep_rp_under = torch.zeros(self.B, device=d)  # outside comfort count

    # ------------------------------------------------------------------ spawn
    def _dist_goal(self):
        return (self.robo_pos - self.robo_goal).norm(dim=-1)

    def _sample_start_goal(self, e, used_list, boundary_clear=0.35):
        s = self.size
        dev = self.device
        if self.scene == "cross" and self.N >= 2:
            # deterministic single-file crossing (robot0 left->right,
            # robot1 bottom->top) with modest jitter-free placement.
            g = 0.8 * s
            lay = np.array([[-s*0.7, 0.0], [0.0, -s*0.7]])
            gl = np.array([[s*0.7, 0.0], [0.0, s*0.7]])
            self.robo_pos[e].copy_(torch.tensor(lay[:self.N],
                                                device=dev).float())
            self.robo_goal[e].copy_(torch.tensor(gl[:self.N],
                                                 device=dev).float())
            self.robo_vel[e].zero_()
            if self.N > 2:
                for a in range(2, self.N):
                    self.robo_pos[e, a].copy_(torch.tensor(
                        [0.0, 0.0], device=dev).float())
                    self.robo_goal[e, a].copy_(torch.tensor(
                        [0.0, 0.0], device=dev).float())
            return

        # "lanes"/"quadrants" scene: each robot starts in its own corner quadrant
        # and aims to the far opposite corner, so head-on cross-lock is minimal
        # (still a shared multi-robot world + crowd). Uses ± half-side targets.
        if self.scene == "lanes" and self.N >= 2:
            half = 0.7 * s
            for a in range(self.N):
                # a==0 from negative-x area -> +x; a==1 from -y -> +y
                if a == 0:
                    s0 = [-half, self._rng.uniform(-0.35, 0.35)*s]
                    t0 = [half, self._rng.uniform(-0.35, 0.35)*s]
                elif a == 1:
                    s0 = [self._rng.uniform(-0.35, 0.35)*s, -half]
                    t0 = [self._rng.uniform(-0.35, 0.35)*s, half]
                else:   # extras use mixed opposite corners
                    ang = 2*np.pi*a/self.N
                    s0 = [0.6*s*np.cos(ang + 0.2), 0.6*s*np.sin(ang + 0.2)]
                    t0 = [-s0[0], -s0[1]]
                self.robo_pos[e, a].copy_(torch.tensor(
                    np.clip(s0, -s+0.4, s-0.4), device=dev).float())
                self.robo_goal[e, a].copy_(torch.tensor(
                    np.clip(t0, -s+0.4, s-0.4), device=dev).float())
            self.robo_vel[e].zero_()
            return

        for a in range(self.N):
            cand = None
            for _tr in range(260):
                pnt = self._rng.uniform(-s+boundary_clear, s-boundary_clear, 2)
                if (not any(np.linalg.norm(np.asarray(u)-pnt) < self.r*2.6
                            for u in used_list)
                        and self._clear_ob(e, pnt, self.r+self.ob_r+0.15)):
                    cand = pnt
                    break
            if cand is None:
                cand = np.array(self._rng.uniform(-0.6, 0.6, 2))
            self.robo_pos[e, a].copy_(torch.tensor(cand, device=dev).float())
            used_list.append(list(map(float, cand)))
        # ---- scene variants for coordination / generalisation tests -------
        # "shared_goal": every robot is assigned the SAME target.  The default
        #   sampler gives each robot its own random goal, so the robots travel
        #   in separate directions and never have to negotiate -- there is
        #   nothing for the centralised critic to coordinate.  A shared goal
        #   forces them into one narrow region and makes yielding/ordering
        #   behaviour necessary.
        # "goal_min_dist": lower bound on the start->goal distance.  The default
        #   sampler only requires > 0.7 m, which under a tight time budget
        #   produces episodes that are IMPOSSIBLE from the outset (measured:
        #   25.8% of environments had a goal farther than T*v_max).
        # "goal_max_dist": upper bound on the start->goal distance.  The default
        #   sampler draws the goal uniformly in the arena, so the distance is
        #   whatever the geometry gives (measured on size=2.5: 0.70-5.26 m).
        #   Under a tight horizon that mixes two different experiments into one:
        #   episodes that are impossible (d > T*v_max) and episodes that are
        #   trivial (d ~ 0.7 m).  The measured consequence at T=2.5 s was that
        #   34.2% of episodes could not be completed by ANY policy, which made
        #   the reported success rate meaningless (see results/bench/
        #   goal_impact.log).  Capping the distance keeps every episode inside
        #   the feasible band, so success measures the POLICY rather than the
        #   sampler.
        # NOTE: the min/max distance band applies to BOTH regimes, but it does
        # NOT by itself select the shared-target regime.  An earlier version had
        # `min_d > 0` in this condition, which silently turned any distance-banded
        # scenario into a shared-goal one; the planned main experiment needs
        # INDEPENDENT per-robot goals with a bounded distance, so the condition is
        # now `shared` alone.  make_figures.py passes shared_goal=True explicitly,
        # so its long-distance condition is unaffected.
        shared = bool(getattr(self.cfg.env, "shared_goal", False))
        min_d = float(getattr(self.cfg.env, "goal_min_dist", 0.0))
        max_d = float(getattr(self.cfg.env, "goal_max_dist", 0.0))
        lo_d = max(0.7, min_d)                       # 0.7 = original floor
        hi_d = max_d if max_d > 0.0 else float("inf")
        if hi_d < lo_d:
            raise ValueError(
                f"empty goal-distance band: goal_min_dist={min_d} (effective "
                f"floor {lo_d}) > goal_max_dist={max_d}")
        if shared:
            # ORDER MATTERS: sample the GOAL first, then place the STARTS, so
            # that a start-distance floor can be satisfied by BOTH robots.
            #
            # The earlier implementation sampled starts first and then looked
            # for a goal, which cannot work in shared-goal mode: once the goal
            # is fixed by robot 0, robot 1's start may simply be too close to
            # it, and no amount of searching fixes that.  Measured
            # consequence: robot 0 started 4.13 m away while robot 1 started
            # only 0.08-1.77 m away, i.e. the "long distance" condition was not
            # actually met for the second robot.
            #
            # With the goal chosen first, the feasible starts are exactly the
            # points at distance >= d from it, which we sample directly.
            lo, hi = -s + 0.4, s - 0.4
            c0 = None
            for _tr in range(200):
                g = self._rng.uniform(lo, hi, 2)
                if self._clear_ob(e, g, self.ob_r + 0.3):
                    c0 = g
                    break
            if c0 is None:
                c0 = np.zeros(2)

            starts = []
            for a in range(self.N):
                p = None
                for _tr in range(200):
                    cand = self._rng.uniform(lo, hi, 2)
                    d0 = float(np.linalg.norm(cand - c0))
                    if d0 < lo_d or d0 > hi_d:
                        continue
                    if any(np.linalg.norm(cand - q) < self.r * 2.2
                           for q in starts):
                        continue
                    if not self._clear_ob(e, cand, self.r + self.ob_r + 0.15):
                        continue
                    p = cand
                    break
                if p is None:
                    # Deterministic fallback: step away from the goal along the
                    # direction that maximises the distance still available
                    # inside the arena.
                    corners = np.array([[lo, lo], [lo, hi], [hi, lo], [hi, hi]])
                    p = corners[int(np.argmax(
                        [np.linalg.norm(c - c0) for c in corners]))]
                starts.append(p)
                self.robo_pos[e, a].copy_(
                    torch.tensor(p, device=dev).float())
                self.robo_goal[e, a].copy_(
                    torch.tensor(c0, device=dev).float())
            self.robo_vel[e].zero_()
            return

        # Default regime: each robot gets its OWN goal, drawn inside the arena
        # subject to the requested start->goal distance band [lo_d, hi_d].
        for a in range(self.N):
            p = self.robo_pos[e, a].cpu().numpy()
            placed = False
            for _tr in range(240):
                c = self._rng.uniform(-s+0.4, s-0.4, 2)
                d = float(np.linalg.norm(c-p))
                if lo_d < d <= hi_d and self._clear_ob(e, c, self.ob_r+0.3):
                    self.robo_goal[e, a].copy_(
                        torch.tensor(c, device=dev).float())
                    placed = True
                    break
            if not placed:
                # Deterministic ring fallback: place the goal at the middle of
                # the requested band, in a random direction, clipped inside the
                # arena.  Only reached when the band is geometrically hard to
                # satisfy (e.g. an inner-city-wide floor in a small arena).
                r_mid = lo_d + 0.5 if hi_d == float("inf") else 0.5*(lo_d+hi_d)
                ang = self._rng.uniform(0.0, 2.0*np.pi)
                q = np.clip(p + r_mid*np.array([np.cos(ang), np.sin(ang)]),
                            -s+0.4, s-0.4)
                self.robo_goal[e, a].copy_(
                    torch.tensor(q, device=dev).float())
        self.robo_vel[e].zero_()

    def _sample_obstacles(self):
        if not self.O:
            return
        s = self.size
        for _tr in range(600):
            pts = self._rng.uniform(-s+0.8, s-0.8, (self.O, 2))
            if self._farther_obs(pts):
                bb = np.broadcast_to(pts, (self.B, self.O, 2)).copy()
                self.ob_pos.copy_(torch.tensor(bb, device=self.device).float())
                return
        bb = np.zeros((self.B, self.O, 2))
        self.ob_pos.copy_(torch.tensor(bb, device=self.device).float())

    def _farther_obs(self, pts):
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if np.linalg.norm(pts[i]-pts[j]) < 1.6:
                    return False
        return True

    def _clear_ob(self, e, pnt, thresh):
        if self.O == 0:
            return True
        ob = self.ob_pos[e].cpu().numpy()
        if ob.shape[0] == 0:
            return True
        return bool((np.linalg.norm(ob-np.asarray(pnt, float), axis=1) >
                     thresh).all())

    def _sample_ped_traits(self, e):
        """Per-episode, per-pedestrian realism traits for environment ``e``.

        Speed:  N(ped_speed_mean, ped_speed_std) clipped to
                [ped_speed_min, ped_speed_max].  With std=0 (the default) every
                pedestrian gets exactly ``pedestrian_max_speed`` -- the legacy
                constant crowd.
        Yield:  sampled from ``ped_yield_levels`` / ``ped_yield_probs`` so the
                crowd mixes oblivious, partially- and fully-yielding people.
                Without levels every pedestrian uses ``ped_avoid_robot``.
        Worker: ``ped_worker_frac`` of pedestrians stay at their workstation
                for the whole episode instead of walking to a random goal.
        """
        if not self.P:
            return
        P = self.P
        if self.ped_sp_std > 0.0:
            sp = self._rng.normal(self.ped_sp_mean, self.ped_sp_std, P)
        else:
            sp = np.full(P, self.ped_sp_mean)
        sp = np.clip(sp, self.ped_sp_lo, self.ped_sp_hi)
        self.ped_sp_vec[e].copy_(torch.tensor(sp, device=self.device).float())

        if self.ped_yield_levels:
            lv = np.asarray(self.ped_yield_levels, float)
            pr = np.asarray(self.ped_yield_probs or np.ones(len(lv)), float)
            pr = pr / pr.sum()
            yv = self._rng.choice(lv, size=P, p=pr)
        else:
            yv = np.full(P, float(self.ped_avoid_robot))
        self.ped_yield_vec[e].copy_(torch.tensor(yv, device=self.device).float())

        if self.ped_worker_frac > 0.0:
            wk = self._rng.random(P) < self.ped_worker_frac
            # NOTE: sampled ONLY when workers exist.  Drawing it unconditionally
            # consumed an extra value from the episode RNG stream and thereby
            # changed every later sample -- a legacy-recipe regression test
            # caught it (wid moved 0.3106 -> 0.2848 with all new knobs off).
            self.ped_worker_phase[e].copy_(torch.tensor(
                self._rng.uniform(0.0, 2*np.pi, P), device=self.device).float())
        else:
            wk = np.zeros(P, bool)
        self.ped_is_worker[e].copy_(torch.tensor(wk, device=self.device))

    def _spawn_peds(self, e):
        s = self.size
        dev = self.device
        used = [self.robo_pos[e, a].cpu().numpy().tolist()
                for a in range(self.N)]
        if self.ped_clear_goals:
            # FOUND BY LOOKING AT TRAJECTORIES (scripts/viz_ws.py), not by
            # reading metrics: reset() samples the robots' goals BEFORE spawning
            # pedestrians, and spawning only kept clear of robot *positions*.
            # A pedestrian -- or an anchored worker, which cannot walk away --
            # could therefore spawn essentially ON a robot's destination.
            # Measured: min pedestrian-to-goal distance 0.016 m; on the old
            # dense scene 100% of environments had a pedestrian within 1 m of a
            # goal and 86% within 0.5 m.  The robot then had to invade someone's
            # personal space simply to finish its task, and that pedestrian
            # stayed pinned at mood -1.0 for the rest of the episode -- an
            # emotional cost the policy could not avoid, which inflates every
            # social metric (and confounds the historical results).
            used += [self.robo_goal[e, a].cpu().numpy().tolist()
                     for a in range(self.N)]
        for i in range(self.P):
            cand = None
            for _tr in range(180):
                c = self._rng.uniform(-s+0.35, s-0.35, 2)
                if (all(np.linalg.norm(np.asarray(u)-c) >
                        self.r+self.ped_r+0.35 for u in used)
                        and self._clear_ob(e, c, self.ob_r+self.ped_r+0.15)):
                    cand = c
                    break
            if cand is None:
                cand = self._rng.uniform(-s*0.4, s*0.4, 2)
            self.ped_pos[e, i].copy_(torch.tensor(cand, device=dev).float())
            used.append([float(cand[0]), float(cand[1])])
            ang = self._rng.uniform(0, 2*math.pi)
            rr = float(np.clip(self._rng.uniform(0.2, 0.9)*s, 0.3, s*0.8))
            gg = np.clip(cand+np.array([math.cos(ang)*rr, math.sin(ang)*rr]),
                         -s+0.3, s-0.3)
            if self.ped_worker_frac > 0.0 and bool(self.ped_is_worker[e, i]):
                # stationary worker: stays at its workstation (goal == start),
                # so it still occupies space, yields, and accumulates mood --
                # it just does not commute.  Matches the target application
                # (people working in the same space as the robots).
                gg = np.array(cand, dtype=float)
                self.ped_home[e, i].copy_(
                    torch.tensor(gg, device=dev).float())
            self.ped_goal[e, i].copy_(torch.tensor(gg, device=dev).float())
        self.ped_vel[e].zero_()
        gdir = self.ped_goal[e] - self.ped_pos[e]
        self.ped_heading[e] = torch.atan2(gdir[..., 1], gdir[..., 0])

    def reset(self, seed=None):
        if seed is not None:
            self.base_seed = int(seed)
        self._rng = np.random.default_rng(
            int(self.base_seed) + 7_777_733 * (self._n_resets % 4096))
        self._n_resets += 1
        self._sample_obstacles()
        for e in range(self.B):
            self._sample_start_goal(e, [], 0.35)
            if self.P:
                self._sample_ped_traits(e)
                self._spawn_peds(e)
        if self.P:
            self.ped_vel.zero_()
            # initial heading aimed at first goal (before any motion)
            gdir = self.ped_goal - self.ped_pos
            self.ped_heading = torch.atan2(gdir[..., 1], gdir[..., 0])
        self.reached = torch.zeros_like(self.reached)
        self.crashed = torch.zeros_like(self.crashed)
        self._cost_terminal.zero_()
        self._fail_counted.zero_()
        self.step_i = torch.zeros(self.B, dtype=torch.long,
                                  device=self.device)
        if self.robot_emo["enabled"]:
            self.robo_mood.zero_()
            self.robo_stress.zero_()
            self._rm_prev_rr = None
            self._rm_prev_rp = None
        self._clear_metrics()
        return self.get_obs(), self.get_state()

    def _replay_env(self, e):
        self._sample_start_goal(int(e), [], 0.35)
        if self.P:
            self._sample_ped_traits(int(e))
            self._spawn_peds(int(e))
        self.robo_vel[int(e)].zero_()
        self.reached[int(e)] = False
        self.crashed[int(e)] = False
        self._cost_terminal[int(e)] = 0.0
        self._fail_counted[int(e)] = False
        self.step_i[int(e)] = 0
        self.rr_coll[int(e)] = 0.
        self.rp_coll[int(e)] = 0.
        self.ob_coll[int(e)] = 0.
        self.rr_min[int(e)] = float("inf")
        self.rp_min[int(e)] = float("inf")
        if self.robot_emo["enabled"]:
            self.robo_mood[int(e)].zero_()
            self.robo_stress[int(e)].zero_()
            # positions were just re-sampled; drop the distance caches so the
            # jump is not misread as "everyone suddenly yielded to me"
            self._rm_prev_rr = None
            self._rm_prev_rp = None
        if self.P:
            self.ped_mood[int(e)] = 0.0
            self.ped_mood_sum[int(e)] = 0.0
            self.ped_mood_n[int(e)] = 0.0
            self.ped_mood_min[int(e)] = 0.0
            self.ped_mood_rec[int(e)] = 0.0
            self.ped_mood_dose[int(e)] = 0.0
            self.ped_dose_impact[int(e)] = 0.0
            self.ped_dose_exposure[int(e)] = 0.0
            self.ped_q[int(e)] = 0.0
            self.ped_mood_min[int(e)] = 0.0
        self.ep_rew[int(e)] = 0.
        self.ep_discom_sum[int(e)] = 0.
        self.ep_discom_steps[int(e)] = 0.
        self.ep_near[int(e)] = 0.
        self.ep_rp_under[int(e)] = 0.

    # ------------------------------------------------------------ pedestrians
    # ------------------------------------------------------------ emotion
    @torch.no_grad()
    def _update_mood(self):
        """Evolve per-pedestrian mood from the social intrusion robots cause.

        dmood/dt = -alpha*mood                       (decay back to neutral)
                   - beta * intrusion                (anisotropic invasion)
                   + delta * courtesy                (robot yields)

        `intrusion` reuses the SAME anisotropic social radius d_soc(phi) that
        RA-CMAPPO's risk field uses, so the anisotropy finally has a variable it
        visibly drives (the pedestrian's emotional state) instead of only a soft
        cost term that full training averages away.
        """
        if self.P == 0 or not self.N:
            return
        cfg = self.emo
        rel = self.robo_pos.unsqueeze(2) - self.ped_pos.unsqueeze(1)   # B,N,P,2
        dist = rel.norm(dim=-1).clamp_min(1e-4)                        # B,N,P
        # bearing of the robot as seen from the pedestrian's heading
        bear = torch.atan2(rel[..., 1], rel[..., 0]) - self.ped_heading.unsqueeze(1)
        bear = torch.atan2(torch.sin(bear), torch.cos(bear))
        from .risk_field import anisotropic_social_radius
        d_soc = anisotropic_social_radius(
            bear, d_front=cfg["d_front"], d_side=cfg["d_side"],
            d_back=cfg["d_back"])                                      # B,N,P
        intrusion = torch.clamp(d_soc - dist, 0.0, None).amax(dim=1)   # B,P
        # Courtesy is credited only when the robot is INSIDE the pedestrian's
        # social radius AND actively opening the gap: that is genuine yielding.
        # (A blanket "moving away" term gave credit to robots that simply
        # finished and left, which made fast/aggressive policies look polite.)
        relv = self.robo_vel.unsqueeze(2) - self.ped_vel.unsqueeze(1)
        closing = -(rel * relv).sum(-1) / dist                         # >0 approaching
        inside = (dist < d_soc)
        yielding = torch.clamp(-closing, 0.0, None) * inside.float()
        courtesy = yielding.amax(dim=1)                                # B,P

        dmood = (-cfg["alpha"] * self.ped_mood
                 - cfg["beta"] * intrusion
                 + cfg["delta"] * courtesy)
        self.ped_mood = torch.clamp(
            self.ped_mood + dmood * self.action_dt, -1.0, 1.0)
        self.ped_mood_sum += self.ped_mood
        self.ped_mood_n += 1.0
        self.ped_mood_min = torch.minimum(self.ped_mood_min, self.ped_mood)
        # cumulative deficit integral (P5)
        self.ped_mood_dose += torch.clamp(-self.ped_mood, 0.0, 1.0) * self.action_dt

        # ---- three-layer emotion decomposition ---------------------------
        # MOTIVATION: `ped_mood` is ITSELF an integral of intrusion (its time
        # constant is 1/alpha = 16.7 s, i.e. 2.1x the 8 s episode, so it barely
        # decays within an episode).  Integrating `max(0, -mood)` again is
        # therefore a DOUBLE integral of intrusion, which grows like t^2.
        # Measured: with a slow policy the double integral (1.50) exceeded the
        # fast policy's (0.62), while the SINGLE integral of instantaneous
        # intrusion ranked correctly (fast 0.53 > slow 0.41).  The double
        # integral was why "slower" looked more harmful, which in turn is why
        # the constraint was previously believed to reward rushing.
        #
        # So we now separate three quantities explicitly:
        #   exposure : D = sum I dt          (spatial exposure)
        #   state    : D = sum relu(-m) dt   (negative-affect duration; = AID)
        #   impact   : D = sum q dt          (composite instantaneous harm)
        # `impact` is the one used as the constraint cost.
        #
        # q_t combines four factors so that "fast lateral cut-in" is not
        # mistaken for "harmless because it was brief":
        #   wI * intrusion   anisotropic spatial invasion
        #   wV * closing     approach speed (only while approaching)
        #   wM * relu(-mood) current negative affect
        # (a predictive term wP is added by the env when anticipation is on)
        if self.emo.get("impact_on", True):
            # `intrusion` is already reduced over robots (B,P) via amax, so the
            # closing term must be too, otherwise the shapes disagree.
            #
            # CRITICAL: the closing term must be GATED by being inside the
            # social radius.  Measured without the gate, `closing` averaged
            # 0.596 while `intrusion` averaged only 0.049 -- a 12x imbalance
            # that made q almost pure approach-speed, so a robot walking
            # toward a pedestrian from 3 m away scored as harmful.  Speed only
            # matters when the robot is actually within personal space.
            appr = torch.clamp(closing, 0.0, None)
            appr = (appr * (dist < d_soc).float()).amax(dim=1)          # B,P
            q = (cfg.get("w_I", 1.0) * intrusion
                 + cfg.get("w_V", 0.25) * appr
                 + cfg.get("w_M", 0.5) * torch.clamp(-self.ped_mood, 0.0, 1.0))
            self.ped_q = q                                             # B,P
            self.ped_dose_impact += q * self.action_dt
            self.ped_dose_exposure += intrusion * self.action_dt

        # ---- ARI: time a distressed pedestrian needs to come back up -----
        # For each pedestrian that has ever dropped below -0.5, accumulate the
        # steps spent recovering until mood is back above -0.2.
        was_distressed = self.ped_mood_min < -0.5
        recovering = was_distressed & (self.ped_mood < -0.2)
        self.ped_mood_rec = torch.where(recovering, self.ped_mood_rec + 1.0,
                                        self.ped_mood_rec)

        # ---- FEEDBACK LOOP (design decision 2) ---------------------------
        # A distressed pedestrian actively gives way: it slows down and its
        # effective goal is pushed away from the nearest robot.  Mood therefore
        # changes the crowd's dynamics, coupling robot behaviour back into the
        # difficulty of the task.
        if cfg.get("feedback", True):
            rel = self.robo_pos.unsqueeze(2) - self.ped_pos.unsqueeze(1)
            dist = rel.norm(dim=-1).clamp_min(1e-4)
            nearest = dist.amin(dim=1)                        # (B,P)
            u = rel / dist.unsqueeze(-1)
            w = torch.softmax(-dist, dim=1)                   # weight by proximity
            away = -(u * w.unsqueeze(-1)).sum(dim=1)          # (B,P,2) away-vector
            stress = torch.clamp(-self.ped_mood, 0.0, 1.0)    # 0 calm .. 1 upset
            self.ped_goal = self.ped_goal + (away * stress.unsqueeze(-1)
                                             * cfg["avoid_gain"]
                                             * self.action_dt)
            self.ped_goal = self.ped_goal.clamp(-self.size + 0.1,
                                                self.size - 0.1)

    def _advance_peds(self):
        d = self.device
        ped, goal = self.ped_pos, self.ped_goal
        # --- circulating walkers: pick a NEW destination on arrival --------
        # Legacy behaviour was "walk to one goal, then stand there forever",
        # which turns a moving crowd into static obstacles after ~5 s and makes
        # long episodes unrealistic.  With ped_circulate the walkers keep
        # commuting (workers stay put either way).
        if self.ped_circulate and self.P:
            dg0 = (goal - ped).norm(dim=-1)
            arr = (dg0 < 0.35) & (~self.ped_is_worker)
            if bool(arr.any()):
                lim = float(self.size) - 0.3
                # REPRODUCIBILITY FIX (2026-09-21): this used the GLOBAL torch
                # RNG, which PyTorch seeds non-deterministically at process
                # start, so every process drew different re-goals.  Consequences
                # measured: (a) two processes evaluating the SAME scripted
                # controller on the same env seed gave different success rates
                # (0.805 vs 0.859); (b) two training runs of the same arm+seed
                # diverged (0.646 vs 0.470 at iter 250); (c) the paper scene was
                # not bit-reproducible, which is why its regression hash moved.
                # Only recipes with ped_circulate=True hit this line, which is
                # why the legacy (amc_mid) guard stayed stable all along.
                # The env's own seeded generator keeps the draw deterministic.
                newg = torch.as_tensor(
                    self._rng.uniform(-lim, lim, size=(self.B, self.P, 2)),
                    device=d, dtype=torch.float32)
                self.ped_goal = torch.where(arr.unsqueeze(-1), newg,
                                            self.ped_goal)
                goal = self.ped_goal
        # Stationary workers are ANCHORED to their workstation: without this
        # they slowly drift (they get pushed by the pedestrian-pedestrian
        # repulsion, and the mood-driven "back off" goal shift moves their
        # target away from a robot).  Measured drift without the anchor was
        # 2.4 m over 10 s -- i.e. they were not stationary at all.
        if self.ped_worker_frac > 0.0 and self.P:
            if self.ped_worker_activity > 0.0:
                # Local work activity: a slow, small-amplitude motion around the
                # workstation.  Measured before this change: 27% of pedestrians
                # moved <0.25 m in a whole episode and were below 0.05 m/s 48%
                # of the time -- frozen pedestrians carry no social information
                # and make the scene look broken.
                tt = (self.step_i.float()*self.action_dt).unsqueeze(-1)   # (B,1)
                ang = (2*np.pi*tt/self.ped_worker_period
                       + self.ped_worker_phase)                           # (B,P)
                g = self.ped_home + self.ped_worker_activity*torch.stack(
                    [torch.cos(ang), torch.sin(ang)], dim=-1)
            else:
                g = self.ped_home
            goal = torch.where(self.ped_is_worker.unsqueeze(-1), g, goal)
            self.ped_goal = goal
        dvec = goal - ped
        dg = dvec.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        unit = dvec / dg
        arrival = (dg.squeeze(-1)/0.7).clamp(0, 1)
        # --- P3a: mood modulates walking speed -----------------------------
        # Previously ped_sp was a constant, so affect had no effect on how fast
        # anyone actually moved.  A distressed pedestrian now slows down.
        # ped_sp_vec carries the PER-PEDESTRIAN free speed (plan_realism §2.1):
        # a heterogeneous stream, not one scalar for the whole crowd.
        if self.emo["enabled"] and self.P:
            stress = torch.clamp(-self.ped_mood, 0.0, 1.0)
            sp_eff = (self.ped_sp_vec * (1.0 - self.mood_slow * stress)
                      ).unsqueeze(-1)                     # (B,P,1)
        else:
            sp_eff = self.ped_sp_vec.unsqueeze(-1)
        vel = unit * sp_eff * arrival.unsqueeze(-1)
        steer = torch.zeros_like(vel)      # avoidance contribution (see below)
        if self.N:
            dp = self.robo_pos.unsqueeze(2) - ped.unsqueeze(1)
            dd = dp.norm(dim=-1)
            safe = self.r + self.ped_r + 0.35
            if (dd < safe).any():
                u = dp / (dd + 1e-6).unsqueeze(-1)
                w = (1.15 - dd/safe).clamp(0, 1).unsqueeze(-1)
                # robot-avoidance strength.  At 2.2 this force is 4.4x the
                # pedestrians' own walking speed, so pedestrians dodge robots
                # almost regardless of what the robot does -- which erases the
                # effect of robot behaviour on pedestrian affect.  Exposed as a
                # knob to test the "pedestrians do not yield" regime.
                # --- P3c: distressed pedestrians dodge harder --------------
                # (contribution is (B,P), so keep `force` as (B,P,1) for the
                #  broadcast against the (B,P,2) velocity)
                push = (u*w).sum(dim=1)                       # (B,P,2)
                if self.emo["enabled"] and self.P:
                    stress_b = torch.clamp(-self.ped_mood, 0.0, 1.0)   # (B,P)
                    force = (self.ped_yield_vec
                             * (1.0 + self.mood_dodge * stress_b)).unsqueeze(-1)
                else:
                    force = self.ped_yield_vec.unsqueeze(-1)
                steer = steer + push * force
        if self.P > 1:
            dp = ped.unsqueeze(2) - ped.unsqueeze(1)
            dd = dp.norm(dim=-1)
            eye = torch.eye(self.P, device=d, dtype=torch.bool).view(
                1, self.P, self.P)
            dd = dd.masked_fill(eye, 1e6)
            safe2 = 2*self.ped_r + 0.2
            if (dd < safe2).any():
                u = dp / (dd+1e-6).unsqueeze(-1)
                w = (1.05 - dd/safe2).clamp(0, 1).unsqueeze(-1)
                steer = steer + (u*w).sum(dim=1)*1.7
        if self.ped_steer_mode == "blend":
            # Bounded direction blending: the avoidance contribution changes
            # WHERE the pedestrian walks, never how fast.  The additive legacy
            # form let neighbours cancel the desired direction, which produced
            # 49% heading flips and 49% speed reversals per step.
            des = unit*sp_eff + steer          # sp_eff is already (B,P,1)
            vel = (des/(des.norm(dim=-1, keepdim=True) + 1e-6)
                   * sp_eff * arrival.unsqueeze(-1))
        else:
            vel = vel + steer
        # cap by each pedestrian's OWN free speed (heterogeneous when sampled)
        sp_cap = self.ped_sp_vec.unsqueeze(-1)
        vel = torch.max(torch.min(vel, sp_cap), -sp_cap)
        # Stationary workers cannot be pushed off their workstation.  The
        # pedestrian-pedestrian repulsion is a steering term whose magnitude is
        # NOT bounded by walking speed, so a worker standing in a crowd was
        # shoved outward without limit (measured drift 2.4 m over 10 s, i.e. the
        # "stationary" workers were not stationary).  People at a workstation
        # brace: allow a small nudge, then remove the outward radial component.
        if self.ped_worker_frac > 0.0 and self.P:
            off = ped - self.ped_home
            dist = off.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            radial = off / dist
            vr = (vel * radial).sum(dim=-1, keepdim=True)
            bound = self.ped_worker_radius
            if self.ped_worker_activity > 0.0:
                bound = max(bound, self.ped_worker_activity + 0.15)
            stuck = (dist > bound) & (vr > 0)
            vel = vel - torch.where(stuck, vr * radial,
                                    torch.zeros_like(vr))
        self.ped_vel = vel
        # heading from velocity when moving; retain previous when near-stationary
        sp = vel.norm(dim=-1)
        moving = sp > 0.05
        hdg = torch.atan2(vel[..., 1], vel[..., 0])
        self.ped_heading = torch.where(moving, hdg, self.ped_heading)

        # --- P3b: distressed pedestrians turn to FACE the nearest robot -----
        # This is the mechanism that couples affect back into the RISK FIELD:
        # turning toward a robot increases the bearing term, which selects the
        # LARGER frontal radius d_front (1.0) instead of d_back (0.4), raising
        # the robot's computed intrusion/cost.  Affect therefore changes the
        # optimisation signal itself, not just the recorded metric.
        if self.emo["enabled"] and self.P and self.N and self.mood_turn > 0:
            relh = self.robo_pos.unsqueeze(2) - self.ped_pos.unsqueeze(1)  # B,N,P,2
            disth = relh.norm(dim=-1).clamp_min(1e-4)
            near_idx = disth.argmin(dim=1)                                # B,P
            bi = torch.arange(self.B, device=d).unsqueeze(1).expand(-1, self.P)
            rel_near = relh[bi, near_idx, torch.arange(self.P, device=d)]  # B,P,2
            face = torch.atan2(rel_near[..., 1], rel_near[..., 0])
            stress_t = torch.clamp(-self.ped_mood, 0.0, 1.0)               # B,P
            blend = (self.mood_turn * stress_t * self.action_dt).clamp(0, 1)
            diff = torch.atan2(torch.sin(face - self.ped_heading),
                               torch.cos(face - self.ped_heading))
            self.ped_heading = self.ped_heading + blend * diff
        self.ped_pos = (vel*self.action_dt + ped).clamp(
            -self.size+0.03, self.size-0.03)

    # ------------------------------------------------- physical plausibility
    def goal_contention_features(self, k=16):
        """(B, N, 2): "is somebody else about to take my destination?"

        Returns [min_p ||predicted ped position - my goal|| / size,
                 whether that pedestrian is moving toward my goal].

        This is the feature that makes yielding at a shared/nearby goal
        learnable: the plain observation cannot distinguish "a pedestrian
        walking past" from "a pedestrian about to arrive at the point I am
        driving to".  Both entries come from positions and velocities only --
        a linear prediction of the kind a real tracker provides -- so no oracle
        intent is leaked to the policy.
        """
        if self.P == 0 or not self.N or k <= 0:
            return torch.zeros(self.B, self.N, 2, device=self.device)
        pp = self.ped_pos + self.ped_vel * (k * self.action_dt)       # (B,P,2)
        rel = self.robo_goal.unsqueeze(2) - pp.unsqueeze(1)           # (B,N,P,2)
        d = rel.norm(dim=-1)                                          # (B,N,P)
        dmin, idx = d.min(dim=-1)                                     # (B,N)
        toward = ((self.robo_goal.unsqueeze(2) - pp.unsqueeze(1))
                  * self.ped_vel.unsqueeze(1)).sum(-1) > 0.0          # (B,N,P)
        closing = toward.gather(2, idx.unsqueeze(-1)).squeeze(-1).float()
        return torch.stack([dmin/self.size, closing], dim=-1)

    def _cancel_inward_velocity(self, rob_mob, ped_mob, tol=0.02):
        """Velocity-level non-penetration at existing contacts.

        Removing only the penetration (position projection) leaves the agent
        driving INTO whatever it touched; measured on the old policy, 68.5% of
        the steps that were in contact with an obstacle still had a positive
        inward velocity -- i.e. it ground along the obstacle instead of
        stopping or routing around it.  A rigid contact removes that inward
        component, which also gives the policy a much cleaner "I am blocked"
        signal.  Only movable agents are affected (weighted by their mobility),
        so obstacles and braced workers are unaffected.
        """
        def cancel(vel, pos_a, pos_b, thr, mob):
            rel = pos_b.unsqueeze(1) - pos_a.unsqueeze(2)             # (B,Ka,Kb,2)
            d = rel.norm(dim=-1)
            touch = d < (thr + tol)
            if not bool(touch.any()):
                return vel
            n = rel / d.clamp_min(1e-6).unsqueeze(-1)
            vn = (vel.unsqueeze(2) * n).sum(-1)                       # (B,Ka,Kb)
            vn = torch.where(touch, vn.clamp_min(0.0), torch.zeros_like(vn))
            w = mob.unsqueeze(2).clamp(0.0, 1.0)
            return vel - ((w*vn).unsqueeze(-1) * n).sum(2)

        if self.N and self.O:
            self.robo_vel = cancel(self.robo_vel, self.robo_pos, self.ob_pos,
                                   self.r + self.ob_r, rob_mob)
        if self.N and self.P:
            self.robo_vel = cancel(self.robo_vel, self.robo_pos, self.ped_pos,
                                   self.r + self.ped_r, rob_mob)
            self.ped_vel = cancel(self.ped_vel, self.ped_pos, self.robo_pos,
                                  self.r + self.ped_r, ped_mob)
        if self.N > 1:
            self.robo_vel = cancel(self.robo_vel, self.robo_pos, self.robo_pos,
                                   2*self.r, rob_mob)
        if self.P > 1:
            self.ped_vel = cancel(self.ped_vel, self.ped_pos, self.ped_pos,
                                  2*self.ped_r, ped_mob)
        if self.P and self.O:
            self.ped_vel = cancel(self.ped_vel, self.ped_pos, self.ob_pos,
                                  self.ped_r + self.ob_r, ped_mob)

    def _resolve_overlaps(self, iters=3):
        """Geometric non-overlap projection, run after the integrator.

        Motivation (found by inspecting trajectories, then measured by
        scripts/check_physics.py): the environment previously only *penalised*
        contacts, so agents could interpenetrate completely -- measured max
        penetration 0.549 m of a 0.55 m robot-pedestrian contact distance,
        0.582 m of 0.60 m for obstacles (paths crossed solid obstacles) and
        0.50 m (full overlap) between pedestrians.

        Mobility decides who absorbs the correction:
          * obstacles never move,
          * a pedestrian anchored at a workstation braces (it is only nudged
            inside ped_worker_radius by the velocity projection),
          * a robot that has already finished or crashed holds its position,
          * otherwise the two agents share the correction equally.
        """
        s = self.size - 0.03
        # Mobility WEIGHTS (not booleans).  Only obstacles are truly immovable.
        # A first version gave reached/crashed robots and anchored workers zero
        # mobility, and an immovable-vs-immovable pair then had no correction at
        # all -- measured residual penetration was still 0.53 m (robot on a
        # worker).  A parked robot can in fact yield, and a worker can be nudged,
        # so both get a small weight and every pair resolves.
        rob_mob = torch.where(
            self.reached | self.crashed,
            torch.full_like(self.robo_pos[..., 0], 0.25),
            torch.ones(self.B, self.N, device=self.device))
        ped_mob = (torch.where(
            self.ped_is_worker,
            torch.full_like(self.ped_mood, 0.25),
            torch.ones(self.B, self.P, device=self.device)) if self.P else None)

        def split(ma, mb):
            """normalised mobility shares; both-immovable pairs get (0, 0)."""
            tot = ma + mb
            live = (tot > 1e-6).float()
            return (ma/(tot + 1e-6))*live, (mb/(tot + 1e-6))*live

        def push(A, B_, thr, ma, mb, drop_self=False):
            """push A (B,Ka,2) and B_ (B,Kb,2) apart to `thr` separation."""
            rel = A.unsqueeze(2) - B_.unsqueeze(1)                     # (B,Ka,Kb,2)
            dist = rel.norm(dim=-1)                                    # (B,Ka,Kb)
            need = (thr - dist).clamp_min(0.0)
            if drop_self:
                k = min(A.shape[1], B_.shape[1])
                idx = torch.arange(k, device=rel.device)
                need[:, idx, idx] = 0.0
            if float(need.max()) < 1e-9:
                return A, B_
            u = rel / dist.clamp_min(1e-6).unsqueeze(-1)
            sa, sb = split(ma.unsqueeze(2), mb.unsqueeze(1))           # (B,Ka,Kb)
            ca = (u*((need*sa).unsqueeze(-1))).sum(2)
            cb = (-u*((need*sb).unsqueeze(-1))).sum(1)
            return A + ca, B_ + cb

        for _ in range(iters):
            if self.P:
                self.ped_pos, _ = push(self.ped_pos, self.ped_pos,
                                       2*self.ped_r, ped_mob, ped_mob,
                                       drop_self=True)
                if self.N:
                    self.robo_pos, self.ped_pos = push(
                        self.robo_pos, self.ped_pos, self.r + self.ped_r,
                        rob_mob, ped_mob)
                if self.O:
                    self.ped_pos, _ = push(
                        self.ped_pos, self.ob_pos, self.ped_r + self.ob_r,
                        ped_mob, torch.zeros(self.B, self.O, device=self.device))
            if self.N > 1:
                self.robo_pos, _ = push(self.robo_pos, self.robo_pos,
                                        2*self.r, rob_mob, rob_mob,
                                        drop_self=True)
            if self.N and self.O:
                # solid obstacles: the obstacle absorbs none of the correction
                self.robo_pos, _ = push(
                    self.robo_pos, self.ob_pos, self.r + self.ob_r,
                    rob_mob, torch.zeros(self.B, self.O, device=self.device))
            self.robo_pos = self.robo_pos.clamp(-s, s)
            if self.P:
                self.ped_pos = self.ped_pos.clamp(-s, s)
        if self.contact_stop:
            # a rigid contact also removes the velocity pushing into it
            self._cancel_inward_velocity(rob_mob, ped_mob)

    # ------------------------------------------------------------ collision
    def _overlaps_rr(self):
        if self.N < 2:
            return torch.zeros(self.B, device=self.device, dtype=torch.bool)
        dd = torch.cdist(self.robo_pos, self.robo_pos)
        eye = torch.eye(self.N, device=self.device).bool().unsqueeze(0)
        dd = dd.masked_fill(eye, 1e9)
        return dd.min(dim=-1).values.min(dim=-1).values < self.coll_rr

    def _overlaps_rp(self):
        """per-ENVIRONMENT flag: does ANY robot touch ANY pedestrian?

        BUG FIX (2026-09-21): this used ``.norm(-1)``, where -1 is interpreted
        as the *p-order* (not a dimension), so it collapsed the whole (B,N,P)
        distance tensor to a 0-dim scalar.  The comparison then produced a
        scalar True, i.e. **the collision flag was constantly 1.0 for every
        environment and every robot** (measured: flag 1.0 while only 6% of
        environments actually had a contact).  Consequences: the per-robot
        robot-pedestrian reward penalty of 1.2 was replaced by a global
        constant, so *nothing in the reward ever taught the robots to avoid
        people* -- which is why every learned policy ended up behaving like the
        oblivious walk-straight controller on the social axis.
        """
        if self.P == 0:
            return torch.zeros(self.B, device=self.device, dtype=torch.bool)
        dd = (self.robo_pos.unsqueeze(2)-self.ped_pos.unsqueeze(1)).norm(dim=-1)
        return dd.min(dim=-1).values.min(dim=-1).values < self.coll_rp

    def _overlaps_ob(self):
        """per-ENVIRONMENT flag: does ANY robot touch ANY obstacle? (same
        .norm(-1) bug as _overlaps_rp -- see the note there.)"""
        if self.O == 0:
            return torch.zeros(self.B, device=self.device, dtype=torch.bool)
        dd = (self.robo_pos.unsqueeze(2)-self.ob_pos.unsqueeze(1)).norm(dim=-1)
        return dd.min(dim=-1).values.min(dim=-1).values < self.r + self.ob_r

    # ------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, action):
        a = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if a.dim() == 2:
            a = a.unsqueeze(0)
        if a.shape[0] != self.B:
            a = a.expand(self.B, -1, -1).contiguous()
        # control delay: queue actions and apply an older one (0 = off)
        if self.control_delay and self.control_delay > 0:
            self._act_queue.append(a.clone())
            if len(self._act_queue) > self.control_delay:
                a = self._act_queue.pop(0)
            else:
                a = self._act_queue[0]
        prev_d = self._dist_goal()

        scl = torch.clamp(self.max_sp/(a.norm(dim=-1)+1e-6), None, 1.0)
        vel = a * scl.unsqueeze(-1)
        if self.detour_prior_gain > 0.0 and self.P:
            # analytic yielding prior: rotate the commanded velocity away from
            # pedestrians inside `detour_prior_radius`, exactly like the
            # scripted detour controller, but KEEP the policy's speed and let
            # the policy fight the prior.  Rotating (rather than adding and
            # re-normalising the magnitude) matters: adding a small repulsion
            # to a unit command barely changes its direction, whereas the
            # scripted controller normalises the SUM, so even a distant
            # pedestrian rotates the full-speed velocity.  Measured: the
            # additive version reached 0.766/28.6 against the scripted
            # controller's 0.797/24.9, i.e. it was a much weaker prior.
            rel = self.robo_pos.unsqueeze(2) - self.ped_pos.unsqueeze(1)
            dist = rel.norm(dim=-1).clamp_min(1e-6)
            u = rel / dist.unsqueeze(-1)
            wt = (self.detour_prior_radius - dist).clamp(0.0, 1.0).unsqueeze(-1)
            rep = self.detour_prior_gain * (u*wt).sum(dim=2)
            comb = vel + rep
            mag = comb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            vel = comb / mag * vel.norm(dim=-1, keepdim=True)
        if self.max_accel > 0.0:
            # physical rate limit: |v_t - v_{t-1}| <= a_max*dt  (plan_realism §2.2)
            # Without it the command is applied instantly (infinite acceleration)
            # and the sudden-stop / oscillation / jerk metrics are meaningless.
            # NOTE: the limit is on the VECTOR magnitude (a = dv/dt is a vector),
            # not per component -- a per-component clamp would allow sqrt(2)*a.
            lim = self.max_accel * self.action_dt
            dv = vel - self.robo_vel
            mag = dv.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            vel = self.robo_vel + dv * torch.clamp(lim / mag, None, 1.0)
        vel = vel.masked_fill(self.reached.unsqueeze(-1), 0.)
        vel = vel.masked_fill(self.crashed.unsqueeze(-1), 0.)
        self.robo_vel = vel
        self.robo_pos = (self.robo_pos + vel*self.action_dt).clamp(
            -self.size+0.03, self.size-0.03)
        self.step_i += 1
        if self.P:
            if self._ped_override is not None:
                # external planner drives the crowd (reciprocal protocols)
                self.ped_vel = self._ped_override.to(self.device).clone()
                self.ped_pos = (self.ped_vel*self.action_dt + self.ped_pos).clamp(
                    -self.size+0.03, self.size-0.03)
                self._ped_override = None
            else:
                self._advance_peds()
        if self.hard_separation:
            # resolve overlaps geometrically (see _resolve_overlaps).  Without
            # this the only response to a contact is a reward penalty, so a
            # policy could drive straight through people and obstacles --
            # measured max penetration 0.55 m (robot-pedestrian), 0.58 m
            # (robot-obstacle) and 0.50 m (pedestrian-pedestrian).
            self._resolve_overlaps()
        if self.emo["enabled"]:
            self._update_mood()
        self._refresh_metrics()

        new_d = self._dist_goal()
        if self.robot_emo["enabled"]:
            self._update_robot_mood(prev_d, new_d)
        rew = (prev_d-new_d)*self.w_pot
        newly = (new_d <= self.goal_tol) & (~self.reached) & (~self.crashed)
        rew = rew + newly.float()*self.rew_reach
        self.reached = self.reached | newly

        timeout = self.step_i >= self.horizon
        # ---- failure-augmented constrained cost (plan_emo.md 模块2) --------
        # plan_emo specified C(tau) = D(tau) + kappa_f*I[failure] + kappa_t*T_rem
        # precisely so that STANDING STILL is the most expensive option.  That
        # term was never implemented: the trainer accumulated only D, so the
        # cheapest way to satisfy the constraint was to not move at all -- which
        # is exactly where lambda drove the policy (three runs collapsed to
        # success ~0 once lambda exceeded ~1).  Applied once per episode, at the
        # step the robot fails, so the trainer's per-step accumulation sees it.
        if self.cost_fail_penalty > 0.0 or self.rew_timeout > 0.0:
            failed = (self.crashed | (timeout.unsqueeze(-1) & ~self.reached)) \
                & (~self._fail_counted)
            if bool(failed.any()):
                if self.cost_fail_penalty > 0.0:
                    self._cost_terminal = self._cost_terminal + \
                        failed.float()*self.cost_fail_penalty
                if self.rew_timeout > 0.0:
                    rew_extra = (-self.rew_timeout*failed.float()
                                 ).sum(dim=-1, keepdim=True)
                else:
                    rew_extra = None
                self._fail_counted = self._fail_counted | failed
            else:
                rew_extra = None
        else:
            rew_extra = None
        rr = self._overlaps_rr(); rp = self._overlaps_rp()
        ob = self._overlaps_ob()
        if self.collision_terminal:
            # per-robot flags (the *_overlaps_* helpers only answer per-env)
            hit = torch.zeros_like(self.crashed)
            if self.P:
                # NOTE: .norm(dim=-1), NOT .norm(-1) -- the latter treats -1 as
                # the p-order and silently returns a scalar, which made every
                # robot look "crashed" on the first step.
                d = (self.robo_pos.unsqueeze(2) - self.ped_pos.unsqueeze(1)).norm(dim=-1)
                hit = hit | (d < self.coll_rp).any(dim=-1)
            if self.N > 1:
                d = torch.cdist(self.robo_pos, self.robo_pos) + \
                    torch.eye(self.N, device=self.device) * 1e3
                hit = hit | (d < self.coll_rr).any(dim=-1)
            if self.O:
                d = (self.robo_pos.unsqueeze(2) - self.ob_pos.unsqueeze(1)).norm(dim=-1)
                hit = hit | (d < (self.r + self.ob_r)).any(dim=-1)
            self.crashed = self.crashed | hit
        rew = rew - self.rew_rr*rr.float().unsqueeze(-1) \
              - self.rew_rp*rp.float().unsqueeze(-1) \
              - self.rew_ob*ob.float().unsqueeze(-1)
        if self.P and (self.w_discom > 0.0 or self.w_ttc > 0.0):
            social = self._social_step()
            rew = rew - social
        # ---- affective reward (E-SocialNav) ------------------------------
        # Penalise collective pedestrian distress.  This is the term that gives
        # the policy a reason to manage mood; without it mood is only logged.
        # Kept as a soft penalty (w_emo) so it shapes rather than dominates.
        if self.emo["enabled"] and self.P and self.w_emo > 0.0:
            if self.emo_shaping_mode == "tail":
                # SAME signal as the CVaR constraint (per-robot, no /P), but
                # inside the return instead of in a dual variable.  Motivation:
                # a constrained agent at an infeasible budget reduces cost by
                # whatever means is cheapest in policy space (creep/stop);
                # a shaped agent that keeps its progress reward strictly
                # prefers "go around and keep moving" over "stop", because the
                # latter forfeits progress while paying the same penalty.
                rew = rew - self.w_emo * self._ped_cost_base()
            else:
                distress = torch.clamp(-self.ped_mood, 0.0, 1.0)      # (B,P)
                rew = rew - self.w_emo * distress.mean(dim=-1, keepdim=True)
        rew = rew + self.rew_step
        if rew_extra is not None:
            # one-off timeout/collision penalty for the robots that just failed
            rew = rew + rew_extra
        self.ep_rew += rew.detach().mean(dim=-1)

        timeout = self.step_i >= self.horizon
        succ_env = (self.reached.all(dim=-1)) & (~self.crashed.any(-1))
        crashed_env = (self.crashed.any(dim=-1) if self.collision_terminal
                       else self.crashed.all(dim=-1))
        trunc = timeout.clone()
        done = (succ_env | timeout | crashed_env).bool()
        info = self._collect_info(done, succ_env)

        if self.auto_reset and done.any():
            for e in done.nonzero(as_tuple=False).flatten().tolist():
                self._replay_env(e)
        return self.get_obs(), self.get_state(), rew, done, trunc, info

    def respawn_done_envs(self, done):
        """evaluation helper: reset only finished envs, return (obs,state)."""
        idx = done.nonzero(as_tuple=False).flatten().tolist()
        for e in idx:
            self._replay_env(e)
        return self.get_obs(), self.get_state()

    def _refresh_metrics(self):
        dev = self.device
        if self.N > 1:
            dd = torch.cdist(self.robo_pos, self.robo_pos)
            eye = torch.eye(self.N, device=dev, dtype=torch.bool).unsqueeze(0)
            dd = dd.masked_fill(eye, float("inf"))
            self.rr_min = torch.minimum(
                self.rr_min, dd.min(dim=-1).values.min(dim=-1).values)
            self.rr_coll += (dd < self.coll_rr).any(dim=(-1, -2)).float()
        if self.P:
            dp = (self.robo_pos.unsqueeze(2) -
                  self.ped_pos.unsqueeze(1)).norm(dim=-1)
            self.rp_min = torch.minimum(self.rp_min, dp.amin(dim=(1, 2)))
            self.rp_coll += (dp < self.coll_rp).any(dim=(1, 2)).float()

    def _social_step(self):
        """personal-space(discomfort) + TTC penalties per agent (week-3)."""
        rel = self.robo_pos.unsqueeze(2) - self.ped_pos.unsqueeze(1)  # B,N,P,2
        dist = rel.norm(dim=-1) + 1e-3
        relv = self.robo_vel.unsqueeze(2) - self.ped_vel.unsqueeze(1)
        # closing approach (positive means moving closer)
        closing = -(rel * relv).sum(-1) / dist.clamp_min(1e-4)
        closing = closing.clamp(min=0.0)                  # B,N,P

        # discomfort (soft-space): near = exp of decreasing distance
        act = dist < (self.d_soc + 0.25)
        expu = torch.exp(-dist / max(self.d_close, 1e-3)).clamp(0, 1.2)
        discom = expu * act.float()
        mind = dist.min(dim=-1).values                    # B,N

        discom_sum = discom.sum(dim=-1)                   # B,N per-agent
        if self.w_discom > 0:
            pen_d = self.w_discom * discom_sum.clamp(max=2.5)
        else:
            pen_d = torch.zeros_like(discom_sum)

        # TTC risk (only when metres apart reasonable and closing)
        scope = (dist < (self.r + self.ped_r + 0.9))
        ttc = dist / (closing + 1e-3)
        risk = torch.where(scope & (closing > 1e-3),
                           torch.exp(-ttc / max(self.ttc_tau, 1e-3)), 0.)
        if self.w_ttc > 0:
            pen_t = self.w_ttc * (risk.sum(dim=-1)).clamp(max=3.0)
        else:
            pen_t = torch.zeros_like(discom_sum)

        # per-episode social bookkeeping (any robot -> env)
        viol = (mind < self.d_soc).any(dim=-1).float()
        near = (mind < self.near_t).any(dim=-1).float()
        self.ep_discom_steps += viol
        self.ep_discom_sum += discom_sum.detach().mean(dim=-1)
        self.ep_rp_under += viol
        self.ep_near += near
        return pen_d + pen_t

    def risk_signal(self, cfg_risk=None):
        """RA-CMAPPO risk terms of current state (anisotropic risk field).

        Extended to PEER ROBOTS (cfg_risk["w_peer"] > 0).

        Why: measured interaction structure shows that going from N=2 to N=3
        shifts the dominant challenge from pedestrian avoidance to robot-robot
        coordination (robot-robot proximity 9.6% -> 25.1% of steps, while
        robot-pedestrian proximity barely moves 8.6% -> 9.9%).  The original
        risk field only modelled pedestrians, so at N>=3 its mechanism no longer
        targeted the binding constraint.  Adding a peer term applies the SAME
        risk machinery to the second entity class instead of changing the method.
        """
        from .risk_field import full_risk   # local import avoids cycles
        cfg_risk = cfg_risk or {}
        B, N = self.B, self.N
        base = full_risk(self.robo_pos, self.robo_vel,
                         self.ped_pos, self.ped_vel, self.ped_heading,
                         dt=self.action_dt, cfg_risk=cfg_risk) if self.P else None
        if base is None:
            z = torch.zeros(B, N, device=self.device)
            base = {"risk": z, "min_ttc": z + float("inf"),
                    "ttc_low1": z, "ttc_low2": z,
                    "directional_soc": z, "dist_min": z, "d_soc": z}

        w_peer = float(cfg_risk.get("w_peer", 0.0))
        if w_peer <= 0.0 or N < 2:
            return base

        # ---- peer-robot risk: same field, applied to robot-robot pairs ----
        # build per-robot "neighbour" tensors by rolling the robot axis
        d_safe = float(cfg_risk.get("d_safe", 0.25))
        sig_d = float(cfg_risk.get("d_sigma", 0.55))
        # (B,N,N): pairwise distances, self excluded
        dmat = torch.cdist(self.robo_pos, self.robo_pos)
        eye = torch.eye(N, device=self.device, dtype=torch.bool)
        dmat = dmat.masked_fill(eye.unsqueeze(0), float("inf"))
        dmin_p = dmat.amin(dim=-1)                       # (B,N) nearest peer
        r_peer = torch.exp(-((dmin_p - d_safe) ** 2) / (sig_d ** 2))
        # peer closing speed along the line of sight
        rv = self.robo_vel.unsqueeze(2) - self.robo_vel.unsqueeze(1)  # (B,N,N,2)
        rel = self.robo_pos.unsqueeze(2) - self.robo_pos.unsqueeze(1)
        rn = rel.norm(dim=-1).clamp_min(1e-4)
        closing = -(rel * rv).sum(-1) / rn               # (B,N,N)
        closing = torch.where(eye.unsqueeze(0), torch.zeros_like(closing),
                              closing)
        dmin_closing = closing.amax(dim=-1).clamp_min(0.0)           # (B,N)
        peer_ttc = torch.where(
            dmin_closing > 1e-3,
            dmin_p / dmin_closing.clamp_min(1e-3),
            torch.full_like(dmin_p, float("inf")))
        tau = float(cfg_risk.get("ttc_tau", 0.8))
        r_ttc_p = torch.where(peer_ttc < 1e3,
                              torch.exp(-peer_ttc / tau),
                              torch.zeros_like(peer_ttc))

        peer_risk = w_peer * (r_peer + r_ttc_p)
        out = dict(base)
        out["risk"] = base["risk"] + peer_risk
        out["peer_risk"] = peer_risk
        out["min_ttc"] = torch.minimum(base["min_ttc"], peer_ttc)
        out["ttc_low1"] = torch.clamp(base["ttc_low1"]
                                      + (peer_ttc < 1.0).float(), 0, 1)
        out["ttc_low2"] = torch.clamp(base["ttc_low2"]
                                      + (peer_ttc < 2.0).float(), 0, 1)
        out["dist_min"] = torch.minimum(base["dist_min"], dmin_p)
        return out

    def _collect_info(self, done, succ_env):
        return {
            "succ": succ_env.float().detach(),
            "done": done.float().detach(),
            "rr_min": self.rr_min.clone().detach(),
            "rp_min": self.rp_min.clone().detach(),
            "rr_coll": self.rr_coll.clone().detach(),
            "rp_coll": self.rp_coll.clone().detach(),
            "ep_rew": self.ep_rew.clone().detach(),
            "soc_steps": self.ep_discom_steps.clone().detach(),
            "soc_sum": self.ep_discom_sum.clone().detach(),
            "soc_near": self.ep_near.clone().detach(),
            # affective snapshots (E-SocialNav)
            "mpi": (self.ped_mood_sum / self.ped_mood_n.clamp_min(1.0)).mean(dim=-1)
                   if self.P else torch.zeros(self.B, device=self.device),
            "wmp": ((self.ped_mood_min < self.wmp_thresh).float().mean(dim=-1))
                   if self.P else torch.zeros(self.B, device=self.device),
            # P5: cumulative dose -- mean over pedestrians, and the worst decile
            "aid": (self.ped_mood_dose.mean(dim=-1))
                   if self.P else torch.zeros(self.B, device=self.device),
            "wid": (torch.topk(self.ped_mood_dose,
                               max(int(0.2 * self.P), 1), dim=-1).values.mean(dim=-1))
                   if self.P else torch.zeros(self.B, device=self.device),
            "ari": (self.ped_mood_rec.mean(dim=-1) * self.action_dt)
                   if self.P else torch.zeros(self.B, device=self.device),
        }

    # ------------------------------------------------------------ observations
    # ------------------------------------------------- ERM (robot affect)
    def _update_robot_mood(self, prev_d, new_d):
        """Robot affect dynamics (plan_realism.md §7.1).

            da/dt = -alpha*a - beta*block - gamma*pressure + delta*courtesy
                    - eta*urgency

        Every driver is measured from observable geometry/kinematics:
          block     progress deficit vs free-space progress toward the goal
          pressure  others (pedestrians AND co-robots) inside MY personal space
          courtesy  others moving AWAY from me while near, i.e. giving me room
          urgency   I cannot reach the goal in the remaining time at full speed
        Because `pressure`/`courtesy` sum over pedestrians *and* co-robots and
        each robot's affect changes its own behaviour, a_i and a_j are coupled
        (affect contagion) -- this is what the pedestrian-only model lacks.
        """
        if not (self.robot_emo["enabled"] and self.N):
            return
        c = self.robot_emo
        d, dt = self.device, self.action_dt
        block = (1.0 - (prev_d - new_d) / max(dt, 1e-9)
                 / max(self.max_sp, 1e-6)).clamp(0.0, 1.0)
        r = c["pressure_r"]
        press = torch.zeros_like(block)
        court = torch.zeros_like(block)
        inv_sp, inv_dt = 1.0/max(self.max_sp, 1e-6), 1.0/max(dt, 1e-9)
        if self.N > 1:
            dvec = self.robo_pos.unsqueeze(2) - self.robo_pos.unsqueeze(1)
            dd = dvec.norm(dim=-1)                                    # (B,N,N)
            eye = torch.eye(self.N, device=d, dtype=torch.bool).unsqueeze(0)
            dd = dd.masked_fill(eye, 1e6)
            near = (1.0 - dd/r).clamp(0.0, 1.0)
            press = press + near.sum(-1)
            if self._rm_prev_rr is not None:
                away = ((dd - self._rm_prev_rr)*inv_dt).clamp(0.0, self.max_sp)
                court = court + (away*near*inv_sp).sum(-1)
            self._rm_prev_rr = dd.detach().clone()
        if self.P:
            dvec = self.ped_pos.unsqueeze(1) - self.robo_pos.unsqueeze(2)
            dp = dvec.norm(dim=-1)                                    # (B,N,P)
            near = (1.0 - dp/r).clamp(0.0, 1.0)
            press = press + near.sum(-1)
            if self._rm_prev_rp is not None:
                away = ((dp - self._rm_prev_rp)*inv_dt).clamp(0.0, self.max_sp)
                court = court + (away*near*inv_sp).sum(-1)
            self._rm_prev_rp = dp.detach().clone()
        rem_t = ((self.horizon - self.step_i).float()*dt).clamp_min(1e-6
                                                                   ).unsqueeze(-1)
        urg = (new_d/max(self.max_sp, 1e-6)/rem_t - 1.0).clamp(0.0, 2.0)
        # driver decomposition kept for validation / behavioural-signature
        # analysis (plan_realism.md 7.3): a pure hand-made scalar would be
        # indefensible, so each driver must be inspectable and correlated with
        # observable behaviour.
        self._rm_last = {"block": block.detach().clone(),
                         "press": press.detach().clone(),
                         "court": court.detach().clone(),
                         "urg": urg.detach().clone()}
        da = (-c["alpha"]*self.robo_mood
              - c["beta"]*block
              - c["gamma"]*press.clamp(0.0, 3.0)
              + c["delta"]*court.clamp(0.0, 3.0)
              - c["eta"]*urg)
        nxt = (self.robo_mood + dt*da).clamp(-1.0, 1.0)
        # a finished robot stops feeling (no post-terminal affect drift)
        fin = self.reached | self.crashed
        self.robo_mood = torch.where(fin, self.robo_mood, nxt)
        self.robo_stress = self.robo_stress + torch.clamp(
            -self.robo_mood, 0.0, 1.0)*dt

    def robot_mood_per_robot(self):
        """(B,N) instantaneous robot-stress rate -- the ERM constraint exposure,
        parallel to ``ped_mood_per_robot`` (plan_realism.md §7.2)."""
        return torch.clamp(-self.robo_mood, 0.0, 1.0)

    def get_obs(self):
        B, N = self.B, self.N
        d = self.device
        dirg = self.robo_goal - self.robo_pos
        own = torch.cat([
            dirg/self.size,
            (dirg.norm(dim=-1, keepdim=True)/(2*self.size)),
            self.robo_vel/self.max_sp], dim=-1)
        if self._rmood_slot:
            # ERM: the robot feels its own affect (calm <-> stressed).  Same
            # information for every arm; only the objective differs.
            rm = self.robo_mood.unsqueeze(-1)
            if self.rmood_obs_zero:            # ablation (channel -> 0)
                rm = torch.zeros_like(rm)
            own = torch.cat([own, rm], dim=-1)
        if self.anticipate_in_obs:
            # predicted intrusion peak/mean under constant velocity
            own = torch.cat([own, self.anticipate_features(self.perception_k)],
                            dim=-1)
        if self.contention_in_obs:
            # who else is heading for MY destination
            own = torch.cat([own, self.goal_contention_features(
                self.perception_k)], dim=-1)
        if not self._cap_mode:
            out = torch.zeros(B, N, self._obs_dim, device=d)
        else:
            out = torch.zeros(B, N, self._obs_dim, device=d)
        nr_self = max(N, self.cap_r) - 1
        for a in range(N):
            cols = [own[:, a]]
            # co-robots up to capacity (skip self)
            placed = 0
            for b0 in range(max(N, self.cap_r)):
                if b0 == a and b0 < N:
                    continue
                if b0 < N:
                    cols.append(torch.cat([
                        (self.robo_pos[:, b0]-self.robo_pos[:, a])/self.size,
                        (self.robo_vel[:, b0]-self.robo_vel[:, a])/self.max_sp], -1))
                else:
                    cols.append(torch.zeros(B, 4, device=d))
                placed += 1
                if placed >= nr_self:
                    break
            # pedestrians up to capacity (zero-pad missing)
            for p in range(self.cap_p if self._cap_mode else self.P):
                if p < self.P:
                    pv = (self.ped_vel[:, p]-self.robo_vel[:, a])/self.max_sp
                    if self.ped_vel_zero:      # ablation (channel -> 0)
                        pv = torch.zeros_like(pv)
                    rel_obs = [(self.ped_pos[:, p]-self.robo_pos[:, a])/self.size, pv]
                    if self._mood_obs:
                        # P1: the pedestrian's affective state, so the policy can
                        # condition its behaviour on how the human feels rather
                        # than only on where the human is.
                        m = self.ped_mood[:, p:p+1]
                        if self.mood_obs_zero:      # ablation (see __init__)
                            m = torch.zeros_like(m)
                        rel_obs.append(m)
                    cols.append(torch.cat(rel_obs, -1))
                else:
                    cols.append(torch.zeros(B, self._ped_slot, device=d))
            # obstacles
            for oi in range(self.cap_o if self._cap_mode else self.O):
                if oi < self.O:
                    cols.append(torch.cat([
                        (self.ob_pos[:, oi]-self.robo_pos[:, a])/self.size,
                        torch.zeros(B, 2, device=d)], -1))
                else:
                    cols.append(torch.zeros(B, 4, device=d))
            out[:, a] = torch.cat(cols, dim=-1)
        if self.obs_noise:
            out = out + self.obs_noise * torch.randn_like(out)
        return out

    # -------------------------------------------------- evaluation hooks
    def set_ped_override(self, ped_vel):
        """let an external planner drive the crowd on the *next* step only."""
        self._ped_override = torch.as_tensor(ped_vel, device=self.device,
                                             dtype=torch.float32)

    @torch.no_grad()
    def ped_mood_per_robot(self, sigma=1.2, mode=None):
        """Public cost accessor: social cost + one-off terminal failure penalty.

        plan_emo.md 模块2 requires the constrained cost to be
        C(tau) = D(tau) + kappa_f * I[failure] + kappa_t * T_remaining, so that
        the degenerate "stand still and harm nobody" policy is the MOST
        expensive one rather than the cheapest.  Only D used to be implemented,
        which is why lambda pushed every run into standing still.
        """
        c = self._ped_cost_base(sigma=sigma, mode=mode)
        if self.cost_fail_penalty > 0.0:
            c = c + self._cost_terminal
        return c

    def _ped_cost_base(self, sigma=1.2, mode=None):
        """Per-robot social cost (B,N) in [0,1].

        Two formulations, selected by `self.cost_mode` (or the `mode` arg):

        "deficit" (legacy) -- the pedestrian's accumulated affective deficit
            -mood, proximity-weighted over robots.

        "encroachment" (current) -- the SEVERITY of the robot's present
            invasion of the pedestrian's anisotropic social radius, i.e.
            how far inside the personal space the robot currently is.

        Why the change: with the deficit formulation the cost was a function
        of TIME-IN-PROXIMITY rather than of how badly anyone was treated.
        Measured on the dense scene with goal-directed policies, per-step cost
        fell monotonically with speed (fast 0.805 / half 2.048 / slow 3.588):
        because the robot must reach its goal within the 8 s episode, a slow
        robot simply remains inside pedestrians' social radius for the whole
        episode and accrues intrusion every step.  The "safest" policy was
        therefore the one that rushed through the crowd fastest, so there was
        no safety/efficiency trade-off for the CVaR constraint to resolve --
        any apparent benefit would have been an artefact of that degeneracy.

        Under the encroachment formulation, being deep inside someone's
        personal space is expensive and skimming the edge of it is nearly
        free, so the two genuine options (rush past everyone vs. give way and
        let people pass) are separated by how close the robot actually gets,
        not by how long the episode lasts.
        """
        B, N = self.B, self.N
        if self.P == 0 or not getattr(self, "_mood_obs", False) and not self.emo["enabled"]:
            return torch.zeros(B, N, device=self.device)
        rel = self.robo_pos.unsqueeze(2) - self.ped_pos.unsqueeze(1)   # B,N,P,2
        dist = rel.norm(dim=-1).clamp_min(1e-4)
        mode = mode or getattr(self, "cost_mode", "encroachment")
        if mode == "deficit":
            deficit = torch.clamp(-self.ped_mood, 0.0, 1.0)            # B,P
            w = torch.softmax(-dist / sigma, dim=1)                    # B,N,P
            return (w * deficit.unsqueeze(1)).sum(dim=-1)              # B,N
        if mode == "mood_impact":
            # CONSTRAINT COST = instantaneous harm q_t, attributed to robots.
            # q is a SINGLE integral (q itself is not a memory of intrusion),
            # which is what avoids the double-integration artefact: the state
            # layer `-mood` already integrates intrusion, so integrating it
            # again ranked a slow policy as more harmful than a fast one.
            #
            # NORMALISATION: the weighted sum runs over pedestrians, so without
            # dividing by P the cost grows with the SIZE OF THE CROWD rather
            # than with the severity of harm (measured: ped_q=0.027 but the
            # summed per-robot cost was 0.17-0.33, a 6-12x inflation).  A cost
            # that scales with pedestrian count would make the constraint mean
            # "there are people here" instead of "someone was harmed", and
            # would make cost_target non-transferable across densities --
            # fatal for the cross-density generalisation experiments.  Scaling
            # by P keeps the cost an average per-pedestrian harm in [0, ~1].
            #
            # BUT: that /P also dilutes the signal.  In a large, sparse
            # workspace most pedestrians never come near the robot, so the
            # per-pedestrian AVERAGE is ~0 (measured 0.0002/step vs 0.03/step on
            # the old dense scene) while the worst pedestrian still accumulates
            # a dose of 13.  The average therefore hides exactly the person the
            # constraint is supposed to protect.  Use "mood_tail" below when the
            # objective is the worst-treated individual.
            w = self.responsibility_weights(
                getattr(self, "resp_mode", "dynamic"))                 # B,N,P
            return (w * self.ped_q.unsqueeze(1)).sum(dim=-1) \
                / max(self.P, 1)                                       # B,N
        if self.cost_fail_penalty > 0.0:
            pass  # terminal penalty is added by ped_mood_per_robot()
        if mode == "mood_tail":
            # TAIL cost: the harm I am currently doing to the WORST-TREATED
            # single pedestrian (no /P dilution).  This is the per-step analogue
            # of "CVaR over pedestrians of the cumulative dose" and it is what
            # gives yielding a payoff: protecting the most harmed person now
            # changes the constrained quantity, whereas improving the crowd
            # average by 1/P does not.
            w = self.responsibility_weights(
                getattr(self, "resp_mode", "dynamic"))                 # B,N,P
            return (w * self.ped_q.unsqueeze(1)).max(dim=-1).values    # B,N
        # ---- "encroachment": severity of the CURRENT invasion ------------
        bear = torch.atan2(rel[..., 1], rel[..., 0]) \
            - self.ped_heading.unsqueeze(1)                            # B,N,P
        bear = torch.atan2(torch.sin(bear), torch.cos(bear))
        from .risk_field import anisotropic_social_radius
        cfg = self.emo
        d_soc = anisotropic_social_radius(
            bear, d_front=cfg["d_front"], d_side=cfg["d_side"],
            d_back=cfg["d_back"])                                      # B,N,P
        # normalized penetration depth in [0,1]: 0 at the radius boundary,
        # 1 at zero separation.  Squared so shallow intrusion is cheap and a
        # genuine violation dominates -- this is what makes "brushing past the
        # edge of personal space" acceptable while "walking through someone"
        # is not.
        pen = torch.clamp((d_soc - dist) / d_soc.clamp_min(1e-6), 0.0, 1.0)
        pen = pen * pen                                                # B,N,P
        # max over pedestrians: the cost is the WORST person the robot is
        # currently imposing on, which is what a CVaR welfare constraint
        # should protect
        return pen.amax(dim=2)                                         # B,N

    @torch.no_grad()
    def responsibility_weights(self, mode="dynamic"):
        """Per-(robot, pedestrian) responsibility weights (B,N,P), summing to 1
        over robots for each pedestrian.

        A pure distance softmax mis-attributes harm: the NEAREST robot is not
        necessarily the one causing it (a robot standing close but moving slowly
        is less harmful than one slightly further away cutting across the
        pedestrian's path at speed).  So the weighting optionally includes the
        closing speed and the anisotropic intrusion, both of which are actual
        causal factors rather than mere proximity.

        mode: "distance"  -> exp(-d/sigma_d)                      (baseline)
              "dynamic"   -> + closing speed + anisotropic intrusion
        """
        rel = self.robo_pos.unsqueeze(2) - self.ped_pos.unsqueeze(1)   # B,N,P,2
        dist = rel.norm(dim=-1).clamp_min(1e-4)
        logits = -dist / max(float(self.emo.get("sigma_d", 0.5)), 1e-6)
        if mode == "dynamic":
            relv = self.robo_vel.unsqueeze(2) - self.ped_vel.unsqueeze(1)
            closing = -(rel * relv).sum(-1) / dist                     # >0 approach
            bear = torch.atan2(rel[..., 1], rel[..., 0]) \
                - self.ped_heading.unsqueeze(1)
            bear = torch.atan2(torch.sin(bear), torch.cos(bear))
            from .risk_field import anisotropic_social_radius
            d_soc = anisotropic_social_radius(
                bear, d_front=self.emo["d_front"], d_side=self.emo["d_side"],
                d_back=self.emo["d_back"])
            intr = torch.clamp(d_soc - dist, 0.0, None)
            logits = (logits
                      + closing / max(float(self.emo.get("sigma_v", 0.5)), 1e-6)
                      + intr / max(float(self.emo.get("sigma_i", 0.5)), 1e-6))
        return torch.softmax(logits, dim=1)                            # B,N,P

    @torch.no_grad()
    def anticipate_features(self, k=8):
        """Short-horizon predicted intrusion, per robot: (B, N, 2).

        Innovation 3 (anticipatory safety critic).  Returns
        [peak predicted intrusion, mean predicted intrusion] under constant
        velocity, so the critic can value *pre-emptive* yielding -- moving aside
        now to avoid distressing a pedestrian a fraction of a second later.
        A purely reactive cost (current intrusion only) cannot express this.
        """
        if self.P == 0 or not self.N or k <= 0:
            B, N = self.B, self.N
            return torch.zeros(B, N, 2, device=self.device)
        dt = self.action_dt
        peak = torch.zeros(self.B, self.N, device=self.device)
        tot = torch.zeros_like(peak)
        from .risk_field import anisotropic_social_radius
        for j in range(1, int(k) + 1):
            rp = self.robo_pos + self.robo_vel * (j * dt)
            pp = self.ped_pos + self.ped_vel * (j * dt)
            rel = rp.unsqueeze(2) - pp.unsqueeze(1)              # B,N,P,2
            dist = rel.norm(dim=-1).clamp_min(1e-4)
            ph = self.ped_heading
            bear = torch.atan2(rel[..., 1], rel[..., 0]) - ph.unsqueeze(1)
            bear = torch.atan2(torch.sin(bear), torch.cos(bear))
            d_soc = anisotropic_social_radius(
                bear, d_front=self.emo["d_front"], d_side=self.emo["d_side"],
                d_back=self.emo["d_back"])
            intr = torch.clamp(d_soc - dist, 0.0, None).amax(dim=2)   # B,N
            peak = torch.maximum(peak, intr)
            tot = tot + intr
        return torch.stack([peak, tot / float(k)], dim=-1)

    def get_state(self):
        B, N = self.B, self.N
        if not self._cap_mode:
            cols = [self.robo_pos.reshape(B, -1)/self.size,
                    self.robo_vel.reshape(B, -1)/self.max_sp,
                    self.robo_goal.reshape(B, -1)/self.size]
            if self.P:
                cols += [self.ped_pos.reshape(B, -1)/self.size,
                         self.ped_vel.reshape(B, -1)/self.max_sp]
            if self.O:
                cols += [self.ob_pos.reshape(B, -1)/self.size]
            if self.robot_emo["enabled"]:
                # ERM: every robot's affect is part of the CENTRAL state, so the
                # centralised critic (and the robot-stress cost critic) can see
                # the whole coupled affective field.
                cols += [self.robo_mood.reshape(B, -1)]
            return torch.cat(cols, dim=-1)
        # capacity mode: fixed-size central state
        nr = max(N, self.cap_r); np_ = max(self.P, self.cap_p)
        no_ = max(self.O, self.cap_o)

        def _pad_agents(attr, slots, divv):
            x = torch.zeros(B, slots, 2, device=self.device)
            n_avail = min(attr.shape[1], slots)
            x[:, :n_avail] = attr[:, :n_avail].clone()
            return x.reshape(B, -1)/divv
        cols = [_pad_agents(self.robo_pos.clamp(-self.size, self.size), nr, self.size),
                _pad_agents(self.robo_vel, nr, max(self.max_sp, 1e-6)),
                _pad_agents(self.robo_goal.clamp(-self.size, self.size), nr, self.size)]
        if self.P or np_:
            ppos = self.ped_pos.clamp(-self.size, self.size)
            cols += [_pad_agents(ppos, np_, self.size),
                     _pad_agents(self.ped_vel, np_, max(self.max_sp, 1e-6))]
        if no_ :
            cols += [_pad_agents(self.ob_pos if self.O else torch.zeros(B, 0, 2,
                                    device=self.device),
                                 no_, self.size)]
        if getattr(self, "_anticipate_k", 0):
            cols += [self.anticipate_features(self._anticipate_k).reshape(B, -1)]
        if self.robot_emo["enabled"]:
            cols += [self.robo_mood.reshape(B, -1)]
        return torch.cat(cols, dim=-1)
