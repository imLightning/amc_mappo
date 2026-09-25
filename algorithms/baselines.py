"""
baselines.py -- non-learning & ablated-learning baselines (plan_add §五).

  * ORCA (reciprocal velocity obstacle) local planner over robot states;
  * function to select the MAPPO-family ablated mode string used by ra_cmappo:
      MAPPO-fixed      (risk_mode='none', no safety term)
      MAPPO+Lagrangian (risk_mode='adaptive' with fixed schedule)
      RA-CMAPPO        (risk_mode='adaptive', anisotropic field on)

ORCA here is a two-agent/vo simple implementation for velocity-per-agent; it
keeps pairwise reciprocal half-plane avoidance and clamps to vmax.
"""
import numpy as np
import torch


def reciprocal_orca_step(positions, velocities, goals, r_agents=0.5,
                         vmax=1.2, dt=0.05, horizon=2.0):
    """one ORCA-style update for N agents (positions/velocities (N,2)).
    Returns desired velocities (N,2) as numpy array."""
    pos = np.asarray(positions, float)
    vel = np.asarray(velocities, float)
    goals = np.asarray(goals, float)
    N = pos.shape[0]
    pref = goals - pos
    dn = np.linalg.norm(pref, axis=1, keepdims=True).clip(min=1e-3)
    pref_v = pref / dn * vmax
    out = pref_v.copy()
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rel = pos[j] - pos[i]
            d = np.linalg.norm(rel)
            if d < 1e-3:
                continue
            # time-to-closest approach along relative velocity
            relv = vel[i] - vel[j]
            t = min(-np.dot(rel, relv) / (np.dot(relv, relv) + 1e-6), horizon)
            # closest approach point
            p = rel + relv * t
            r = max(2 * r_agents, 0.5)
            pn = np.linalg.norm(p)
            if pn < r:
                u = p / max(pn, 1e-3)
                # half-plane normal (ORCA-ish): push away from collision
                out[i] += (r - pn) * u * 2.0
    n = np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-3)
    return (out / n * np.minimum(n, vmax)).astype(float)


class ORCAPolicy:
    """ORCA-style deterministic local avoidance.
    For multi-robot (N>1): all robots mutually avoid (reciprocal).
    In pedestrian scenes the robot also avoids each moving pedestrian using
    a velocity-obstacle repulsion term with their current velocity — the classic
    single-ego CrowdNav ORCA-style reference.  Actions: (B,N,2) continuous
    velocities, torque clipped by env.
    """

    def __init__(self, vmax=None, r_agent=0.5, influence=2.0, avoidance=1.5):
        self.vmax = vmax
        self.avoid = avoidance
        self.r = r_agent
        self.infl = influence

    def predict_env(self, env):
        """vectorized batch: return (B,N,2) torch velocity commands."""
        dev = env.device
        pos = env.robo_pos                    # (B,N,2)
        B, N, _ = pos.shape
        self.vmax = self.vmax or float(getattr(env, "max_sp", 1.2))
        g = env.robo_goal - pos
        vel = g / (g.norm(dim=-1, keepdim=True).clamp_min(1e-4)) * self.vmax
        if N > 1:  # reciprocal robot-robot
            rel = pos.unsqueeze(2) - pos.unsqueeze(1)          # B,N,N,2
            dist = rel.norm(dim=-1)
            eye = torch.eye(N, device=dev, dtype=torch.bool)[None]
            dist = dist.masked_fill(eye, float("inf"))
            near = dist < self.infl
            if near.any():
                u = rel / (dist.unsqueeze(-1).clamp_min(1e-3))
                w = (1.0 - dist / self.infl).clamp(0, 1)
                vel = vel + (u * w.unsqueeze(-1)).sum(dim=2) * self.avoid
        if getattr(env, "P", 0) > 0:         # robot vs pedestrians
            pp = env.ped_pos
            relp = pos[:, :, None] - pp[:, None]              # B,N,P,2
            dp = relp.norm(dim=-1)
            active = (dp < self.infl) & (dp > 1e-3)
            if active.any():
                u = relp / (dp.unsqueeze(-1).clamp_min(1e-3))
                w = (1.0 - dp / self.infl).clamp_min(0)
                vel = vel + (u * w.unsqueeze(-1)).sum(dim=2) * self.avoid
        sp = vel.norm(dim=-1, keepdim=True)
        vel = vel * torch.clamp(self.vmax / sp.clamp_min(1e-3), None, 1.0)
        # guard shape
        return vel.reshape(B, N, 2)

    # compat: single-episode predict used elsewhere
    def predict(self, env, obs=None):
        return self.predict_env(env)[0].cpu().numpy()

    def actor(self):
        """returns callable (obs, env)->(B,N,2) for eval_protocol."""
        return lambda obs, env: self.predict_env(env)


class RealORCA:
    """ORCA baseline backed by the official RVO2 library (``rvo2``).

    Protocol (this is the standard CrowdNav-family ORCA protocol, documented
    explicitly so the comparison is not a strawman):
      * all agents that can move -- robots *and* pedestrians -- are modelled as
        RVO2 agents, so avoidance is genuinely reciprocal;
      * each control step the sim is re-synced to the ground-truth env state,
        preferred velocities are set toward each agent's goal/pedestrian
        velocity, ``doStep`` computes collision-free velocities, and the robot
        velocities are returned as actions for the environment to integrate.

    Because our own pedestrians normally follow a social-force walker, using
    ORCA-driven pedestrians is a *protocol choice* for this baseline; it is
    recorded here and must be stated in the paper.

    One RVO2 simulator is kept per parallel environment (they are not
    vectorised), so this is meant for evaluation, not training.
    """

    def __init__(self, neighbor_dist=2.0, max_neighbors=10, time_horizon=2.0,
                 time_horizon_obst=2.0, max_speed=None, ped_mode="reciprocal"):
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.max_speed = max_speed
        # 'reciprocal': ORCA also drives pedestrians (written back to the env),
        #               i.e. the faithful CrowdNav-protocol ORCA.
        # 'env'       : pedestrians keep the env's social-force motion and ORCA
        #               only yields to them (one-sided; understates ORCA).
        assert ped_mode in ("reciprocal", "env")
        self.ped_mode = ped_mode
        self._sims = None
        self._sig = None

    # ---------------------------------------------------------------- setup
    def _ensure(self, env):
        """create per-env simulators lazily when batch/shape changes."""
        sig = (env.B, env.N, env.P, float(getattr(env, "r", 0.25)),
               float(getattr(env, "ped_r", 0.22)),
               float(getattr(env, "max_sp", 1.2)),
               float(getattr(env, "action_dt", 0.05)))
        if self._sims is not None and sig == self._sig:
            return
        from rvo2 import PyRVOSimulator
        vmax = self.max_speed or float(getattr(env, "max_sp", 1.2))
        self.max_speed = vmax
        r_rob = float(getattr(env, "r", 0.25))
        r_ped = float(getattr(env, "ped_r", 0.22))
        n_all = env.N + env.P
        sims = []
        for b in range(env.B):
            sim = PyRVOSimulator(float(env.action_dt), self.neighbor_dist,
                                 self.max_neighbors, self.time_horizon,
                                 self.time_horizon_obst, r_ped, vmax)
            for _ in range(n_all):
                sim.addAgent((0.0, 0.0))
            sims.append(sim)
        self._sims = sims
        self._sig = sig
        self._r_rob, self._r_ped = r_rob, r_ped

    # ---------------------------------------------------------------- act
    def predict_env(self, env):
        """(B,N,2) torch velocities for the robots (RVO2 ORCA planner)."""
        import numpy as np
        self._ensure(env)
        dev = env.device
        rp = env.robo_pos.detach().cpu().numpy()
        rv = env.robo_vel.detach().cpu().numpy()
        rg = env.robo_goal.detach().cpu().numpy()
        pp = env.ped_pos.detach().cpu().numpy() if env.P else None
        pv = env.ped_vel.detach().cpu().numpy() if env.P else None
        out = np.zeros((env.B, env.N, 2), dtype=np.float32)
        vmax = self.max_speed
        for b in range(env.B):
            sim = self._sims[b]
            n_all = env.N + env.P
            # sync ground-truth state into the ORCA sim
            for i in range(env.N):
                sim.setAgentRadius(i, self._r_rob)
                sim.setAgentPosition(i, (float(rp[b, i, 0]), float(rp[b, i, 1])))
                sim.setAgentVelocity(i, (float(rv[b, i, 0]), float(rv[b, i, 1])))
                rx, ry = rg[b, i] - rp[b, i]
                n = float(np.hypot(rx, ry)) or 1.0
                sim.setAgentPrefVelocity(i, (vmax * rx / n, vmax * ry / n))
            for j in range(env.P):
                i = env.N + j
                sim.setAgentRadius(i, self._r_ped)
                sim.setAgentPosition(i, (float(pp[b, j, 0]), float(pp[b, j, 1])))
                sim.setAgentVelocity(i, (float(pv[b, j, 0]), float(pv[b, j, 1])))
                # pedestrians keep their own (env) motion as preference
                sim.setAgentPrefVelocity(i, (float(pv[b, j, 0]),
                                             float(pv[b, j, 1])))
            sim.doStep()
            for i in range(env.N):
                vx, vy = sim.getAgentVelocity(i)
                out[b, i] = (vx, vy)
        if self.ped_mode == "reciprocal" and env.P:
            # write ORCA's pedestrian velocities back into the environment so the
            # crowd is genuinely reciprocal (faithful ORCA protocol).
            pv_out = np.zeros((env.B, env.P, 2), dtype=np.float32)
            for b in range(env.B):
                for j in range(env.P):
                    vx, vy = self._sims[b].getAgentVelocity(env.N + j)
                    pv_out[b, j] = (vx, vy)
            env.set_ped_override(torch.as_tensor(pv_out, device=dev))
        return torch.as_tensor(out, device=dev)

    def actor(self):
        return lambda obs, env: self.predict_env(env)

    def predict(self, env, obs=None):
        return self.predict_env(env)[0].cpu().numpy()


BASELINE_MODES = {
    "MAPPO": dict(risk_mode="none", anisotropic=False),
    "MAPPO_fixed": dict(risk_mode="fixed", anisotropic=False),
    "MAPPO_Lagrangian": dict(risk_mode="adaptive", anisotropic=False),
    "RA_CMAPPO": dict(risk_mode="adaptive", anisotropic=True),
    # ---- Mood-Shaping: the information-matched control --------------------
    # Identical information to AMC-MAPPO (both see pedestrian mood via the
    # environment) but the mood enters the REWARD as a shaping term instead of
    # becoming a constrained objective.  This isolates "constraint
    # optimisation" from "merely knowing about affect", which is the most
    # likely reviewer objection to our claim.
    "Mood_Shaping": dict(risk_mode="none", anisotropic=False),
}


# Environment-side deltas of each mode (the reward/constraint split lives
# partly in the environment, so a mode cannot be fully described by algo keys).
MODE_ENV = {
    # Mood-Shaping sees exactly the information the constrained arm sees, and
    # penalises exactly the same per-robot affect signal -- only the mechanism
    # differs (return shaping vs dual ascent on a CVaR constraint).  The
    # coefficient itself comes from the recipe / --emo_reward_weight.
    "Mood_Shaping": dict(emotion_shaping_mode="tail"),
}


def baseline_cfg(mode, cfg_algo):
    """return algo config updated for a given ablated mode."""
    import copy
    c = copy.deepcopy(dict(cfg_algo))
    c.update(BASELINE_MODES[mode])
    return c
