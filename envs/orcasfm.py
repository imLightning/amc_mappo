"""orcasfm.py -- classical, non-learned navigation controllers for the paper's
baseline set.

Two methods, both implemented in THIS environment (10x10 m, N robots, P
pedestrians, circular obstacles) and both evaluated through the same
`scripts/pareto_eval.py` protocol as the learned arms:

  ORCAPolicy : the real ORCA algorithm (van den Berg et al. 2011) via RVO2.
               All robots AND all pedestrians are RVO2 agents, so the avoidance
               is genuinely reciprocal; pedestrians keep their environment
               velocity as their preferred velocity.  Circular obstacles are
               approximated by inscribed 12-gons (RVO2 takes line-segment
               obstacles).  This is the classical safety baseline.

  SFMPolicy  : the social force model as a ROBOT controller (Helbing & Molnar
               1995): goal force + pedestrian repulsion + robot-robot
               repulsion + obstacle repulsion, integrated at the environment's
               own acceleration limit.  This is the classical SOCIAL baseline
               (it respects personal space, not just collisions).

Why they live here and not in algorithms/baselines.py: that file's "ORCAPolicy"
is a heuristic repulsion potential, NOT ORCA.  Publishing it under the name ORCA
would be wrong, so the real one is implemented separately.

Usage from pareto_eval:  --policy orca   /   --policy sfm
"""
from __future__ import annotations

import math
import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  ORCA (RVO2)
# --------------------------------------------------------------------------- #
class ORCAPolicy:
    def __init__(self, neighbor=3.0, max_neighbors=12, horizon=2.0,
                 horizon_obst=1.0, safety=0.05):
        self.neighbor = float(neighbor)
        self.max_neighbors = int(max_neighbors)
        self.horizon = float(horizon)
        self.horizon_obst = float(horizon_obst)
        self.safety = float(safety)      # extra radius (m) on top of the real one

    @torch.no_grad()
    def act(self, env):
        import rvo2
        B, N, P = env.B, env.N, env.P
        dt = float(env.action_dt)
        vmax = float(env.max_sp)
        out = torch.zeros(B, N, 2, device=env.device)
        r_rob = float(env.r)
        r_ped = float(env.ped_r)
        ob_r = float(env.ob_r) if env.O else 0.0
        robo = env.robo_pos.detach().cpu().numpy()
        rvel = env.robo_vel.detach().cpu().numpy()
        ped = env.ped_pos.detach().cpu().numpy() if P else np.zeros((B, 0, 2))
        pvel = env.ped_vel.detach().cpu().numpy() if P else np.zeros((B, 0, 2))
        goal = env.robo_goal.detach().cpu().numpy()
        ob = env.ob_pos.detach().cpu().numpy() if env.O else np.zeros((B, 0, 2))
        for b in range(B):
            sim = rvo2.PyRVOSimulator(dt, self.neighbor, self.max_neighbors,
                                      self.horizon, self.horizon_obst,
                                      r_rob + self.safety, vmax)
            # NOTE: rvo2's addAgent takes EITHER just a position (then the
            # simulator defaults apply) OR *all eight* parameters, i.e.
            # (pos, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst,
            #  radius, maxSpeed, velocity).  Passing seven raises
            # "Either pass only 'pos', or pass all parameters."
            for n in range(N):
                sim.addAgent(tuple(robo[b, n]), self.neighbor, self.max_neighbors,
                             self.horizon, self.horizon_obst,
                             r_rob + self.safety, vmax,
                             (float(rvel[b, n, 0]), float(rvel[b, n, 1])))
            for p in range(P):
                pv = (float(pvel[b, p, 0]), float(pvel[b, p, 1]))
                sim.addAgent(tuple(ped[b, p]), self.neighbor, self.max_neighbors,
                             self.horizon, self.horizon_obst,
                             r_ped, float(np.linalg.norm(pv)) + 1e-3, pv)
            # circular obstacles -> inscribed polygon
            for o in range(ob.shape[1]):
                cx, cy = ob[b, o]
                verts = []
                for k in range(12):
                    a = 2 * math.pi * k / 12
                    verts.append((float(cx + ob_r * math.cos(a)),
                                  float(cy + ob_r * math.sin(a))))
                sim.addObstacle(verts)
            sim.processObstacles()
            # preferred velocities
            for n in range(N):
                d = goal[b, n] - robo[b, n]
                nrm = float(np.linalg.norm(d))
                pref = (d / nrm * vmax) if nrm > 1e-6 else np.zeros(2)
                sim.setAgentPrefVelocity(n, (float(pref[0]), float(pref[1])))
            for p in range(P):
                sim.setAgentPrefVelocity(N + p, (float(pvel[b, p, 0]),
                                                 float(pvel[b, p, 1])))
            sim.doStep()
            for n in range(N):
                v = sim.getAgentVelocity(n)
                out[b, n, 0], out[b, n, 1] = float(v[0]), float(v[1])
        return out


# --------------------------------------------------------------------------- #
#  Social force model (as a robot controller)
# --------------------------------------------------------------------------- #
class SFMPolicy:
    """Helbing-style forces, integrated at the env's acceleration limit.

    A_ped, B_ped control the pedestrian repulsion (A in m/s^2, B in m),
    A_rob/B_rob the robot-robot term, A_ob/B_ob the obstacle term.
    The 'force' is turned into a velocity by v <- v + (F - k*(v - v_des))*dt,
    i.e. a first-order relaxation toward the desired velocity, then the command
    is clipped to vmax (the environment applies its own acceleration limit).
    """

    def __init__(self, A_ped=8.0, B_ped=0.35, A_rob=6.0, B_rob=0.30,
                 A_ob=10.0, B_ob=0.20, k=1.5):
        self.A_ped, self.B_ped = float(A_ped), float(B_ped)
        self.A_rob, self.B_rob = float(A_rob), float(B_rob)
        self.A_ob, self.B_ob = float(A_ob), float(B_ob)
        self.k = float(k)

    @torch.no_grad()
    def act(self, env):
        dt = float(env.action_dt)
        vmax = float(env.max_sp)
        pos = env.robo_pos                     # (B,N,2)
        B, N, _ = pos.shape
        d = env.robo_goal - pos
        dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_des = d / dist * vmax
        F = self.k * (v_des - env.robo_vel)

        if env.P:
            rel = pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)      # B,N,P,2
            dd = rel.norm(dim=-1).clamp_min(1e-4)
            gap = (dd - (env.r + env.ped_r)).clamp_min(0.0)
            mag = self.A_ped * torch.exp(-gap / self.B_ped)
            F = F + (rel / dd.unsqueeze(-1) * mag.unsqueeze(-1)).sum(dim=2)
        if N > 1:
            rel = pos.unsqueeze(2) - pos.unsqueeze(1)
            dd = rel.norm(dim=-1).clamp_min(1e-4)
            eye = torch.eye(N, device=env.device).bool().unsqueeze(0)
            dd = dd.masked_fill(eye, 1e4)
            gap = (dd - 2 * env.r).clamp_min(0.0)
            mag = self.A_rob * torch.exp(-gap / self.B_rob)
            F = F + (rel / dd.unsqueeze(-1) * mag.unsqueeze(-1)).sum(dim=2)
        if env.O:
            rel = pos.unsqueeze(2) - env.ob_pos.unsqueeze(1)
            dd = rel.norm(dim=-1).clamp_min(1e-4)
            gap = (dd - (env.r + env.ob_r)).clamp_min(0.0)
            mag = self.A_ob * torch.exp(-gap / self.B_ob)
            F = F + (rel / dd.unsqueeze(-1) * mag.unsqueeze(-1)).sum(dim=2)

        v = env.robo_vel + F * dt
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        v = v * torch.clamp(vmax / n, max=1.0)
        return v


# --------------------------------------------------------------------------- #
#  Dynamic Window Approach (classic local planner)
# --------------------------------------------------------------------------- #
class DWAPolicy:
    """Sampling local planner over (v, omega) inside the dynamic window.

    score = alpha * heading(v,omega) + beta * clearance(v,omega) + gamma * velocity

    The robot's current velocity direction plays the role of the heading (our
    action space is a velocity vector, not a unicycle).  Dynamic window:
        v in [v_cur - a_v*dt, v_cur + a_v*dt] cap [0, v_max]
        w in [w_cur - a_w*dt, w_cur + a_w*dt] cap [-w_max, w_max]
    with a_v taken from the environment's acceleration limit.
    """

    def __init__(self, alpha=1.0, beta=0.25, gamma=0.2, aw=3.0,
                 wmax=2.0, nv=5, nw=9, horizon=1):
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.aw, self.wmax = float(aw), float(wmax)
        self.nv, self.nw = int(nv), int(nw)
        self.horizon = int(horizon)

    @torch.no_grad()
    def act(self, env):
        dt = float(env.action_dt)
        vmax = float(env.max_sp)
        av = float(getattr(env, "max_accel", 0.0) or 1.0)
        pos, vel = env.robo_pos, env.robo_vel
        B, N, _ = pos.shape
        dev = pos.device
        speed = vel.norm(dim=-1)                       # (B,N)
        # heading: use the velocity direction when moving, otherwise aim at the
        # goal (atan2(0,0)=0 would otherwise point every robot along +x)
        g_dir = env.robo_goal - pos
        head = torch.where(
            (speed > 0.05).logical_and(vel.norm(dim=-1) > 0),
            torch.atan2(vel[..., 1], vel[..., 0]),
            torch.atan2(g_dir[..., 1], g_dir[..., 0]))

        # Candidate velocity inside the dynamic window of the CURRENT speed.
        # A fixed absolute grid (0..vmax) contains no candidate within a_max*dt
        # of a standstill robot, so the only feasible action would be v=0.
        v_lo = (speed - av * dt).clamp(0.0, vmax)
        v_hi = (speed + av * dt).clamp(0.0, vmax)
        rel = torch.linspace(0.0, 1.0, self.nv, device=dev)
        v = (v_lo.unsqueeze(-1) + (v_hi - v_lo).unsqueeze(-1) * rel)       # (B,N,V)
        v = v.unsqueeze(-1).expand(B, N, self.nv, self.nw)
        # Heading change per step is bounded by omega_max*dt.  The first version
        # bounded omega itself by a_w*dt (0.15 rad/s), i.e. 0.4 deg per step, so
        # a robot could not turn 90 deg within an episode (success stayed 0.000
        # for every parameter set) -- sample the heading CHANGE instead.
        dth = torch.linspace(-1.0, 1.0, self.nw, device=dev) * (self.wmax * dt)
        th = (head.unsqueeze(-1).unsqueeze(-1)
              + dth.view(1, 1, 1, self.nw)).expand(B, N, self.nv, self.nw)

        # roll the candidate commands forward `horizon` steps to score clearance
        px = pos[..., 0].unsqueeze(-1).unsqueeze(-1)
        py = pos[..., 1].unsqueeze(-1).unsqueeze(-1)
        for _k in range(max(self.horizon, 1)):
            px = px + v * torch.cos(th) * dt
            py = py + v * torch.sin(th) * dt

        gl = (env.robo_goal - pos)
        gl = gl / gl.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gth = torch.atan2(gl[..., 1], gl[..., 0]).unsqueeze(-1).unsqueeze(-1)
        heading = torch.cos(th - gth)

        clear = torch.full_like(v, 1e3)
        if env.P:
            dx = px.unsqueeze(-1) - env.ped_pos[:, None, None, None, :, 0]
            dy = py.unsqueeze(-1) - env.ped_pos[:, None, None, None, :, 1]
            clear = torch.minimum(clear, (dx ** 2 + dy ** 2).sqrt().amin(dim=-1))
        if env.O:
            dx = px.unsqueeze(-1) - env.ob_pos[:, None, None, None, :, 0]
            dy = py.unsqueeze(-1) - env.ob_pos[:, None, None, None, :, 1]
            clear = torch.minimum(clear, (dx ** 2 + dy ** 2).sqrt().amin(dim=-1))
        clear_n = (clear / 1.5).clamp(0.0, 1.0)

        score = self.alpha * heading + self.beta * clear_n + self.gamma * (v / vmax)
        idx = score.reshape(B, N, -1).argmax(dim=-1)
        # .reshape, not .view: the expanded candidate tensors are not contiguous
        vv = v.reshape(B, N, -1).gather(2, idx.unsqueeze(-1)).squeeze(-1)
        th_new = th.reshape(B, N, -1).gather(2, idx.unsqueeze(-1)).squeeze(-1)
        return torch.stack([vv * torch.cos(th_new), vv * torch.sin(th_new)], dim=-1)
