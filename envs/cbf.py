"""cbf.py -- discrete-time control-barrier-function velocity filter.

Why a filter and not a training term: in this project the learned policies never
acquired pedestrian avoidance under any reward/task design we tried (broken
per-robot penalty, fixed per-robot penalty, fatal contacts -- NIGHT_REPORT
sections 15-16), while an analytic controller reaches 2.3x their social success.
A CBF filter is the standard way to make "keep at least d_min" a property of the
EXECUTED command regardless of what the policy learned.

Barrier and one-step-ahead condition (discrete time):

    h(x)        = ||p_r - p_p||^2 - d_min^2
    p_p'        = p_p + v_p * dt                 (constant-velocity prediction)
    a           = p_r - p_p'                     (relative position)
    h(x') >= (1 - alpha) * h(x)
    <=> ||a + v_r dt||^2 >= c ,  c = d_min^2 + (1-alpha) h(x)

Two ways to enforce it on the commanded velocity, both implemented here:

  mode="linear" (default): drop the O(dt^2) term to get the HALF-SPACE
      a . v_r >= (c - ||a||^2) / (2 dt)
    and apply the minimal-norm correction of each violated constraint,
    iterated over pedestrians.  This is what a CBF-QP with a first-order barrier
    does, and it changes the command as little as possible -- measured to be far
    gentler than the exact projection (which collapsed task success to 0.03-0.25
    when it was the only mode).

  mode="exact": project onto the EXTERIOR of the disk of radius sqrt(c)/dt
    centred at -a/dt (exact for the quadratic condition, but violent at close
    range, where the desired velocity sits deep inside the disk).

  mode="joint" (route A, 2026-09-22): solve ALL constraints of one robot
    TOGETHER instead of cycling through them.  The cyclic version applies each
    constraint's minimal-norm correction independently, so a later correction can
    undo an earlier one, and the pedestrian and peer terms compete for the same
    two velocity components (measured: the all-peer term cut R-R contacts 3.3x
    but also cost 3.9pp of social success).  Every constraint here is a
    half-space a_i . v >= b_i, so the exact minimum-norm feasible velocity in 2D
    is found by active-set enumeration: candidates are v_des itself, the
    projection of v_des onto each constraint's boundary, and the pairwise
    line intersections; the feasible candidate closest to v_des wins.  The
    acceleration limit is included as a K-tangent polygon (inscribed in the
    |v - v_prev| <= a_max*dt disk), so it is part of the same solve rather than
    a post-hoc projection that can silently undo a safety correction.

Honest caveats (must be reported with the numbers):
  * the pedestrian is assumed to hold its velocity for one step; a pedestrian
    acceleration bound can be folded in as `margin` (inflates d_min);
  * a feasible set defined by several non-convex constraints is approximated by
    a few cyclic passes, so the REALISED barrier is reported as a metric
    (cbf_hmin / cbf_viol), never assumed.
"""
from __future__ import annotations

import torch


class CBFFilter:
    def __init__(self, d_min=0.7, alpha=0.5, vmax=None, amax=0.0, dt=0.05,
                 passes=2, mode="linear", margin=0.05, enforce_accel=False,
                 emotion=False, mood_margin=0.0, q_gain=0.0,
                 peer=False, d_peer=0.65, peer_front_only=True,
                 n_tangents=12, joint_tol=1e-6, ttc_gain=0.0,
                 pred_gain=0.0, pred_horizon=1.0, pred_sub=5,
                 ttc_adapt=0.0, ttc_near=1.5, shape=None):
        self.d_min = float(d_min)
        self.alpha = float(alpha)
        self.vmax = vmax
        self.amax = float(amax)
        self.dt = float(dt)
        self.passes = int(passes)
        self.mode = str(mode)
        self.margin = float(margin)      # extra safety distance (m)
        self.enforce_accel = bool(enforce_accel)
        # ---- affect-aware barrier (opt-in, default OFF -> legacy geometry) ---
        # WHY: the geometric barrier h = ||p_r-p_p||^2 - d_min^2 is the same for
        # every pedestrian, so it cannot express the thing this project is about
        # -- that intruding on someone who is ALREADY distressed is worse than
        # the same geometry with a calm person.  Our affect model supplies both
        # ingredients: an anisotropic personal-space radius (front/side/back)
        # and the pedestrian's current mood.  With `emotion=True` the barrier
        # becomes
        #     d_eff = max(d_soc(bearing), d_min) + mood_margin * relu(-mood)
        # i.e. the OUTER of the fixed safety radius and the emotional personal
        # space, enlarged for pedestrians who are already upset.  The fixed part
        # is kept as a floor so the safety guarantee of the legacy filter is
        # never weakened -- the two rows are directly comparable.
        self.emotion = bool(emotion)
        self.mood_margin = float(mood_margin)
        # `q_gain` uses the env's THREE-LAYER emotional impact q_t (intrusion +
        # gated closing speed + accumulated distress, the same quantity the CVaR
        # constraint targets) as an extra radius:
        #     d_eff = max(d_soc(bearing), d_min) + mood_margin*relu(-mood) + q_gain*q
        # Unlike `relu(-mood)` -- which is zero for almost every pedestrian
        # because mood rarely goes negative (measured: adding mood_margin 0.3
        # changed nothing) -- q_t is nonzero exactly while the interaction is
        # harmful NOW and depends on the robot's own velocity, so the barrier
        # reacts to the harm being inflicted instead of only to harm already done.
        self.q_gain = float(q_gain)
        # ---- RECIPROCAL PEER-ROBOT barrier (opt-in, default OFF) ------------
        # WHY (measured): the filter only ever constrained robot-pedestrian
        # distance, so nothing in this project addressed robot-robot collisions:
        # R-R contact was 16.4-18.1% for every learned+CBF row and ~14-16% for
        # the scripted arms, against ORCA's 0.5% -- ORCA avoids peers
        # RECIPROCALLY, which a per-robot pedestrian-only barrier cannot do.
        # With `peer=True` each robot also enforces the same one-step barrier
        # against every OTHER robot's predicted position, using the peer contact
        # distance (env collision_dist_robot_robot) plus a margin.  Because all
        # robots run the same filter, the correction is mutual, which is what
        # makes this a multi-robot mechanism rather than N independent ones.
        self.peer = bool(peer)
        self.d_peer = float(d_peer)
        # Gate the peer constraint to robots that are AHEAD of the intended
        # direction of travel.  Measured motivation: with the constraint applied
        # to every peer, R-R contacts halved (18.0% -> 8.6%) but success fell
        # 0.570 -> 0.484, because a peer travelling alongside or behind still
        # couples the two velocity commands and slows the group.  A robot only
        # needs to yield to someone it is actually heading towards.
        self.peer_front_only = bool(peer_front_only)
        self.n_tangents = int(n_tangents)
        self.joint_tol = float(joint_tol)
        # ---- early-yield margin: radius grows with the CLOSING SPEED --------
        # WHY (measured 2026-09-22, and it explains the whole route-A result):
        # the exact minimum-correction joint QP scored success 0.818 / cvar 25.32,
        # i.e. barely different from NO filter, while the crude cyclic projection
        # scored 0.539 / 14.79.  The reason is not constraint satisfaction -- the
        # joint solve satisfies strictly more constraints (validated against a
        # brute-force grid search, agreement 0.006 m/s).  It is that the one-step
        # barrier condition, evaluated far from a pedestrian, is satisfied by a
        # tiny correction, whereas the cyclic filter's repeated corrections do not
        # cancel and therefore over-rotate: it YIELDS EARLY.  Early yielding is
        # exactly what the analytic detour does (cvar 20.2 vs beeline 29.2), so the
        # useful ingredient was implicit in the crude solver all along.
        # This knob makes it explicit and principled: inflate the exclusion radius
        # by the time-to-collision margin tau * closing_speed, so a robot moving
        # towards someone must start avoiding tau seconds earlier.
        self.ttc_gain = float(ttc_gain)
        # ---- AFFECT-PREDICTIVE margin (route A, P1) -------------------------
        # The fixed `ttc_gain` tau is a hand-set "yield this many seconds early".
        # This replaces it with the quantity the project actually cares about:
        # the pedestrian's PREDICTED mood loss if the robot keeps its current
        # relative velocity for `pred_horizon` seconds.  Using the env's own mood
        # dynamics (dmood/dt = -alpha*mood - beta*intrusion + ...) the predicted
        # loss is
        #     dLoss = beta * sum_k exp(-alpha s_k) * intrusion(s_k) * ds
        # with intrusion evaluated along the constant-velocity predicted path, so
        # the margin automatically accounts for closing speed, the anisotropic
        # personal space and how long the robot would stay inside it -- instead of
        # a single hand-tuned seconds value.  The barrier radius becomes
        #     d_eff = max(d_soc(bearing), d_min) + pred_gain * dLoss
        # (metres per unit of predicted mood loss).
        self.pred_gain = float(pred_gain)
        self.pred_horizon = float(pred_horizon)
        self.pred_sub = max(int(pred_sub), 1)
        # ---- density-adaptive early yield ----------------------------------
        # Measured: the same tau costs 18.6pp of task success at P=9 but 24.5pp at
        # P=16, and the best tau moves from ~1.0 s (P=9) to 0.25-0.5 s (P=16) --
        # a denser crowd makes the barrier bind far more often.  Rather than hand
        # tuning per density, scale tau by the LOCAL crowding around the robot:
        #     tau_eff = tau * (1 + adapt) / (1 + adapt * n_near)
        # where n_near counts pedestrians within `ttc_near` metres.  The anchor
        # keeps tau unchanged at n_near = 1 (the P=9 regime) and shrinks it in a
        # crowd, which is exactly the direction the measurements demand.
        self.ttc_adapt = float(ttc_adapt)
        self.ttc_near = float(ttc_near)
        # ---- EXPLICIT assumed personal-space shape (deconfounding control) ---
        # By default the barrier reads the personal-space shape from the
        # environment (`env.emo`), i.e. the SAME shape the affect COST is
        # computed with.  That makes the barrier-vs-nothing comparison fine, but
        # it makes any "is the emotion-derived shape the reason?" experiment
        # circular: changing `--env mood_d_front/...` moves the controller AND
        # the metric together.
        # `shape=(d_front, d_side, d_back)` lets the barrier ASSUME a shape that
        # differs from the one the cost is measured with, so the cost geometry
        # stays fixed across arms and the only thing that varies is the shape the
        # controller believes in.  None (default) = legacy behaviour = read env.
        self.shape = None if shape is None else tuple(float(x) for x in shape)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _ball(v, centre, radius):
        if torch.is_tensor(radius) and radius.dim() == v.dim() - 1:
            radius = radius.unsqueeze(-1)
        d = v - centre
        n = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return centre + d * torch.clamp(radius / n, max=1.0)

    @staticmethod
    def _exterior(v, centre, radius):
        if torch.is_tensor(radius) and radius.dim() == v.dim() - 1:
            radius = radius.unsqueeze(-1)
        d = v - centre
        n = d.norm(dim=-1, keepdim=True)
        out = centre + radius * d / n.clamp_min(1e-9)
        return torch.where(n >= radius, v, out)

    # ------------------------------------------------------- effective radius
    @torch.no_grad()
    def _d2(self, env):
        """squared effective exclusion radius per (B,N,P) (scalar if legacy)."""
        if not self.emotion or getattr(env, "P", 0) == 0:
            return self.d_min ** 2
        from .risk_field import anisotropic_social_radius
        emo = getattr(env, "emo", {}) or {}
        if self.shape is not None:
            df, ds, db = self.shape
        else:
            df = float(emo.get("d_front", 0.9))
            ds = float(emo.get("d_side", 0.6))
            db = float(emo.get("d_back", 0.5))
        rel = env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)     # (B,N,P,2)
        bear = torch.atan2(rel[..., 1], rel[..., 0]) \
            - env.ped_heading.unsqueeze(1)
        bear = torch.atan2(torch.sin(bear), torch.cos(bear))
        d_soc = anisotropic_social_radius(
            bear, d_front=df, d_side=ds, d_back=db)                   # (B,N,P)
        d_eff = torch.maximum(d_soc, torch.full_like(d_soc, self.d_min))
        if self.mood_margin != 0.0:
            distress = torch.clamp(-env.ped_mood, 0.0, 1.0).unsqueeze(1)
            d_eff = d_eff + self.mood_margin * distress
        if self.q_gain != 0.0:
            q = getattr(env, "ped_q", None)
            if q is not None and q.numel():
                d_eff = d_eff + self.q_gain * q.unsqueeze(1)      # (B,1,P)
        if self.ttc_gain != 0.0:
            relv = env.robo_vel.unsqueeze(2) - env.ped_vel.unsqueeze(1)
            dist_now = rel.norm(dim=-1)
            closing = (-(rel * relv).sum(-1) / (dist_now + 1e-6))
            tau = self.ttc_gain
            if self.ttc_adapt > 0.0:
                near = (dist_now < self.ttc_near).sum(dim=-1, keepdim=True).float()
                tau = tau * (1.0 + self.ttc_adapt) / (1.0 + self.ttc_adapt * near)
            d_eff = d_eff + tau * closing.clamp_min(0.0)
        if self.pred_gain != 0.0:
            relv = env.robo_vel.unsqueeze(2) - env.ped_vel.unsqueeze(1)
            beta = float(emo.get("beta", 1.8))
            alpha = float(emo.get("alpha", 0.06))
            import math
            ds = self.pred_horizon / self.pred_sub
            loss = torch.zeros_like(d_eff)
            for k in range(1, self.pred_sub + 1):
                sk = ds * k
                d_pred = (rel - relv * sk).norm(dim=-1)
                intr_k = torch.clamp(d_soc - d_pred, 0.0, None)
                loss = loss + math.exp(-alpha * sk) * intr_k * ds
            d_eff = d_eff + self.pred_gain * beta * loss
        return d_eff ** 2

    # ------------------------------------------------------- joint 2-D QP solve
    def _constraints(self, env, v_des):
        """stack every half-space constraint a.v >= b for each (env, robot).

        Returns (A, b) with shapes (B, N, M, 2) and (B, N, M).  Constraints whose
        right-hand side is -inf are inert and take part only in the feasibility
        check (they are always satisfied).
        """
        dt = self.dt
        device = v_des.device
        B, N = v_des.shape[0], v_des.shape[1]
        A_list, b_list = [], []

        # ---- pedestrians ---------------------------------------------------
        if getattr(env, "P", 0) > 0:
            p_r = env.robo_pos
            p_p = env.ped_pos + env.ped_vel * dt
            a = p_r.unsqueeze(2) - p_p.unsqueeze(1)                 # (B,N,P,2)
            d_now = (p_r.unsqueeze(2) - env.ped_pos.unsqueeze(1)).norm(dim=-1)
            d2 = self._d2(env)
            h = d_now ** 2 - d2
            c = d2 + (1.0 - self.alpha) * h
            aa = (a * a).sum(-1).clamp_min(1e-9)
            b = (c - aa) / (2.0 * dt) + self.margin * aa.sqrt() / dt
            A_list.append(a)
            b_list.append(b)
        else:
            A_list.append(torch.zeros(B, N, 0, 2, device=device))
            b_list.append(torch.zeros(B, N, 0, device=device))

        # ---- peers (reciprocal: half the responsibility each) --------------
        if self.peer and N > 1:
            p_r = env.robo_pos
            p_q = p_r + env.robo_vel * dt
            a = p_r.unsqueeze(2) - p_q.unsqueeze(1)                  # (B,N,N,2)
            d_now = (p_r.unsqueeze(2) - p_r.unsqueeze(1)).norm(dim=-1)
            eye = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
            h = d_now ** 2 - self.d_peer ** 2
            c = self.d_peer ** 2 + (1.0 - self.alpha) * h
            an = a.norm(dim=-1).clamp_min(1e-9)
            # relative form: (v - v_q).n >= (c - d^2)/(2 dt |a|); the robot takes
            # half, which is expressed as a lower bound on its own v.n
            v_q = env.robo_vel.unsqueeze(1).expand(-1, N, -1, -1)
            thr = (c - d_now ** 2) / (2.0 * dt * an) \
                + (v_q.mul(a / an.unsqueeze(-1)).sum(-1))
            b = thr * 0.5 + (v_q.mul(a / an.unsqueeze(-1)).sum(-1)) * 0.5
            # inactive where self-pair or (optionally) peer not ahead
            if self.peer_front_only:
                u = v_des / v_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                ahead = ((a / an.unsqueeze(-1)) * u.unsqueeze(2)).sum(-1) > 0.0
                b = torch.where(ahead & (~eye), b, torch.full_like(b, -1e6))
            else:
                b = torch.where(eye, torch.full_like(b, -1e6), b)
            b = b + self.margin * an / dt
            A_list.append(a / an.unsqueeze(-1))
            b_list.append(b)

        # ---- acceleration disk (K-tangent inscribed polygon) ---------------
        if self.enforce_accel and self.amax > 0.0:
            K = self.n_tangents
            th = torch.arange(K, device=device, dtype=v_des.dtype) * (2 * 3.141592653589793 / K)
            u = torch.stack([torch.cos(th), torch.sin(th)], -1)       # (K,2)
            vp = env.robo_vel                                         # (B,N,2)
            b = (vp.unsqueeze(2) * u.view(1, 1, K, 2)).sum(-1) - self.amax * dt
            A_list.append(u.view(1, 1, K, 2).expand(B, N, K, 2))
            b_list.append(b)

        # ---- speed cap (K-tangent inscribed polygon, centred at 0) --------
        if self.vmax is not None or getattr(env, "max_sp", None):
            vmax = self.vmax if self.vmax is not None else float(getattr(env, "max_sp", 1.3))
            K = self.n_tangents
            th = torch.arange(K, device=device, dtype=v_des.dtype) * (2 * 3.141592653589793 / K)
            u = torch.stack([torch.cos(th), torch.sin(th)], -1)
            A_list.append(u.view(1, 1, K, 2).expand(B, N, K, 2))
            b_list.append(torch.full((B, N, K), -vmax, device=device, dtype=v_des.dtype))

        return torch.cat(A_list, dim=2), torch.cat(b_list, dim=2)

    @torch.no_grad()
    def _constraints_grouped(self, env, v_des):
        """same constraints, but returned as named groups.

        Needed for the LEXICOGRAPHIC solve (route A, P3): pedestrian safety is the
        primary objective and the reciprocal peer term is secondary.  In the flat
        joint solve the two compete inside one minimum-norm problem, which was
        measured to cost +1.3pp of robot-pedestrian contact when the peer term was
        switched on.  With the groups separated we can (i) solve for
        pedestrians + hard limits, (ii) solve again including peers, and
        (iii) move from (i) towards (ii) only as far as the pedestrian
        constraints stay satisfied -- so the peer term can never degrade R-P
        safety below the pedestrian-only solution.
        """
        dt = self.dt
        device = v_des.device
        B, N = v_des.shape[0], v_des.shape[1]
        groups = {}

        if getattr(env, "P", 0) > 0:
            p_r = env.robo_pos
            p_p = env.ped_pos + env.ped_vel * dt
            a = p_r.unsqueeze(2) - p_p.unsqueeze(1)
            d_now = (p_r.unsqueeze(2) - env.ped_pos.unsqueeze(1)).norm(dim=-1)
            d2 = self._d2(env)
            h = d_now ** 2 - d2
            c = d2 + (1.0 - self.alpha) * h
            aa = (a * a).sum(-1).clamp_min(1e-9)
            b = (c - aa) / (2.0 * dt) + self.margin * aa.sqrt() / dt
            groups["ped"] = (a, b)
        else:
            groups["ped"] = (torch.zeros(B, N, 0, 2, device=device),
                             torch.zeros(B, N, 0, device=device))

        if self.peer and N > 1:
            p_r = env.robo_pos
            p_q = p_r + env.robo_vel * dt
            a = p_r.unsqueeze(2) - p_q.unsqueeze(1)
            d_now = (p_r.unsqueeze(2) - p_r.unsqueeze(1)).norm(dim=-1)
            eye = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
            h = d_now ** 2 - self.d_peer ** 2
            c = self.d_peer ** 2 + (1.0 - self.alpha) * h
            an = a.norm(dim=-1).clamp_min(1e-9)
            v_q = env.robo_vel.unsqueeze(1).expand(-1, N, -1, -1)
            thr = (c - d_now ** 2) / (2.0 * dt * an) \
                + (v_q.mul(a / an.unsqueeze(-1)).sum(-1))
            b = thr * 0.5 + (v_q.mul(a / an.unsqueeze(-1)).sum(-1)) * 0.5
            if self.peer_front_only:
                u = v_des / v_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                ahead = ((a / an.unsqueeze(-1)) * u.unsqueeze(2)).sum(-1) > 0.0
                b = torch.where(ahead & (~eye), b, torch.full_like(b, -1e6))
            else:
                b = torch.where(eye, torch.full_like(b, -1e6), b)
            b = b + self.margin * an / dt
            groups["peer"] = (a / an.unsqueeze(-1), b)
        else:
            groups["peer"] = (torch.zeros(B, N, 0, 2, device=device),
                              torch.zeros(B, N, 0, device=device))

        A_h, b_h = [], []
        if self.enforce_accel and self.amax > 0.0:
            K = self.n_tangents
            th = torch.arange(K, device=device, dtype=v_des.dtype) * (2 * 3.141592653589793 / K)
            u = torch.stack([torch.cos(th), torch.sin(th)], -1)
            vp = env.robo_vel
            A_h.append(u.view(1, 1, K, 2).expand(B, N, K, 2))
            b_h.append((vp.unsqueeze(2) * u.view(1, 1, K, 2)).sum(-1) - self.amax * dt)
        vmax = self.vmax if self.vmax is not None else float(getattr(env, "max_sp", 1.3))
        K = self.n_tangents
        th = torch.arange(K, device=device, dtype=v_des.dtype) * (2 * 3.141592653589793 / K)
        u = torch.stack([torch.cos(th), torch.sin(th)], -1)
        A_h.append(u.view(1, 1, K, 2).expand(B, N, K, 2))
        b_h.append(torch.full((B, N, K), -vmax, device=device, dtype=v_des.dtype))
        groups["hard"] = (torch.cat(A_h, dim=2), torch.cat(b_h, dim=2))
        return groups

    @staticmethod
    @torch.no_grad()
    def _solve_active_set(a, bb, v0, tol=1e-4):
        """exact minimum-norm point of {a.v >= b} closest to v0 (2-D active set).

        a: (P,M,2), bb: (P,M), v0: (P,2) -> (P,2).  Assumes v0 is feasible when a
        feasible point exists nearby; returns v0 when no candidate is feasible
        (the caller decides what to do in that case).
        """
        P, M, _ = a.shape
        slack0 = (a * v0.unsqueeze(1)).sum(-1) - bb
        feas0 = (slack0 >= -tol).all(dim=1)
        best = v0.clone()
        bestd = torch.where(feas0, torch.zeros(P, device=v0.device),
                            torch.full((P,), float("inf"), device=v0.device))
        an2 = (a * a).sum(-1).clamp_min(1e-12)
        need = ((bb - (a * v0.unsqueeze(1)).sum(-1)).clamp_min(0.0) / an2).unsqueeze(-1)
        cand1 = v0.unsqueeze(1) + need * a
        a1 = a.unsqueeze(2)
        a2 = a.unsqueeze(1)
        b1 = bb.unsqueeze(2)
        b2 = bb.unsqueeze(1)
        det = a1[..., 0] * a2[..., 1] - a1[..., 1] * a2[..., 0]
        ok = det.abs() > 1e-9
        num_x = b1 * a2[..., 1] - a1[..., 1] * b2
        num_y = a1[..., 0] * b2 - b1 * a2[..., 0]
        det_safe = torch.where(ok, det, torch.ones_like(det))
        cand2 = torch.stack([num_x / det_safe, num_y / det_safe], -1)
        cand2 = torch.where(ok.unsqueeze(-1), cand2, torch.full_like(cand2, float("nan")))
        for cand in (cand1, cand2.reshape(P, M * M, 2)):
            slack = (a.unsqueeze(1) * cand.unsqueeze(2)).sum(-1) - bb.unsqueeze(1)
            feas = (slack >= -tol).all(dim=-1)
            d = (cand - v0.unsqueeze(1)).norm(dim=-1)
            d = torch.where(feas, d, torch.full_like(d, float("inf")))
            m, arg = d.min(dim=1)
            take = m < bestd
            best = torch.where(take.unsqueeze(-1),
                               cand.gather(1, arg.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1),
                               best)
            bestd = torch.where(take, m, bestd)
        return best, bestd

    @torch.no_grad()
    def _project_lexicographic(self, env, v_des):
        """pedestrian-first (lexicographic) joint solve.

        Step 1: minimum-norm correction against pedestrians + hard limits.
        Step 2: the same against pedestrians + peers + hard limits.
        Step 3: move from step 1 towards step 2 by the largest factor alpha <= 1
                that keeps every pedestrian constraint satisfied.  The peer term
                therefore never makes R-P safety worse than the pedestrian-only
                solution, while still doing as much peer avoidance as that allows.
        """
        vmax = self.vmax if self.vmax is not None else float(getattr(env, "max_sp", 1.3))
        v_cap = self._ball(v_des, torch.zeros_like(v_des), vmax)
        g = self._constraints_grouped(env, v_des)
        B, N = v_des.shape[0], v_des.shape[1]
        P = B * N

        def cat(keys):
            A = torch.cat([g[k][0] for k in keys], dim=2).reshape(P, -1, 2)
            b = torch.cat([g[k][1] for k in keys], dim=2).reshape(P, -1)
            return A, b

        v0 = v_cap.reshape(P, 2)
        A_ped, b_ped = cat(["ped", "hard"])
        v_ped, ok_ped = self._solve_active_set(A_ped, b_ped, v0)
        # Infeasible pedestrian set (common with a large tau: the one-step barrier
        # would demand more speed than the cap allows).  Fall back to the cyclic
        # projection for exactly those instances, same as the flat joint solve --
        # otherwise the lexicographic path would silently return the UNFILTERED
        # command, which is strictly worse.
        bad_ped = ~torch.isfinite(ok_ped)
        if bool(bad_ped.any()):
            v_fb = self._project_cyclic(env, v_des).reshape(P, 2)
            v_ped = torch.where(bad_ped.unsqueeze(-1), v_fb, v_ped)
        if self.peer and env.N > 1:
            A_all, b_all = cat(["ped", "peer", "hard"])
            v_all, ok_all = self._solve_active_set(A_all, b_all, v0)
            bad_all = ~torch.isfinite(ok_all)
            if bool(bad_all.any()):
                v_all = torch.where(bad_all.unsqueeze(-1), v_ped, v_all)
            Ap, bp = g["ped"][0].reshape(P, -1, 2), g["ped"][1].reshape(P, -1)
            if Ap.shape[1] > 0:
                slack = (Ap * v_ped.unsqueeze(1)).sum(-1) - bp
                d = v_all - v_ped
                proj = (Ap * d.unsqueeze(1)).sum(-1)
                alpha = torch.where(proj < -1e-9,
                                    slack / (-proj).clamp_min(1e-9),
                                    torch.ones_like(proj))
                alpha = alpha.clamp(0.0, 1.0).amin(dim=1, keepdim=True)
                v = v_ped + alpha * d
            else:
                v = v_all
        else:
            v = v_ped
        return self._ball(v.reshape(B, N, 2), torch.zeros_like(v_des), vmax)

    @torch.no_grad()
    def _project_joint(self, env, v_des):
        """exact minimum-norm feasible velocity per robot (active-set in 2D).

        CRITICAL DETAIL (found by measurement): the reference point of the QP must
        be the velocity the robot would ACTUALLY execute (rate-limited), not the raw
        network output -- otherwise the speed cap dominates the objective and the
        barrier constraints are ignored.
        """
        vmax = self.vmax if self.vmax is not None else float(getattr(env, "max_sp", 1.3))
        v_cap = self._ball(v_des, torch.zeros_like(v_des), vmax)
        A, b = self._constraints(env, v_des)
        B, N, M, _ = A.shape
        a = A.reshape(B * N, M, 2)
        bb = b.reshape(B * N, M)
        v0 = v_cap.reshape(B * N, 2)
        best, bestd = self._solve_active_set(a, bb, v0, tol=self.joint_tol)
        bad = ~torch.isfinite(bestd)
        if bool(bad.any()):
            v_fb = self._project_cyclic(env, v_des)
            best = torch.where(bad.unsqueeze(-1), v_fb.reshape(B * N, 2), best)
        return self._ball(best.reshape(B, N, 2), torch.zeros_like(v_des), vmax)

    # ------------------------------------------------------------------- main
    @torch.no_grad()
    def project(self, env, v_des):
        """v_des: (B,N,2) desired velocity -> filtered (B,N,2)."""
        if self.mode == "lex":
            return self._project_lexicographic(env, v_des)
        if self.mode == "joint":
            return self._project_joint(env, v_des)
        return self._project_cyclic(env, v_des)

    @torch.no_grad()
    def _project_cyclic(self, env, v_des):
        """legacy path: sequential minimal-norm corrections (linear/exact)."""
        dt = self.dt
        v = v_des.clone()
        vmax = self.vmax if self.vmax is not None else float(getattr(env, "max_sp", 1.3))

        if getattr(env, "P", 0) > 0:
            p_r = env.robo_pos                                   # (B,N,2)
            p_p = env.ped_pos + env.ped_vel * dt                  # (B,P,2) prediction
            a = p_r.unsqueeze(2) - p_p.unsqueeze(1)               # (B,N,P,2)
            d_now = (p_r.unsqueeze(2) - env.ped_pos.unsqueeze(1)).norm(dim=-1)
            d2 = self._d2(env)                                    # (B,N,P) or scalar
            h = d_now ** 2 - d2                                   # (B,N,P)
            c = d2 + (1.0 - self.alpha) * h
            if self.mode == "linear":
                aa = (a * a).sum(-1).clamp_min(1e-9)              # (B,N,P)
                b = (c - aa) / (2.0 * dt)                         # (B,N,P)
                b = b + self.margin * aa.sqrt() / dt              # strict margin
                for _ in range(self.passes):
                    for i in range(a.shape[2]):
                        ai, bi = a[..., i, :], b[..., i]
                        av = (v * ai).sum(-1)                     # (B,N)
                        need = (bi - av).clamp_min(0.0)           # (B,N)
                        v = v + (need / aa[..., i]).unsqueeze(-1) * ai
            else:                                                 # exact disk
                centre = -a / dt
                radius = c.clamp_min(0.0).sqrt() / dt
                active = c > 0.0
                for _ in range(self.passes):
                    for i in range(centre.shape[2]):
                        vi = self._exterior(v, centre[..., i, :], radius[..., i])
                        v = torch.where(active[..., i].unsqueeze(-1), vi, v)
        if self.peer and getattr(env, "N", 1) > 1:
            # RECIPROCAL peer condition (ORCA-style half-split), NOT the absolute
            # half-space used for pedestrians.  Measured reason: the absolute
            # version with d_peer=0.75 drove success to 0.000 with timeout 1.000
            # -- three robots that start within the barrier each conclude they
            # must move away from the others, cancel their goal-directed motion
            # and deadlock (coll_rr did drop to 0, but nobody arrived).
            # Each robot therefore takes HALF of the responsibility for the
            # pair's closing speed, which is exactly what makes reciprocal
            # avoidance non-degenerate -- and is what ORCA does for peers.
            N = env.N
            p_r = env.robo_pos
            p_q = env.robo_pos + env.robo_vel * dt                  # prediction
            a = p_r.unsqueeze(2) - p_q.unsqueeze(1)                  # (B,N,N,2)
            d_now = (p_r.unsqueeze(2) - p_r.unsqueeze(1)).norm(dim=-1)
            eye = torch.eye(N, device=v.device, dtype=torch.bool).view(1, N, N)
            h = d_now ** 2 - self.d_peer ** 2
            c = self.d_peer ** 2 + (1.0 - self.alpha) * h
            an = a.norm(dim=-1).clamp_min(1e-6)                      # (B,N,N)
            n = a / an.unsqueeze(-1)
            v_q = env.robo_vel.unsqueeze(1).expand(-1, N, -1, -1)    # (B,N,N,2)
            v_self = v.unsqueeze(2)                                  # (B,N,1,2)
            need = (c - d_now ** 2) / (2.0 * dt * an) \
                - (v_self - v_q).mul(n).sum(-1)                      # (B,N,N)
            act_mask = (~eye).float()
            if self.peer_front_only:
                u = v_des / v_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                ahead = (n * u.unsqueeze(2)).sum(-1) > 0.0            # (B,N,N)
                act_mask = act_mask * ahead.float()
            need = need.clamp_min(0.0) * act_mask
            if self.mode == "linear":
                corr = 0.5 * (need.unsqueeze(-1) * n).sum(dim=2)
                for _ in range(self.passes):
                    v = v + corr
                    # recompute the residual closing for a second pass
                    need = (c - d_now ** 2) / (2.0 * dt * an) \
                        - (v.unsqueeze(2) - v_q).mul(n).sum(-1)
                    need = need.clamp_min(0.0) * act_mask
                    corr = 0.5 * (need.unsqueeze(-1) * n).sum(dim=2)
            else:
                centre = -a / dt
                radius = c.clamp_min(0.0).sqrt() / dt
                active = (c > 0.0) & (~eye)
                for _ in range(self.passes):
                    for i in range(centre.shape[2]):
                        vi = self._exterior(v, centre[..., i, :], radius[..., i])
                        v = torch.where(active[..., i].unsqueeze(-1), vi, v)
        v = self._ball(v, torch.zeros_like(v), vmax)
        if self.enforce_accel and self.amax > 0.0:
            v = self._ball(v, env.robo_vel, self.amax * dt)
        return v

    # -------------------------------------------------------------- metrics
    @staticmethod
    @torch.no_grad()
    def barrier(env, d_min):
        """per-env (B,) barrier value h = min_{n,p}(||p_r-p_p||^2 - d_min^2)"""
        if getattr(env, "P", 0) == 0:
            return torch.full((env.B,), float("inf"), device=env.device)
        d = (env.robo_pos.unsqueeze(2) - env.ped_pos.unsqueeze(1)).norm(dim=-1)
        return (d.amin(dim=(1, 2)) ** 2) - d_min ** 2
