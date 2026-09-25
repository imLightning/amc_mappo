"""
metrics.py -- structured aggregation for evaluation, incl. the full social/efficiency
indicator set from plan.md §五. Serialized as plain JSON-able numbers.
"""

import numpy as np


class Outcome:
    """incremental episode/trace aggregator.

    As well as the core task/safety counters it may receive per-episode optional
    arrays via ``add_episode(..., trace=dict)`` and the analyser computes the
    efficiency & social indicators.
    """

    def __init__(self, n_agents=1):
        self.n = n_agents
        self.reset()

    def reset(self):
        self.n_eps = 0
        self.n_succ = 0
        self.n_timeout = 0
        self.used_steps = 0
        self.coll_rr_eps = 0
        self.coll_rp_eps = 0
        self.len_list = []
        self.min_rr_list = []
        self.min_rp_list = []
        # richer per-episode scalar accumulations
        self._path = []          # path length (each agent? use joint avg)
        self._dur = []           # navigation time (steps*dt)
        self._speed = []         # mean speed over trace (agent mean)
        self._discomfort = []    # mean robot-human discomfort while active
        self._near_rate = []     # near-miss (TTC-low) per-step fraction
        self._osc = []           # accel / velocity-jerk cost
        self._personal = []      # fraction steps under personal zone
        self._ttc_low = []       # fraction robot-steps with small TTC
        self._ttc_low1 = []      # TTC < 1.0s step fraction
        self._ttc_low2 = []      # TTC < 2.0s step fraction
        self._spl = []
        self._fail_reason = []
        self._deadlock = []      # 1 if episode flagged deadlock, else 0
        self._ped_disturb = []   # pedestrian disturbance scalar
        # --- affective metrics (E-SocialNav) -----------------------------
        self._mpi = []           # mean pedestrian mood over the episode
        self._wmp = []           # fraction of peds whose worst mood < -0.5
        self._ari = []           # affective recovery: time to return near neutral
        self._aid = []           # P5: mean cumulative affective dose (threshold-free)
        self._wid = []           # P5: worst-decile cumulative dose

    # ------------------------------------------------------------------ feeders
    def add_episode(self, success, timedout, length, min_rr, min_rp,
                    coll_rr, coll_rp, trace=None):
        self.n_eps += 1
        self.used_steps += length
        self.len_list.append(length)
        if success:
            self.n_succ += 1
        elif timedout:
            self.n_timeout += 1
        if coll_rr and coll_rr > 0:
            self.coll_rr_eps += 1
        if coll_rp and coll_rp > 0:
            self.coll_rp_eps += 1
        if min_rr is not None:
            self.min_rr_list.append(min_rr)
        if min_rp is not None:
            self.min_rp_list.append(min_rp)
        if trace:
            self._path.append(trace.get("path_len"))
            self._dur.append(trace.get("nav_time"))
            self._speed.append(trace.get("avg_speed"))
            self._discomfort.append(trace.get("discomfort"))
            self._near_rate.append(trace.get("near_rate"))
            self._osc.append(trace.get("oscillation"))
            self._personal.append(trace.get("personal_viol"))
            self._ttc_low.append(trace.get("ttc_low"))
            self._ttc_low1.append(trace.get("ttc_low1"))
            self._ttc_low2.append(trace.get("ttc_low2"))
            self._spl.append(trace.get("spl"))
            self._fail_reason.append(trace.get("fail_reason", ""))
            self._deadlock.append(1 if trace.get("deadlock") else 0)
            self._ped_disturb.append(trace.get("ped_disturb"))
            self._mpi.append(trace.get("mpi"))
            self._wmp.append(trace.get("wmp"))
            self._ari.append(trace.get("ari"))
            self._aid.append(trace.get("aid"))
            self._wid.append(trace.get("wid"))

    # ------------------------------------------------------------------ summary
    def summary(self, dt=0.05):
        n = max(self.n_eps, 1)

        def _r(lst, d=0.0):
            lst = [x for x in lst if x is not None and np.isfinite(x)]
            return float(np.mean(lst)) if lst else d

        def _rr(lst):
            return _r([1 if x else 0 for x in lst])

        sc = {
            "episodes": self.n_eps,
            "success_rate": float(self.n_succ) / n,
            "timeout_rate": float(self.n_timeout) / n,
            "avg_steps": float(np.mean(self.len_list)) if self.len_list else 0.0,
            "nav_time_s": _r(self._dur, self.avg_steps_s(dt)),
            "avg_path_len": _r(self._path),
            "avg_speed": _r(self._speed),
            "spl": _r(self._spl),
            "path_efficiency": _r(self._spl, 0.0),      # same scale placeholder
            "avg_discomfort": _r(self._discomfort),
            "near_rate": _r(self._near_rate),     # robot-human near-miss step frac
            "ttc_low_rate": _r(self._ttc_low),
            "ttc_low1_rate": _r(self._ttc_low1),
            "ttc_low2_rate": _r(self._ttc_low2),
            "personal_space_viol": _r(self._personal),
            "oscillation": _r(self._osc),
            "deadlock_rate": _r(self._deadlock),
            "pedestrian_disturbance": _r(self._ped_disturb),
            "collrr_rate": float(self.coll_rr_eps) / n,
            "collrp_rate": float(self.coll_rp_eps) / n,
            "min_rr_avg": _r(self.min_rr_list),
            "min_rp_avg": _r(self.min_rp_list),
            # affective metrics: MPI higher is better, WMP/ARI lower is better
            "mpi": _r(self._mpi),
            "wmp": _r(self._wmp),
            "ari": _r(self._ari),
            "aid": _r(self._aid),    # cumulative affective dose (lower better)
            "wid": _r(self._wid),    # worst-decile dose (lower better)
        }
        # composite social + efficiency cost (plan §五)
        sc["social_cost"] = (1.0 * sc.get("personal_space_viol", 0)
                             + 2.0 * sc.get("near_rate", 0)
                             + 1.0 * sc.get("ttc_low_rate", 0)
                             + 0.5 * sc.get("oscillation", 0))
        sc["fail_rate"] = 1.0 - sc["success_rate"] - sc["timeout_rate"]
        return sc

    def avg_steps_s(self, dt):
        return (float(np.mean(self._dur)) if self._dur
                else float(np.mean(self.len_list)) * dt)


def summarize_seeds(list_of_summaries, n_boot=1000, rng=None):
    """aggregate list of per-seed summary dicts -> mean/std/95%CI + bootstrap.

    For each numeric key compute mean, population std, 95% bootstrap CI, and
    (if 2 groups passed) a Welch t-test p-value.  Returns JSON-friendly dict.
    """
    rng = rng or np.random.default_rng(0)
    keys = set()
    for s in list_of_summaries:
        keys.update(s.keys())
    out = {}
    for k in keys:
        vals = [s.get(k) for s in list_of_summaries if s.get(k) is not None]
        vals = [float(v) for v in vals if np.isfinite(v)]
        if not vals:
            continue
        arr = np.asarray(vals)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        # bootstrap 95% CI on the mean
        if len(arr) > 1:
            boots = [arr[rng.integers(0, len(arr), len(arr))].mean()
                     for _ in range(n_boot)]
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        else:
            lo = hi = mean
        out[k] = {"mean": mean, "std": std,
                  "ci95_low": lo, "ci95_high": hi, "n": len(arr)}
    return out


def welch_t(group_a, group_b):
    """Welch two-sample t-test (equal-variance not assumed) p-value."""
    a = np.asarray([float(x) for x in group_a if np.isfinite(float(x))])
    b = np.asarray([float(x) for x in group_b if np.isfinite(float(x))])
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    t = (ma - mb) / np.sqrt(va/len(a) + vb/len(b) + 1e-12)
    from scipy import stats
    denom = (va/len(a) + vb/len(b)) ** 2
    df = denom / ((va/len(a))**2/(len(a)-1) + (vb/len(b))**2/(len(b)-1))
    return float(2 * stats.t.sf(abs(t), df))
