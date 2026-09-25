"""
curriculum.py -- density & dynamics curriculum (plan_add §三.5/§十一).

A pure scheduler mapping (update-step or risk level) -> target parameters,
so training code can bump pedestrian count/speed and dynamics difficulty as
performance improves.  No training here.
"""
from __future__ import annotations


class Curriculum:
    """piecewise-linear difficulty curriculum.

    stages: list of (fraction_of_total, {param: value}) milestones.
    Parameter examples: n_pedestrians, pedestrian_max_speed, max_robot_speed,
    max_accel, control_delay, obs_noise.
    """

    def __init__(self, stages=None):
        # default: easy -> hard density & speed
        self.stages = stages or [
            (0.0,  dict(n_pedestrians=2, pedestrian_max_speed=0.5)),
            (0.35, dict(n_pedestrians=5, pedestrian_max_speed=0.8)),
            (0.7,  dict(n_pedestrians=10, pedestrian_max_speed=1.2)),
            (1.0,  dict(n_pedestrians=15, pedestrian_max_speed=1.8)),
        ]

    def at(self, progress):
        """progress in [0,1]; interpolate between adjacent milestones."""
        progress = max(0.0, min(1.0, progress))
        out = {}
        for i, (f, params) in enumerate(self.stages):
            if progress >= f:
                out.update(params)
                if i + 1 < len(self.stages) and progress < self.stages[i+1][0]:
                    f2, p2 = self.stages[i+1]
                    span = max(f2 - f, 1e-9)
                    w = (progress - f) / span
                    for k, v2 in p2.items():
                        v1 = params.get(k, v2)
                        if isinstance(v1, int) and isinstance(v2, int):
                            out[k] = int(round(v1 + (v2 - v1) * w))
                        else:
                            out[k] = v1 + (v2 - v1) * w
                elif i + 1 < len(self.stages):
                    pass
        if not out and self.stages:
            out.update(self.stages[0][1])
        return dict(out)


class RiskCurriculum(Curriculum):
    """curriculum keyed by current safety-cost (instead of fixed schedule).
    Bumps difficulty when mean episode cost < threshold."""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or [0.3, 0.15, 0.08]
        self.level = 0
        super().__init__()

    def step_by_cost(self, mean_cost):
        if self.level < len(self.thresholds) and mean_cost < \
                self.thresholds[self.level]:
            self.level += 1
            return True
        return False
