"""
scenario_generator.py -- training / held-out test domain construction
(plan_add §五.3, §九).

Produces configs that keep the *policy representation* fixed where needed, plus
explicit test-side factors (higher density, faster humans, more robots,
dynamics & sensing perturbations).  Used by eval_protocol / full-metric runs.
"""
from __future__ import annotations

import copy


TrainProfile = dict(
    name="train", n_agents=2, n_pedestrians=5, pedestrian_max_speed=1.0,
    max_robot_speed=1.2, obs_noise=0.0, control_delay=0,
)


TestProfiles = [
    dict(name="in_dist", n_agents=2, n_pedestrians=5, pedestrian_max_speed=1.0,
         max_robot_speed=1.2, obs_noise=0.0, control_delay=0),
    dict(name="dense_10", n_agents=2, n_pedestrians=10, pedestrian_max_speed=1.2,
         max_robot_speed=1.2, obs_noise=0.0, control_delay=0),
    dict(name="dense_20", n_agents=2, n_pedestrians=20, pedestrian_max_speed=1.5,
         max_robot_speed=1.2, obs_noise=0.0, control_delay=0),
    dict(name="fast_ped", n_agents=2, n_pedestrians=10, pedestrian_max_speed=1.8,
         max_robot_speed=1.2, obs_noise=0.0, control_delay=0),
    dict(name="more_robots", n_agents=4, n_pedestrians=10,
         pedestrian_max_speed=1.2, max_robot_speed=1.2, obs_noise=0.0,
         control_delay=0),
    dict(name="noisy_delay", n_agents=2, n_pedestrians=10,
         pedestrian_max_speed=1.2, max_robot_speed=1.2, obs_noise=0.1,
         control_delay=2),
]


def profile(name, base=None):
    base = base or {}
    for p in TestProfiles:
        if p["name"] == name:
            return dict(p)
    raise KeyError(name)


def apply_profile(cfg, prof):
    """mutate a config[env] using a profile (keeps obs_dim-compatible swaps
    when possible; NOTE robot/ped count changes require obs-capacity policy)."""
    env = cfg.env
    for k in ("n_agents", "n_pedestrians", "pedestrian_max_speed",
              "max_robot_speed", "obs_noise", "control_delay"):
        if k in prof:
            env[k] = prof[k]
    return cfg


def train_config(base):
    return apply_profile(copy.deepcopy(base), TrainProfile)
