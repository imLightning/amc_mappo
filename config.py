"""
config.py -- tiny nested attribute-config (YAML overlays optional).
"""
import copy
import os


class Config(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _from_dict(d):
    c = Config()
    for k, v in d.items():
        c[k] = _from_dict(v) if isinstance(v, dict) else v
    return c


def checkpoints_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "checkpoints")


DEFAULTS = {
    "env": dict(
        size=4.0,
        n_agents=2,
        n_pedestrians=0,
        n_obstacles=0,
        robot_radius=0.25,
        ped_radius=0.22,
        obstacle_radius=0.3,
        max_robot_speed=1.2,
        pedestrian_max_speed=0.8,
        goal_tolerance=0.35,
        collision_dist_robot_robot=0.5,
        collision_dist_robot_ped=0.47,
        action_dt=0.05,
        max_episode_seconds=25.0,
        # ---- realism knobs (see plan_realism.md) -------------------------
        # Every default below reproduces the LEGACY behaviour, so existing
        # recipes keep their exact numbers; new recipes opt in explicitly.
        # per-pedestrian speed N(mean, std) clipped; std=0 -> one constant
        # speed (= pedestrian_max_speed) for the whole crowd, as before.
        ped_speed_mean=None,
        ped_speed_std=0.0,
        ped_speed_min=None,
        ped_speed_max=None,
        # mixed willingness to yield to robots (levels + probabilities),
        # sampled per pedestrian; empty -> everyone uses ped_avoid_robot.
        ped_yield_levels=[],
        ped_yield_probs=[],
        # stationary workers vs circulating commuters
        ped_worker_frac=0.0,
        ped_worker_radius=0.35,
        # keep pedestrian spawns clear of the robots' goals (bug found by
        # inspecting trajectories); False = legacy, so old runs reproduce
        ped_clear_goals=False,
        # physical plausibility (see scripts/check_physics.py)
        hard_separation=False,     # geometrically resolve overlaps
        ped_steer_mode='add',      # 'blend' = bounded direction blending
        ped_worker_activity=0.0,   # m; local motion of station workers
        ped_worker_period=12.0,
        # perception into the ACTOR observation (no oracle intent: linear
        # prediction from observed positions/velocities only)
        anticipate_in_obs=False,
        contention_in_obs=False,
        perception_k=16,
        # velocity-level non-penetration at contacts
        contact_stop=False,
        # failure-augmented constrained cost (plan_emo.md 模块2):
        # added once to the CONSTRAINT cost at the step a robot fails, so
        # that 'stand still and harm nobody' is the most expensive option
        # rather than the cheapest.  0 = legacy (D only).
        cost_fail_penalty=0.0,
        # matching reward-side penalty
        # (reward.timeout_penalty, default 0 = legacy)
        ped_circulate=False,
        # robot acceleration limit (m/s^2); 0 -> immediate, i.e. legacy
        max_robot_accel=0.0,
        # ---- ERM: robot affect (plan_realism.md §7) ----------------------
        # a_i in [-1,1], +1 calm / -1 stressed; driven by blocking, space
        # pressure, courtesy received and time urgency.  Disabled = legacy.
        robot_emotion_enabled=False,
        robot_mood_alpha=1.0,       # recovery rate (tau ~ 1 s)
        robot_mood_beta=1.0,        # blocking sensitivity
        robot_mood_gamma=0.3,       # space-pressure sensitivity
        robot_mood_delta=0.5,       # courtesy sensitivity
        robot_mood_eta=0.3,         # time-urgency sensitivity
        robot_mood_in_obs=True,
        robot_pressure_radius=0.8,  # the robot's own personal space (m)
        # WHICH signal `emotion_reward_weight` penalises (Mood-Shaping arm):
        #   "mean" (legacy) -- mean pedestrian distress, i.e. the /P-diluted
        #       quantity.  Measured to be ~0.1/step against a ~2.0/step progress
        #       reward, so the 0.5 weight used in week2/3 was a no-op.
        #   "tail"         -- the SAME per-robot instantaneous cost the CVaR
        #       constraint uses (`self.cost_mode`, i.e. mood_tail = harm to the
        #       worst-treated pedestrian, no /P).  This makes the
        #       reward-shaping vs constraint contrast information-matched: the
        #       only difference is the mechanism (return shaping vs dual
        #       ascent), not the signal.
        # Default "mean" keeps every legacy recipe bit-identical (all of them
        # use emotion_reward_weight=0.0 anyway, where nothing is added).
        emotion_shaping_mode="mean",
        # collision_terminal: a robot-pedestrian / robot-robot / obstacle
        # contact ends the episode as a failure.  False = legacy (contacts were
        # free, so ignoring pedestrians entirely had the highest success rate).
        collision_terminal=False,
        # ---- analytic yielding prior (default 0 = OFF, legacy) ------------
        # Adds the scripted detour controller's pedestrian-repulsion term to
        # the robot's commanded velocity (before the acceleration limit), so
        # the learned policy becomes a residual on top of a behaviour that
        # reaches the task/affect frontier on its own.
        detour_prior_gain=0.0,
        detour_prior_radius=1.5,
        # ABLATION: keep the mood slot in the observation layout but always
        # feed 0.0, so the actor stays warm-startable from a mood-aware
        # baseline (mood_in_obs=False would change the input dimension).
        # False = legacy.
        mood_obs_zero=False,
        ped_vel_zero=False,      # ABLATION: pedestrian relative-velocity channel -> 0
        rmood_obs_zero=False,    # ABLATION: robot own-affect (ERM) channel -> 0
    ),
    "reward": dict(
        progress_omega=0.6,
        reach_bonus=3.0,
        step_penalty=0.01,
        collision_robot_robot=1.0,
        collision_robot_ped=1.2,
        collide_with_obstacle=1.0,
        # social terms enabled only when weight>0 (week2/3)
        discomfort_weight=0.0,
        discom_sigma=0.45,
        ttc_tau=0.8,
        ttc_weight=0.0,
        oscillation_weight=0.0,
    ),
    "vec": dict(n_parallel_envs=64, device="cuda", seed=0),
    "algo": dict(kind="mappo", gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 lr=3.0e-4, grad_N=10.0, value_loss_coef=0.5,
                 entropy_coef=0.001, train_epochs=6, num_minibatches=4,
                 hidden=128, obs_norm=True),
    "train": dict(total_steps=1_000_000, rollout_len=128, num_seeds=1,
                  seed=0, log_every=10, exp_name="default",
                  store_every=200_000, eval_every=50_000,
                  n_eval_episodes=24, eval_rollouts=32),
}


def default():
    return _from_dict(copy.deepcopy(DEFAULTS))


def load_cfg(yaml_file=None, overrides=None):
    """Load a recipe, merging onto defaults.

    Raising on a missing path is deliberate: the previous silent fallback
    (`if yaml_file and os.path.exists(yaml_file)`) meant a typo'd or not-yet
    -created recipe silently trained on DEFAULTS.  That actually happened --
    `configs/mr_cap_v6_n2p2.yaml` did not exist, so the "N=2, P=2" long
    runs trained with n_pedestrians=0 (no crowd at all) and every method
    trivially scored success 1.000 / 0 collisions.  A loud failure is far
    cheaper than an invalid experiment.
    """
    cfg = default()
    if yaml_file is not None:
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(
                f"recipe not found: {yaml_file!r}. Refusing to fall back to "
                f"defaults, which would silently change the scenario.")
        import yaml
        with open(yaml_file) as f:
            raw = yaml.safe_load(f) or {}
        _merge(cfg, raw)
    if overrides:
        _merge(cfg, overrides)
    return cfg


def _merge(node, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(node.get(k), Config):
            _merge(node[k], v)
        else:
            node[k] = v
