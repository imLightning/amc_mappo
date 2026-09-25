"""
policy.py
Shared networks for PPO / MAPPO:
  - actor : Gaussian MLP over states -> diagonal-normal action
  - critic: MLP value over (obs) for IPPO, or (global state) for MAPPO.
Parameter sharing across agents (MAPPO standard).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def orthogonal_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        nn.init.zeros_(m.bias)
    return m


def layer_init(size_in, size_out, std=math.sqrt(2), bias=0.0):
    layer = nn.Linear(size_in, size_out)
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias)
    return layer


class RunningMeanStd:
    """normalized statistics cache from the marl benchmark."""

    def __init__(self, epsilon=1e-4, shape=(), device=None, var_floor=0.0,
                 clip=0.0):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float)
        self.var = torch.ones(shape, device=device, dtype=torch.float)
        self.count = epsilon
        self._shape = shape
        # ---- variance floor (opt-in; 0.0 == legacy behaviour) --------------
        # WHY (measured, NIGHT_REPORT section 49): observation dimensions that are
        # CONSTANT during training (typically capacity padding for entities that
        # are absent -- e.g. pedestrian slots beyond P=9) end up with var ~ 0, so
        # normalisation divides them by 1e-5.  When a denser scene fills those
        # slots, the same transform emits values of order 1e4-1e5, saturating the
        # network: measured at P=16 the mean |normalised obs| was 8794 (vs 0.33 at
        # P=9) and the commanded direction flipped to cos(action, goal) = -0.35,
        # i.e. the "cross-density failure" is largely a NORMALISATION artefact,
        # not evidence that the task is infeasible at P=16.
        self.var_floor = float(var_floor)
        self.clip = float(clip)

    def update(self, x):
        if x is None:
            return
        batch_mean = x.mean(dim=tuple(range(x.ndim - len(self._shape))) or ())
        batch_var = (x - batch_mean.expand_as(x)).pow(2).mean(
            dim=tuple(range(x.ndim - len(self._shape))) or ())
        batch_count = x.numel() / max(int(batch_mean.numel()), 1)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        self.mean = new_mean.detach()
        self.var = new_var.detach()
        self.count = new_count

    def normalize(self, x):
        var = self.var.clamp_min(self.var_floor) if self.var_floor > 0.0 \
            else self.var
        z = (x - self.mean) / (torch.sqrt(var) + 1e-5)
        # optional hard clip (0 == legacy).  Measured reason: at P=16 the mean
        # |normalised obs| reaches 8794 (vs 0.33 at P=9) because capacity-padding
        # dimensions have var ~ 0; clipping bounds the damage without touching the
        # in-distribution range (|z| <= 2.9 at P=9).
        if self.clip > 0.0:
            z = z.clamp(-self.clip, self.clip)
        return z


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128,
                 mean_gain=0.01, log_std_init=-0.5,
                 bounded_mean=False, mean_scale=1.3):
        """mean_gain / log_std_init are exposed because the default near-zero
        output head is a poor starting point for goal-reaching tasks: the
        untrained policy then emits |action| ~ 0.002 (measured), i.e. it stands
        still, and PPO must bootstrap a usable action magnitude from scratch
        through a very wide reward range.  A trained N=1 policy converges to
        |action| ~ 1.8, so a larger initial gain is a valid, cheaper start.
        """
        super().__init__()
        self.net = nn.Sequential(
            layer_init(obs_dim, hidden), nn.Tanh(),
            layer_init(hidden, hidden), nn.Tanh(),
            layer_init(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = layer_init(hidden, act_dim, std=float(mean_gain))
        # ---- action-parameterisation fix (opt-in) -------------------------
        # The environment converts a command into a velocity with
        # `vel = a * clamp(max_sp/|a|, 1)`, i.e. ONLY THE DIRECTION of `a` is a
        # command; its magnitude merely saturates.  Leaving the magnitude
        # unbounded therefore costs no expressiveness but wrecks exploration:
        # measured on the trained baselines log_std sat at its clamp (std 7.4)
        # while |mean| grew to ~52, so a sampled action differs from the mean
        # direction by only 6-7 degrees -- PPO is effectively a deterministic
        # policy-gradient method and cannot discover a qualitatively different
        # manoeuvre (e.g. yielding).  Bounding |mean| to `mean_scale` (the
        # env's max speed) keeps the command identical while making the same
        # std worth ~25 degrees, and it bounds the entropy term.
        self.bounded_mean = bool(bounded_mean)
        self.mean_scale = float(mean_scale)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5
                                    + float(log_std_init))

    def dist(self, obs):
        h = self.net(obs)
        mean = self.mean_head(h)
        if self.bounded_mean:
            n = mean.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            mean = mean * (self.mean_scale
                           * torch.tanh(n / self.mean_scale) / n)
        std = torch.exp(self.log_std.clamp(-5, 2))
        return Normal(mean, std)

    def sample(self, obs, deterministic=False):
        dist = self.dist(obs)
        if deterministic:
            return dist.mean, dist.mean
        act = dist.sample()
        logp = dist.log_prob(act).sum(-1)
        return act, logp

    def evaluate(self, obs, act):
        dist = self.dist(obs)
        logp = dist.log_prob(act).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy


class CriticMLP(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(dim, hidden), nn.Tanh(),
            layer_init(hidden, hidden), nn.Tanh(),
            layer_init(hidden, 1, std=1.0),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Policy(nn.Module):
    """container = actor + (optional) critic against central/individual state."""

    def __init__(self, obs_dim, state_dim, act_dim, hidden=128,
                 critic_uses_state=True):
        super().__init__()
        self.actor = GaussianActor(obs_dim, act_dim, hidden)
        self.critic = CriticMLP(state_dim if critic_uses_state else obs_dim,
                                hidden)
        self.critic_uses_state = critic_uses_state

    def act(self, obs, deterministic=False):
        return self.actor.sample(obs, deterministic)

    def critic_value(self, state_or_obs):
        return self.critic(state_or_obs)
