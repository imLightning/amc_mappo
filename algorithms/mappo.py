"""
mappo.py -- week-2 trainers.

Two on-policy baselines implemented over a vectorised env:
  * MAPPO  (kind=="mappo")  param-shared Gaussian actor + a centralised critic
                            over the global state (Yu et al., 2022).
  * IPPO   (kind=="ippo")   each agent learns its OWN actor + local critic and
                            sees only its own observation (independent PPO).

Shared code path: rollout L steps -> piecewise GAE -> epoch/minibatch PPO clip.
Agents N>1 use identical obs dimension (only content differs) so a single
normalizer (feature statistics) may be shared even for per-agent policies.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .buffer import RolloutBuffer
from .policy import Policy, RunningMeanStd


def make_net(obs_dim, state_dim, act_dim, hidden, central):
    return Policy(obs_dim, state_dim, act_dim, hidden=hidden,
                  critic_uses_state=central)


class Trainer:
    """kind: 'mappo' | 'ippo'.  N agents share feature normaliser only."""

    def __init__(self, env, cfg, device=None):
        cfg = dict(cfg)
        self.env = env
        self.device = torch.device(device) if device else env.device
        self.N = env.N
        self.kind = cfg.get("kind", "mappo")
        self.central = (self.kind == "mappo") and \
            hasattr(env, "_state_dim") and env._state_dim is not None

        self.obs_dim = env._obs_dim
        self.act_dim = 2
        self.state_dim = env._state_dim if self.central else env._obs_dim
        hidden = int(cfg.get("hidden", 128))

        # one shared policy for MAPPO; one policy per agent for IPPO
        self.policies = nn.ModuleList([
            make_net(self.obs_dim, self.state_dim, self.act_dim, hidden,
                     self.central).to(self.device)
            for _ in range(1 if self.central else self.N)
        ])

        self.opt = optim.Adam(self.policies.parameters(),
                              lr=float(cfg.get("lr", 3e-4)), eps=1e-5)
        self.gamma = float(cfg.get("gamma", 0.99))
        self.lam = float(cfg.get("gae_lambda", 0.95))
        self.clip = float(cfg.get("clip_eps", 0.2))
        self.ent_coef = float(cfg.get("entropy_coef", 0.001))
        self.val_coef = float(cfg.get("value_loss_coef", 0.5))
        self.grad_max = float(cfg.get("grad_N", 10.0))
        self.epochs = int(cfg.get("train_epochs", 5))
        self.minib = int(cfg.get("num_minibatches", 4))

        self.rms_obs = RunningMeanStd(shape=(self.obs_dim,),
                                      device=self.device)
        # sequential env steps already consumed (one per rollout inner step,
        # irrespective of parallel width).  We also track true collected
        # transition count = sequential*B*N.
        self.rollout_steps = 0

    # ------------------------------------------------------------------
    def _policy_of(self, a):
        """policy index for agent a (0 for shared MAPPO)."""
        return 0 if self.central else int(a)

    def _value_of_block(self, net, obs_n, state, a):
        """agent 'a' block value across (B,); returns (B,)."""
        if net.critic_uses_state:
            return net.critic(state)              # (B,)
        oa = obs_n[:, a] if obs_n.ndim == 3 else obs_n
        return net.critic(oa)                     # (B,)

    @torch.no_grad()
    def collect(self, rollout_len):
        env = self.env
        buf = RolloutBuffer()
        obs, state = env.reset()
        obs = torch.as_tensor(obs, device=self.device).float()
        state = torch.as_tensor(state, device=self.device).float()
        self.rms_obs.update(obs.reshape(-1, self.obs_dim))

        stats = {"success": 0.0, "episodes": 0.0, "rew_mean": 0.0}
        B = int(obs.shape[0])
        for _ in range(rollout_len):
            obs_n = self.rms_obs.normalize(obs)
            val = torch.empty(B, self.N, device=self.device)
            act = torch.empty(B, self.N, self.act_dim, device=self.device)
            logp = torch.empty(B, self.N, device=self.device)
            for a in range(self.N):
                net = self.policies[self._policy_of(a)]
                oa = obs_n[:, a] if self.N > 1 else obs_n[:, 0]
                mu_std = net.actor
                fa = oa.reshape(-1, self.obs_dim)
                aa, lpa = mu_std.sample(fa)
                act[:, a] = aa.view(B, self.act_dim)
                logp[:, a] = lpa.view(B)
                val[:, a] = self._value_of_block(net, obs_n, state, a)

            nobs, nstate, rew, done, _trunc, info = env.step(act.detach())
            rew = torch.as_tensor(rew, device=self.device).float()
            done_b = torch.as_tensor(done, device=self.device,
                                     dtype=torch.bool)
            buf.push(obs, state, act, logp, rew, done_b, val)

            self.rollout_steps += self.N * B          # agent-transitions
            stats["rew_mean"] += float(rew.mean())
            stats["success"] += int(done_b.sum())
            stats["episodes"] += int(done_b.sum())

            obs = torch.as_tensor(nobs, device=self.device).float()
            state = torch.as_tensor(nstate, device=self.device).float()

        # final bootstrap values
        obs_n = self.rms_obs.normalize(obs)
        final_val = torch.empty(B, self.N, device=self.device)
        for a in range(self.N):
            net = self.policies[self._policy_of(a)]
            final_val[:, a] = self._value_of_block(net, obs_n, state, a)
        advs = buf.compute_returns_gae(self.gamma, self.lam, final_val)
        rets = [ad + v for ad, v in zip(advs, buf.val)]
        data = buf.stack()
        data["adv"] = torch.stack(advs, 0)
        data["ret"] = torch.stack(rets, 0)
        if rollout_len:
            stats["rew_mean"] /= rollout_len
        return data, stats

    # ------------------------------------------------------------------
    def update(self, data):
        obs = data["obs"]        # (L,B,N,obs)
        act = data["act"]
        logp0 = data["logp_old"]
        adv = data["adv"]
        ret = data["ret"]
        L_ = obs.shape[0]
        B_ = obs.shape[1]
        N_ = obs.shape[2]

        self.rms_obs.update(obs.reshape(-1, self.obs_dim))
        obs_n_all = self.rms_obs.normalize(obs)

        tot = 0.0
        steps = 0
        # number of policy instances optimized this call:
        npol = 1 if self.central else self.N
        for p in range(npol):
            if self.central:
                # p unused; all agents use policy 0, data flattened across N
                o = obs_n_all.reshape(-1, self.obs_dim)
                a = act.reshape(-1, self.act_dim)
                o0 = logp0.reshape(-1)
                adv_ = adv.reshape(-1)
                ret_ = ret.reshape(-1)
                S = data["state"].shape[-1]
                st = data["state"].unsqueeze(2).expand(L_, B_, N_, S)\
                    .reshape(-1, S)
            else:
                o = obs_n_all[:, :, p, :].reshape(-1, self.obs_dim)
                a = act[:, :, p, :].reshape(-1, self.act_dim)
                o0 = logp0[:, :, p].reshape(-1)
                adv_ = adv[:, :, p].reshape(-1)
                ret_ = ret[:, :, p].reshape(-1)
                st = o              # local
            net = self.policies[p]
            M = o.shape[0]
            advz = (adv_ - adv_.mean()) / (adv_.std().clamp_min(1e-6))
            perm = torch.randperm(M)
            bsize = max((M + self.minib - 1)//self.minib, 1)
            for _ in range(self.epochs):
                for s in range(0, M, bsize):
                    ids = perm[s:s+bsize]
                    nob = o[ids]; na = a[ids]
                    lp, ent = net.actor.evaluate(nob, na)
                    ratio = torch.exp(lp - o0[ids])
                    s1 = ratio * advz[ids]
                    s2 = torch.clamp(ratio, 1-self.clip,
                                     1+self.clip) * advz[ids]
                    pol = -torch.min(s1, s2).mean()
                    v = net.critic(st[ids] if self.central else nob)
                    vL = 0.5 * ((v - ret_[ids]) ** 2).mean()
                    eL = -ent.mean()
                    Ll = pol + self.val_coef * vL + self.ent_coef * eL
                    self.opt.zero_grad()
                    Ll.backward()
                    nn.utils.clip_grad_norm_(self.policies.parameters(),
                                             self.grad_max)
                    self.opt.step()
                    tot += float(Ll.detach())
                    steps += 1
        return {"loss": tot/max(steps, 1)}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act_det(self, obs):
        """deterministic actions for eval (all agents, vectorised)."""
        obs = torch.as_tensor(obs, device=self.device).float()
        B, N = obs.shape[0], self.N
        flat_n = self.rms_obs.normalize(obs)
        out = torch.empty(B, N, self.act_dim, device=self.device)
        for a in range(N):
            net = self.policies[self._policy_of(a)]
            fa = flat_n[:, a].reshape(-1, self.obs_dim)
            mean = net.actor.dist(fa).mean
            out[:, a] = mean.view(B, self.act_dim)
        return out

    def state_dict(self):
        return {"policies": self.policies.state_dict()}

    def load_state(self, sd):
        if "policies" in sd:
            self.policies.load_state_dict(sd["policies"])
        elif "state" in sd:            # legacy week-1 single-policy checkpoint
            self.policies[0].load_state_dict(sd["state"])

    def save(self, path):
        # NOTE: rms_obs MUST be persisted.  act_det() normalises observations
        # with these running statistics; without them a freshly constructed
        # Trainer has mean=0/var=1, feeds the policy mis-scaled inputs, and the
        # policy emits near-zero actions.  Every checkpoint-based evaluation was
        # silently invalid before this fix.
        torch.save({"__kind__": self.kind, "N": self.N,
                    "policies": self.policies.state_dict(),
                    "rms_mean": self.rms_obs.mean, "rms_var": self.rms_obs.var,
                    "rms_count": self.rms_obs.count}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.load_state(ck)
        self._load_rms(ck)

    def _load_rms(self, ck):
        if "rms_mean" in ck and "rms_var" in ck:
            self.rms_obs.mean = ck["rms_mean"].to(self.device).float()
            self.rms_obs.var = ck["rms_var"].to(self.device).float()
            self.rms_obs.count = float(ck.get("rms_count", 1.0))
