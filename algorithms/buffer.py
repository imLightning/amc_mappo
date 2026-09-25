"""
buffer.py -- on-the-fly rollout storage used by the PPO/MAPPO trainer.

Data is stacked into big tensors over shape (T*B*N) whenever an update is
triggered, but accumulation uses python lists of per-step torch tensors to
keep it transparent & debuggable.  RAM is negligible for our sizes.
"""
import torch


class RolloutBuffer:
    """accumulates per (step,B,N) samples across kind *vector_env tasks*.

    On each env-step the trainer feeds, for every parallel env & agent:
        obs      (B,N,obs_dim)
        state    (B,state_dim)            (same for each agent; broadcast)
        action   (B,N,act)
        logp     (B,N)
        rew      (B,N)
        done     (B,)                     1 -> episode terminated (no bootstrap)
        value    (B,N)   *critic estimate for current step (obs or state)*
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.state = []
        self.act = []
        self.logp = []
        self.rew = []
        self.mask = []      # 0 at done else 1  → for returns bootstrap
        self.val = []
        self.val_cost = []  # (B,N) safety-cost value estimate (RA-CMAPPO)
        self.cost = []      # (B,N) per-step safety cost (RA-CMAPPO)

    def push(self, obs, state, act, logp, rew, done, val):
        self.obs.append(obs)
        self.state.append(state)
        self.act.append(act)
        self.logp.append(logp)
        self.rew.append(rew)
        self.mask.append(1.0 - done.float().unsqueeze(-1).expand_as(rew))
        self.val.append(val.detach())

    def push_cost(self, val_cost, cost):
        """store safety-cost value and per-step cost, matched to last step."""
        self.val_cost.append(val_cost.detach())
        self.cost.append(cost.detach())

    def cost_stack(self):
        return {"val_cost": torch.stack(self.val_cost, 0) if self.val_cost
                else None,
                "cost": torch.stack(self.cost, 0) if self.cost else None}

    def compute_returns_gae_cost(self, gamma, lam, last_cost_val=None):
        """GAE advantages + discounted returns for the SAFETY cost.

        Returns (advs, rets), each a list of per-step tensors.

        NOTE: this function previously ended at `advs.reverse()` with NO return
        statement and no callers, so the trainer silently fell back to
        regressing the safety critic onto the instantaneous cost.  That removed
        any temporal credit assignment from the safety objective -- exactly the
        ability needed to trade a small detour now for avoiding distress later.
        """
        L = len(self.mask)
        advs = []
        next_value = last_cost_val if last_cost_val is not None \
            else torch.zeros_like(self.cost[0])
        gae_prev = torch.zeros_like(self.cost[0])
        for t in reversed(range(L)):
            boot = next_value * self.mask[t]
            delta = self.cost[t] + gamma * boot - self.val_cost[t]
            gae = delta + gamma * lam * self.mask[t] * gae_prev
            advs.append(gae)
            next_value = self.val_cost[t]
            gae_prev = gae
        advs.reverse()
        rets = [a + v for a, v in zip(advs, self.val_cost)]
        return advs, rets
    def compute_returns_gae(self, gamma, lam, last_val=None):
        """compute GAE advantage & returns in place, returns flat arrays.

        last_val: if provided, used to bootstrap the final step (only when the
                  final step is NOT done). Trainer supplies value(state_final).
        This helper follows the classic synchronous-PPO scan over the stored
        sequence; because the vector envs' internal resets make each step
        already a fresh continuation, we treat episode boundary via mask values
        observed at each step (bootstrap only where mask==1).
        """
        L = len(self.mask)
        advs = []
        next_value = last_val if last_val is not None else torch.zeros_like(
            self.val[-1])
        gae_prev = torch.zeros_like(self.val[0])
        # masks stored mask==1 means "non-terminal / continues" already
        for t in reversed(range(L)):
            boot = next_value * self.mask[t]
            delta = self.rew[t] + gamma * boot - self.val[t]
            gae = delta + gamma * lam * self.mask[t] * gae_prev
            advs.append(gae)
            next_value = self.val[t]
            gae_prev = gae
        advs.reverse()
        return advs

    def stack(self):
        return {
            "obs": torch.stack(self.obs, 0),
            "state": torch.stack(self.state, 0),
            "act": torch.stack(self.act, 0),
            "logp_old": torch.stack(self.logp, 0),
            "rew": torch.stack(self.rew, 0),
            "mask": torch.stack(self.mask, 0),
            "val": torch.stack(self.val, 0),
        }
