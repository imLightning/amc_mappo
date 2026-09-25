"""sarl.py -- a compact port of the CADRL/SARL family (Chen et al. 2017, 2019)
into THIS environment, as the paper's "social DRL" baseline.

What is faithful to the original:
  * value-based RL over a DISCRETE action set (speed x heading), value network
    over (self state, each human's state) with a **social attention** head that
    pools the humans (SARL);  swapping attention for an LSTM gives the LSTM-RL
    ablation of the same paper;
  * reward = goal progress + goal bonus - collision penalty - discomfort
    (TTC-based), i.e. the "socially aware" reward rather than only collision;
  * single-robot, decentralised: every robot runs the SAME network with no
    communication, which is exactly how a single-agent method is applied to a
    multi-robot scene (this must be stated in the paper).

What is adapted (and must be stated):
  * our action space is a velocity vector; the discrete set is
    {speeds} x {heading offsets relative to the goal direction}, and the chosen
    (v, theta) is emitted as a velocity command (the env then applies its own
    acceleration limit);
  * the crowd is the environment's 16 pedestrians (with affect), not the
    original 2D circle scenarios.

Training (scripts/train_sarl.py) uses DQN with a replay buffer and a target
network.  NOTE the honest risk: our scene needed ~55M agent-steps for MAPPO, so
a SARL port trained for a few hundred thousand transitions may be under-trained;
if so the results must be reported as such rather than dressed up.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn


def build_state(env, order_by_distance=True):
    """(B,N,Fself), (B,N,P,Fhuman), (B,N,P) mask -- SARL-style raw state.

    self : [cos(goal angle), sin(goal angle), dist/norm, vx/vmax, vy/vmax, r]
    human: [rel px/norm, rel py/norm, rel vx/vmax, rel vy/vmax, dist/norm, r]
    """
    B, N, P = env.B, env.N, env.P
    dev = env.device
    vmax = float(env.max_sp)
    pos, vel, goal = env.robo_pos, env.robo_vel, env.robo_goal
    g = goal - pos
    gd = g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    s = torch.cat([g / gd, gd / 6.0, vel / vmax, env.r * torch.ones_like(gd)], dim=-1)
    if P:
        rel = env.ped_pos.unsqueeze(1) - pos.unsqueeze(2)            # B,N,P,2
        rv = env.ped_vel.unsqueeze(1) - vel.unsqueeze(2)
        dist = rel.norm(dim=-1, keepdim=True)
        h = torch.cat([rel / 5.0, rv / vmax, dist / 5.0,
                       env.ped_r * torch.ones_like(dist)], dim=-1)
        mask = torch.ones(B, N, P, device=dev, dtype=torch.bool)
    else:
        h = torch.zeros(B, N, 0, 6, device=dev)
        mask = torch.zeros(B, N, 0, device=dev, dtype=torch.bool)
    return s, h, mask


class SARLNet(nn.Module):
    """value network V(s, humans) -> |A| values, with attention (or LSTM)."""

    def __init__(self, n_actions=40, hidden=128, n_human_feat=6, self_feat=6,
                 use_attention=True):
        super().__init__()
        self.use_attention = bool(use_attention)
        self.emb_self = nn.Sequential(nn.Linear(self_feat, hidden), nn.ReLU())
        self.emb_h = nn.Sequential(nn.Linear(n_human_feat, hidden), nn.ReLU())
        self.att = nn.Linear(hidden, hidden, bias=False)
        if not use_attention:
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_actions))

    def forward(self, s, h, mask=None):
        # CAREFUL with shapes: the closed-loop path passes h as (B,N,P,F) while
        # the DQN training batch passes the flattened buffer (M,P,F).  The original
        # code assumed the 4-D form, so `--use_attention 0` (the LSTM ablation)
        # crashed with "tuple index out of range" at the first update.
        es = self.emb_self(s)
        n_hum = h.shape[-2] if h.dim() >= 3 else 0
        if n_hum == 0:
            ctx = torch.zeros_like(es)
        else:
            eh = self.emb_h(h)                      # B,N,P,H
            if self.use_attention:
                # supports both (B,N,F) evaluation and (M,F) training batches
                a_ = self.att(es)
                a_ = a_.unsqueeze(1) if es.dim() == 2 else a_.unsqueeze(2)
                logits = (a_ * eh).sum(-1)                              # ...,P
                if mask is not None:
                    mm = mask
                    while mm.dim() < logits.dim():
                        mm = mm.unsqueeze(1) if mm.shape[0] == logits.shape[0] else mm
                    logits = logits.masked_fill(~mm, -1e9)
                w = torch.softmax(logits, dim=-1)
                ctx = (w.unsqueeze(-1) * eh).sum(dim=-2)
            elif eh.dim() == 3:                     # (M, P, H) training batch
                out, _ = self.lstm(eh)              # LSTM runs over the people
                ctx = out[:, -1]
            else:                                   # (B, N, P, H) closed loop
                out, _ = self.lstm(eh.reshape(-1, eh.shape[2], eh.shape[3]))
                ctx = out[:, -1].reshape(es.shape)
        return self.head(torch.cat([es, ctx], dim=-1))


class SARLPolicy:
    """evaluation-time wrapper: argmax over the discrete action set."""

    def __init__(self, ckpt=None, n_speeds=5, n_head=8, hidden=128,
                 use_attention=True, head_span=1.0, device="cpu"):
        self.n_speeds, self.n_head = int(n_speeds), int(n_head)
        self.head_span = float(head_span)
        self.net = SARLNet(n_actions=self.n_speeds * self.n_head, hidden=hidden,
                           use_attention=use_attention).to(device)
        if ckpt and os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location=device)
            self.net.load_state_dict(sd.get("net", sd))
        self.net.eval()
        self.device = device

    @torch.no_grad()
    def act(self, env):
        s, h, m = build_state(env)
        s, h = s.to(self.device), h.to(self.device)
        q = self.net(s, h, m.to(self.device))                     # B,N,A
        idx = q.argmax(dim=-1)
        iv, ih = idx // self.n_head, idx % self.n_head
        vs = torch.linspace(0.0, 1.0, self.n_speeds, device=self.device) * float(env.max_sp)
        hs = (torch.linspace(-1.0, 1.0, self.n_head, device=self.device)
              * self.head_span * np.pi)
        v = vs[iv]
        g = env.robo_goal - env.robo_pos
        gth = torch.atan2(g[..., 1], g[..., 0])
        th = gth + hs[ih]
        return torch.stack([v * torch.cos(th), v * torch.sin(th)], dim=-1)
