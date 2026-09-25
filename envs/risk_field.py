"""
risk_field.py -- RA-CMAPPO risk/social modelling utilities (plan_add.md §三.2).

Pure vectorised functions over (B,N,P,...) robot-vs-pedestrian geometry.  These
compute the terms the constrained MAPPO will consume (and mirror as signals for
the safety critic):

  * time-to-collision (TTC) with closing speed gating;
  * approach angle / relative orientation;
  * anisotropic personal space d_social(phi)  (front > side > rear);
  * short-term linear pedestrian prediction + per-ped covariance-based
    uncertainty proxy;
  * aggregate risk per robot.

All tensors/args should live on the same torch device.  No learning here.
"""
from __future__ import annotations

import numpy as np
import torch


def _stable(a, eps=1e-4):
    return torch.clamp_min(a, eps)


# ------------------------------------------------------------------ TTC
def closing_speed(rel, relv):
    """relative velocity component along the rel direction; >0 = approaching.
    rel : (...,2), relv : (...,2)."""
    rn = _stable(rel.norm(dim=-1, keepdim=True))
    return -(rel * relv).sum(dim=-1) / rn.squeeze(-1)


def ttc(rel, relv, eps=1e-3):
    """TTC (time-to-collision) when closing; +inf otherwise."""
    d = _stable(rel.norm(dim=-1))
    cs = closing_speed(rel, relv)
    t = torch.where(cs > eps, d / _stable(cs, eps), float("inf"))
    return t


# ------------------------------------------------------------------ angle
def approach_angle(rel):
    """heading of rel in world frame [rad], shape (...,)."""
    a = torch.atan2(rel[..., 1], rel[..., 0])
    return a


def relative_bearing(human_heading, phi):
    """angle of the robot as seen from the human's heading, wrapped to [-pi,pi].
    human_heading: (...,)  phi: (...,) absolute bearing in world."""
    d = (phi - human_heading)
    return torch.remainder(d + np.pi, 2 * np.pi) - np.pi


# ------------------------------------------------------------------ anisotropy
def anisotropic_social_radius(rel_bearing, d_front=1.0, d_side=0.6,
                              d_back=0.4):
    """front is larger than side/rear (usually robot is behind/front of human).
    Approximates an ellipse by interpolating the radius between the three
    cardinal directions based on cos/sign symmetry of bearing.
    """
    c = torch.cos(rel_bearing)
    # bl1 for rear (bearing near ±pi), front bearing near 0
    front_w = torch.clamp(c, 0.0, 1.0)          # cos aligned with human front
    back_w = torch.clamp(-c, 0.0, 1.0)
    side_w = 1.0 - torch.abs(c)
    return front_w * d_front + back_w * d_back + side_w * d_side


# ------------------------------------------------------------------ predict
def linear_forecast(pos, vel, k_steps, dt):
    """deterministic short-term predicted position (...,2)."""
    return pos + vel * (k_steps * dt)


def prediction_uncertainty(vel, dt, sigma_v=0.2, t_horizon=0.5):
    """simple growing-with-horizon positional covariance proxy (isotropic)."""
    return (sigma_v ** 2) * (t_horizon ** 2)


def ttc_from_all(rel, relv):
    return ttc(rel, relv)


def full_risk(robo_pos, robo_vel, ped_pos, ped_vel, ped_heading,
              dt=0.05, k_pred=4, cfg_risk=None):
    """aggregate per-(B,N) social risk given full pedestrian field.

    Returns dict of tensors (each (B,N)): risk, min_ttc, ttc_low1, ttc_low2,
    directional_soc_min, plus the merged risk value used by critic scaling."""
    cfg_risk = cfg_risk or {}
    B, N, _ = robo_pos.shape
    P = ped_pos.shape[1]
    dev = robo_pos.device
    rel = robo_pos.unsqueeze(2) - ped_pos.unsqueeze(1)       # B,N,P,2
    relv = robo_vel.unsqueeze(2) - ped_vel.unsqueeze(1)      # B,N,P,2
    dist = _stable(rel.norm(dim=-1))                         # B,N,P
    cs = closing_speed(rel, relv)                            # B,N,P
    t = ttc_from_all(rel, relv)                              # B,N,P
    phi = approach_angle(rel)                                # B,N,P
    # directional social radius
    if ped_heading is None:
        ped_heading = torch.atan2(ped_vel[..., 1], ped_vel[..., 0])
    bear = relative_bearing(ped_heading[:, None, :].expand(B, N, P), phi)
    d_soc = anisotropic_social_radius(bear,
                                      d_front=cfg_risk.get("d_front", 1.0),
                                      d_side=cfg_risk.get("d_side", 0.6),
                                      d_back=cfg_risk.get("d_back", 0.4))
    # risk components (clamped into practical magnitude)
    sig_d = cfg_risk.get("d_sigma", 0.55); d_safe = cfg_risk.get("d_safe", 0.25)
    tau = cfg_risk.get("ttc_tau", 0.8)
    r_dist = torch.exp(-((dist - d_safe) ** 2) / (sig_d ** 2))
    r_ttc = torch.where(cs > 1e-3, torch.exp(-t / tau), torch.zeros_like(t))
    r_vel = torch.clamp(relv.norm(dim=-1), 0.0, 2.0) / 2.0
    bear_cos = torch.cos(bear)                                # approach dir
    r_angle = torch.clamp(-bear_cos, 0.0, 1.0)                # more when front, opp
    # uncertainty proxy grows with speed for each pair
    unc = (relv.norm(dim=-1) ** 2) * (dt ** 2) * (k_pred ** 2)
    r_unc = torch.sigmoid(unc).clamp(0.0, 1.0)

    wd = cfg_risk.get("w_distance", 1.0); wt = cfg_risk.get("w_ttc", 1.2)
    wv = cfg_risk.get("w_relative_velocity", 0.3)
    wa = cfg_risk.get("w_approach_angle", 0.3)
    wu = cfg_risk.get("w_uncertainty", 0.2)
    raw = wd * r_dist + wt * r_ttc + wv * r_vel + wa * r_angle + wu * r_unc
    # aggregation: sum then smooth with spatial gate
    agg = raw.sum(dim=-1)                                     # B,N
    risk = agg
    # TTC below thresholds (per robot any ped)
    min_ttc = t.amin(dim=-1)                                  # B,N
    ttc_low1 = (t < 1.0).any(dim=-1).float()                  # B,N
    ttc_low2 = (t < 2.0).any(dim=-1).float()                  # B,N
    # directional social minimum distance (real distance, radius soft)
    viol_frac = torch.clamp(d_soc - dist, 0.0, None)
    dir_soc_min = viol_frac.amin(dim=-1)                       # most-invaded pair
    return {
        "risk": risk, "min_ttc": min_ttc,
        "ttc_low1": ttc_low1, "ttc_low2": ttc_low2,
        "directional_soc": dir_soc_min, "dist_min": dist.amin(dim=-1),
        "d_soc": d_soc,
    }
