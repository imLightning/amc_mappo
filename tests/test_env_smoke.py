"""Build the environment from config.py defaults and step it; needs no data files."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from config import default  # noqa: E402
from envs.social_nav import SocialNavVecEnv  # noqa: E402


def main():
    cfg = default()
    cfg.env["n_pedestrians"] = 4.0
    env = SocialNavVecEnv(cfg, n_parallel_envs=2, device="cpu")
    obs = env.get_obs()
    assert obs.dim() == 3 and obs.shape[0] == 2
    for _ in range(5):
        env.step(torch.zeros(2, env.N, 2, device="cpu"))
    print(f"env smoke OK: obs_dim={env._obs_dim} N={env.N} P={env.P} "
          f"horizon={env.horizon}")


if __name__ == "__main__":
    main()
