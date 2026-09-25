#!/usr/bin/env python3
"""Train a policy.

    python tools/train.py --algorithm mappo --recipe configs/<recipe>.yaml --seed 0 ...
    python tools/train.py --algorithm cvar  --recipe configs/<recipe>.yaml --seed 0 ...
    python tools/train.py --algorithm sarl  --recipe configs/<recipe>.yaml --seed 0 ...

`mappo` trains the multi-agent policy-gradient baseline, `cvar` the same trainer under a
conditional-value-at-risk constraint, and `sarl` the discrete-action ported
social-navigation baseline.  Arguments after `--algorithm` are passed through."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dispatch import run  # noqa: E402

ALGOS = {"mappo": "train_ra", "cvar": "train_ra", "sarl": "train_sarl"}


def main():
    argv = sys.argv[1:]
    algo = "mappo"
    if "--algorithm" in argv:
        i = argv.index("--algorithm")
        algo = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if algo not in ALGOS:
        raise SystemExit(f"--algorithm must be one of {sorted(ALGOS)}")
    if algo == "cvar":
        argv += ["--mode", "RA-CMAPPO"]
    run(ALGOS[algo], argv)


if __name__ == "__main__":
    main()
