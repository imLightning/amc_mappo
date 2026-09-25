#!/usr/bin/env python3
"""Diagnostics that explain a failure rather than only reporting it.

    python tools/diagnose.py --what noise          # policy sensitivity to observation noise
    python tools/diagnose.py --what policy --ckpt <ckpt> [--attention 1] [--head-span 1.0]

`noise` sweeps an observation-noise level and reports success, mean speed, the alignment
between velocity and goal direction, and the net progress, so a collapse can be
attributed to perception rather than assumed.
`policy` reports the action histogram of a discrete-action checkpoint and its velocity-goal
alignment; `--head-span` must match the value the checkpoint was trained with, because the
span is part of the action decoding and not of the network."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dispatch import run  # noqa: E402


def main():
    argv = sys.argv[1:]
    what = "noise"
    if "--what" in argv:
        i = argv.index("--what")
        what = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if what not in ("noise", "policy"):
        raise SystemExit("--what must be 'noise' or 'policy'")
    run("diag_obs_noise" if what == "noise" else "diag_sarl", argv)


if __name__ == "__main__":
    main()
