#!/usr/bin/env python3
"""Calibrate the early-yield time margin for a policy and scene.

    python tools/calibrate.py --recipe configs/<recipe>.yaml --ckpt <ckpt> --P 9 \
        --peer --cbf-mode lex --out results/bench/tau_calib.json

Sweeps the margin, reports success, affect cost, contact rates and the minimum
distance, marks a knee by the maximum-distance-to-the-chord rule, and writes the curve
as JSON.  Arguments are passed straight through."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dispatch import run  # noqa: E402

if __name__ == "__main__":
    run("calibrate_tau")
