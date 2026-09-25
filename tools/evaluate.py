#!/usr/bin/env python3
"""Evaluate an arm and write one result file.

Examples:
    python tools/evaluate.py --device cuda --seeds 1000,1001,1002 \
        --env n_pedestrians=9 --scripted beeline detour \
        --cbf-min 0.6 --cbf-emotion --cbf-mode lex --cbf-ttc-gain 0.75 \
        --cbf-peer --cbf-peer-all --cbf-d-peer 0.70 \
        --out results/bench/example.json \
        --arms "configs/<recipe>.yaml:runs/<run>/ckpt/seed0/final.pt:label"

With the safety-layer arguments omitted the unfiltered policy is evaluated.
See `python tools/evaluate.py --help` for the full flag list."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dispatch import run  # noqa: E402

if __name__ == "__main__":
    run("pareto_eval")
