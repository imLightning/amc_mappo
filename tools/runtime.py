#!/usr/bin/env python3
"""Measure the per-step cost of the safety layer against the policy forward pass.

    python tools/runtime.py --device cuda [--batches 1,128] [--peds 5,9,16,20]

Writes results/bench/FILTER_RUNTIME.md and results/bench/filter_runtime.json."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dispatch import run  # noqa: E402

if __name__ == "__main__":
    run("bench_filter_time")
