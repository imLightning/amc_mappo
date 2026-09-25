#!/usr/bin/env python3
"""Bit-exact regression guard.

    python tools/guard.py --threads 1
    # expected last line: both guarded behaviours unchanged (bit-exact) OK

Pins the legacy reward, cost and robot-update behaviour by hash, so any accidental
change to those paths fails loudly instead of silently altering published numbers."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dispatch import run  # noqa: E402

if __name__ == "__main__":
    run("regress_legacy")
