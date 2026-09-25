"""
results_paths.py -- single source of truth for where result artifacts live.

results/ is organized by topic (mirroring runs/):

    results/bench/   aggregated comparison tables + JSON (main data)
    results/media/   paper figures + animation/GIF assets
    results/ra/      RA-CMAPPO / AMC-MAPPO paper tables
    results/eval/    one-off evaluation JSON

    (Legacy per-week dirs week1/week2/week3 and vmas/ were removed in the
     2026-09-19 cleanup; see results/CLEANUP_LOG.md.)

Usage:
    from results_paths import RP
    p = RP("bench", "hard_sweep_eval.json")   # -> .../results/bench/hard_sweep_eval.json
    p.parent.mkdir(parents=True, exist_ok=True)
"""
from __future__ import annotations

import os
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJ / "results"

SUBDIRS = ("bench", "media", "ra", "eval")


def RP(sub: str = "", *parts) -> Path:
    """Return results/<sub>/<parts...>; creates results/<sub> lazily on write.

    Passing sub=""/None returns the results root (back-compat for legacy files).
    """
    if sub in ("", None):
        return RESULTS.joinpath(*parts)
    base = RESULTS / sub
    base.mkdir(parents=True, exist_ok=True)
    return base.joinpath(*parts)


def RP_dir(sub: str) -> Path:
    p = RESULTS / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
