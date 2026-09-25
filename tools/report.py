#!/usr/bin/env python3
"""Regenerate the paper's tables and figures, and check them.

    python tools/report.py --list                 # what each group would run
    python tools/report.py --pool                 # pool multi-seed evaluations
    python tools/report.py --tables               # every table
    python tools/report.py --figures              # every figure
    python tools/report.py --checks               # the five table checks
    python tools/report.py --all                  # pool, tables, figures, checks
    python tools/report.py --one density_table    # a single generator
    python tools/report.py --paired A.json:label B.json:label

The generators read the evaluation result files under `results/bench/` and write into the
same directory (some print markdown to standard output, and this tool redirects them to
their artifact).  Without those result files the generators report missing inputs; see the
README for what is available on request.

Order matters: `--pool` merges the per-policy evaluations that the multi-seed tables need.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dispatch import ROOT, run  # noqa: E402

BENCH = os.path.join(ROOT, "results", "bench")

# generator -> artifact it must produce (empty string: the generator writes its own file)
TABLES = {
    "fill_main_table": "",
    "paper_table_en": "",
    "paper_table_route_a": "PAPER_TABLE_ROUTE_A_EN.md",
    "affect_layer_table": "AFFECT_LAYER_TABLE.md",
    "density_table": "DENSITY_TABLE.md",
    "robustness_table": "ROBUSTNESS.md",
    "shape_control_table": "SHAPE_CONTROL.md",
    "layer_3seed": "LAYER_3SEED.md",
    "calibration_table": "TAU_CALIBRATION.md",
    "seed_aggregate": "SEED_AGGREGATE.md",
    "frontier_summary": "FRONTIER_SUMMARY.md",
    "significance": "",
    "claims": "CLAIMS.md",
}
FIGURES = ["make_tau_figure", "make_route_a_density", "make_trajectory_figure",
           "make_pareto_figure", "make_density_figure"]
POOL = ["merge_layer_seeds", "merge_sarl_seeds"]
CHECKS = ["verify_tables"]


def call(module: str) -> None:
    importlib.import_module(f"tools._impl.{module}").main()


def call_capture(module: str, dest: str) -> None:
    """Run a generator that prints markdown, redirecting it to its artifact."""
    os.makedirs(BENCH, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        call(module)
    with open(os.path.join(BENCH, dest), "w") as fh:
        fh.write(buf.getvalue())
    print(f"  {module:22s} -> results/bench/{dest}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    if "--list" in argv:
        print("pool    :", " ".join(POOL))
        print("tables  :", " ".join(TABLES))
        print("figures :", " ".join(FIGURES))
        print("checks  :", " ".join(CHECKS))
        return 0
    todo = []
    if "--all" in argv:
        todo = [("pool", None)] + [("tables", None)] + [("figures", None)] + [("checks", None)]
    else:
        if "--pool" in argv:
            todo.append(("pool", None))
        if "--tables" in argv:
            todo.append(("tables", None))
        if "--figures" in argv:
            todo.append(("figures", None))
        if "--checks" in argv:
            todo.append(("checks", None))
        if "--one" in argv:
            todo.append(("one", argv[argv.index("--one") + 1]))
        if "--paired" in argv:
            i = argv.index("--paired")
            todo.append(("paired", argv[i + 1:]))
    if not todo:
        print(__doc__)
        return 2
    # Preflight: the generators read the evaluation result files, which are not part of
    # this repository (available on request).  Say so once, clearly, instead of letting
    # every generator fail in the middle of its work.
    needs_data = any(w in ("tables", "figures", "checks", "pool", "one") for w, _ in todo)
    have_data = os.path.isdir(BENCH) and any(
        f.endswith(".json") for f in os.listdir(BENCH)) if os.path.isdir(BENCH) else False
    if needs_data and not have_data:
        print("No evaluation results found under results/bench/.")
        print("They are not part of this repository and are available from the authors "
              "on request (see the README).  Nothing was regenerated.")
        return 3
    for what, arg in todo:
        if what == "pool":
            for m in POOL:
                call(m)
        elif what == "tables":
            for m, dest in TABLES.items():
                if dest:
                    call_capture(m, dest)
                else:
                    call(m)
        elif what == "figures":
            for m in FIGURES:
                call(m)
        elif what == "checks":
            for m in CHECKS:
                call(m)
        elif what == "one":
            if arg not in TABLES:
                raise SystemExit(f"--one takes one of {sorted(TABLES)}")
            dest = TABLES[arg]
            if dest:
                call_capture(arg, dest)
            else:
                call(arg)
        elif what == "paired":
            run("paired_test", arg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
