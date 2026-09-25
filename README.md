# AMC-MAPPO

Code for **AMC-MAPPO** (affective multi-robot control with multi-agent proximal policy
optimisation).  The method is an **affect-aware safety layer** on the commanded velocity
of a robot team.  It attaches to a learned policy or to a classical planner, needs no
retraining, and can be switched off at run time.

The layer combines four things: an **anisotropic personal space** taken from an explicit
pedestrian affect model; an **early-yield time margin** that inflates the safety region in
proportion to the closing speed; a **pedestrian-first joint quadratic program** over all
per-robot constraints; and a **reciprocal robot-robot barrier**.  The definitions are in
`envs/cbf.py`, `envs/risk_field.py` and in the paper.

## Layout

```
config.py        configuration schema (code defaults)
envs/            affect model, risk field, safety layer, classical planners, metrics
algorithms/      policy, MAPPO trainer, constrained (risk-averse) trainer
configs/         the two recipes the regression guard needs
tools/           seven command-line tools (below); implementation in tools/_impl/
tests/           self-contained checks
```

## Install and check

```bash
python -m pip install -r requirements.txt     # torch 2.5, numpy, matplotlib, pyyaml
bash tests/run_all.sh
```

`tests/run_all.sh` runs three checks: every module imports, the environment builds and
steps from code defaults, and the bit-exact regression guard passes
(`both guarded behaviours unchanged (bit-exact) OK`).  The guard pins the legacy reward,
cost and robot-update behaviour by hash.

## The tools

| tool | what it does |
|---|---|
| `tools/evaluate.py` | evaluate an arm (policy or planner, with or without the layer) and write one result file |
| `tools/train.py` | train a policy: `--algorithm mappo`, `cvar` (constrained) or `sarl` (ported social-navigation baseline) |
| `tools/calibrate.py` | sweep the early-yield time margin for a policy and scene, mark a knee, write the curve |
| `tools/diagnose.py` | `--what noise` (policy sensitivity to observation noise) or `--what policy` (action histogram of a discrete-action checkpoint) |
| `tools/runtime.py` | per-step cost of the layer against the policy forward pass |
| `tools/guard.py` | the bit-exact regression guard |
| `tools/report.py` | regenerate every table and figure, and run the five table checks |

Every tool prints its own usage with `--help`.

### Evaluating an arm

```bash
python tools/evaluate.py --device cuda --seeds 1000,1001,1002 \
    --env n_pedestrians=9 --scripted beeline detour \
    --cbf-min 0.6 --cbf-alpha 0.5 --cbf-emotion --cbf-mode lex \
    --cbf-ttc-gain 0.75 --cbf-peer --cbf-peer-all --cbf-d-peer 0.70 \
    --out results/bench/example.json \
    --arms "configs/<recipe>.yaml:runs/<run>/ckpt/seed0/final.pt:label"
```

Omit the layer flags to evaluate the unfiltered policy; `--policy orca|sfm|dwa|sarl`
replaces the learned action with a classical planner or a ported social-navigation policy.
A discrete-action checkpoint must be accompanied by the `runs/<run>/seed<k>.json` record
of its training arguments: the evaluator reads the action-grid parameters from it and
refuses to decode with a mismatched grid, because the grid is part of the action interface
and not of the network.

### Regenerating the paper's tables and figures

```bash
python tools/report.py --list       # what each group runs
python tools/report.py --all        # pool, tables, figures, checks
python tools/report.py --one density_table
```

The groups are `--pool` (merge the per-policy evaluations of multi-seed arms),
`--tables` (thirteen table generators), `--figures` (five figure scripts) and `--checks`
(five mechanical checks: a table that differs from a fresh regeneration, a blank cell, an
arm label absent from the data, two arms with bit-identical metrics, and a document citing
a missing file).  `--paired` produces the paired comparisons by scenario seed.

## What is not in this repository

Evaluation results, the remaining experiment configurations, trained checkpoints,
training and evaluation logs, and the internal development reports.  They are available
from the authors on request; the paper states this.  `configs/` contains the only two
configurations included, because the regression guard needs them.  Without the withheld
files the table and figure generators cannot produce numbers; `tools/report.py` reports
that explicitly instead of failing mid-way.

## Licence

Not selected yet; all rights reserved by the authors until one is added.
