"""merge_layer_seeds.py -- pool the three-policy-seed evaluations of the RECOMMENDED
configuration into one file, so the paper's main table can report the proposed method
(AMC-MAPPO) with a training-seed spread, exactly like every other multi-seed arm.

The recommended configuration is the pedestrian-first (lexicographic) solver with an
early-yield margin of 0.75 seconds and the reciprocal robot-robot barrier, i.e. the
`lex-t0.75` arm of `ops/queue/queue_layer_3seed.sh`.

Usage: python scripts/merge_layer_seeds.py
Writes: results/bench/amc_mappo_p9.json  (labels AMC-MAPPO-s0/s1/s2)
"""
from __future__ import annotations

import json
import os

B = "results/bench"
TAG = "lex-t0.75"
N = 3


def main():
    summary, per_seed = [], []
    for k in range(N):
        p = os.path.join(B, f"layer3s_{TAG}-s{k}.json")
        if not os.path.exists(p):
            print(f"  [skip] {p} missing")
            continue
        d = json.load(open(p))
        for r in d["summary"]:
            if r["label"] == f"{TAG}-s{k}":
                r = dict(r)
                r["label"] = f"AMC-MAPPO-s{k}"
                summary.append(r)
        per_seed += [dict(r, label=f"AMC-MAPPO-s{k}")
                     for r in d.get("per_seed", []) if r.get("label") == f"{TAG}-s{k}"]
    if not summary:
        raise SystemExit("no layer evaluations found; run queue_layer_3seed.sh first")
    out = os.path.join(B, "amc_mappo_p9.json")
    json.dump(dict(summary=summary, per_seed=per_seed), open(out, "w"), indent=1)
    print(f"merged {len(summary)} policy seeds -> {out}")


if __name__ == "__main__":
    main()
