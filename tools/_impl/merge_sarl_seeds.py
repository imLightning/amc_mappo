"""merge_sarl_seeds.py -- pool the repaired-SARL per-seed evaluations into ONE
file, the way every other multi-seed learned arm is stored in this project
(`p9_main.json`, `cbf3_acc1fix_3seed.json`, ... all hold the rows of every
training seed in a single json).

Why: `fill_main_table.py` reports `mean ± training-seed spread` by pooling the
labels it finds in ONE file, so three separate files would render as three
one-seed rows with no error bar.  Merging keeps the reporting convention.

Usage: python scripts/merge_sarl_seeds.py
Writes: results/bench/sarl_fix_3seed.json
"""
from __future__ import annotations

import json
import os

B = "results/bench"
PARTS = [("sarl_fix_P9.json", "SARL-port-fixed"),
         ("sarl_fix_s1_P9.json", "SARL-port-fixed-s1"),
         ("sarl_fix_s2_P9.json", "SARL-port-fixed-s2")]


def main():
    summary, per_seed, n = [], [], 0
    for f, lab in PARTS:
        p = os.path.join(B, f)
        if not os.path.exists(p):
            print(f"  [skip] {f} not evaluated yet")
            continue
        d = json.load(open(p))
        for r in d.get("summary", []):
            if r["label"] == lab:
                summary.append(r)
                n += 1
        per_seed += [r for r in d.get("per_seed", []) if r.get("label") == lab]
        print(f"  [ok] {f}: {lab}")
    if not summary:
        raise SystemExit("no repaired-SARL evaluations found; nothing merged")
    out = os.path.join(B, "sarl_fix_3seed.json")
    json.dump(dict(summary=summary, per_seed=per_seed), open(out, "w"), indent=1)
    print(f"merged {n} training seeds -> {out}")


if __name__ == "__main__":
    main()
