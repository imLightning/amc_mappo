"""QA for the auto-generated bench tables.

Three checks, all mechanical:

1. **Stale check** -- re-run every generator into a temp copy and diff against the
   committed markdown.  A diff means the committed table is stale (someone forgot
   to regenerate) or a generator is non-deterministic.
2. **Empty-cell check** -- every markdown row inside a table must have all its
   cells filled.  An empty cell is a "we planned this row but never measured it"
   marker; it must never survive into a paper artifact silently.
3. **Label check** -- every JSON label a table *asks for* must exist in the JSON
   it reads.  A silently-missing label previously produced blank rows that looked
   like real data, so this is checked explicitly.
"""
import os
import re
import shutil
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = os.path.join(PROJ, "results", "bench")
PY = sys.executable

GENERATORS = [
    "fill_main_table.py", "paper_table_en.py", "significance.py",
    "affect_layer_table.py", "paper_table_route_a.py", "frontier_summary.py",
    "calibration_table.py", "density_table.py", "seed_aggregate.py",
    "layer_3seed.py", "shape_control_table.py", "robustness_table.py",
    "claims.py",
]
# Which artifact each generator must produce.  An EMPTY value means the script
# writes its file itself; a non-empty value means it prints markdown to stdout
# and must be redirected into that artifact.  (Getting this wrong is how the
# committed tables went stale while every generator still reported "ok".)
GEN_ARTIFACT = {
    "fill_main_table.py": "",
    "paper_table_en.py": "",
    "significance.py": "",
    "affect_layer_table.py": "AFFECT_LAYER_TABLE.md",
    "paper_table_route_a.py": "PAPER_TABLE_ROUTE_A_EN.md",
    "frontier_summary.py": "FRONTIER_SUMMARY.md",
    "calibration_table.py": "TAU_CALIBRATION.md",
    "layer_3seed.py": "LAYER_3SEED.md",
    "shape_control_table.py": "SHAPE_CONTROL.md",
    "robustness_table.py": "ROBUSTNESS.md",
    "claims.py": "CLAIMS.md",
    "density_table.py": "DENSITY_TABLE.md",
    "seed_aggregate.py": "SEED_AGGREGATE.md",
}

ARTIFACTS = [
    "MAIN_TABLE_FILLED.md", "PAPER_TABLE_EN.md", "AFFECT_LAYER_TABLE.md",
    "PAPER_TABLE_ROUTE_A_EN.md", "FRONTIER_SUMMARY.md", "TAU_CALIBRATION.md",
    "DENSITY_TABLE.md", "SEED_AGGREGATE.md", "LAYER_3SEED.md",
    "SHAPE_CONTROL.md", "FILTER_RUNTIME.md", "ROBUSTNESS.md", "CLAIMS.md",
]

CELL = re.compile(r"^\|(.+)\|\s*$")


def run(cmd, cwd=PROJ):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def regenerate(gen):
    """Run one generator the way its docstring says, returning (ok, stderr)."""
    dest = GEN_ARTIFACT.get(gen, "")
    if dest:
        with open(os.path.join(B, dest), "w") as fh:
            r = subprocess.run([PY, os.path.join("tools", "_impl", gen)], cwd=PROJ,
                               stdout=fh, stderr=subprocess.PIPE, text=True)
    else:
        r = run([PY, os.path.join("tools", "_impl", gen)])
    return (r.returncode == 0), (r.stderr or "").strip()[-300:]


def stale_check():
    """Regenerate every artifact the way it is built; diff against the committed one."""
    bad = []
    before = {}
    for a in ARTIFACTS:
        p = os.path.join(B, a)
        if os.path.exists(p):
            before[a] = open(p).read()
    for gen in GENERATORS:
        ok, err = regenerate(gen)
        if not ok:
            bad.append(f"{gen}: exited non-zero: {err}")
    for a, old in before.items():
        p = os.path.join(B, a)
        new = open(p).read() if os.path.exists(p) else ""
        if new != old:
            bad.append(f"{a}: committed copy was STALE (regeneration changed it)")
    return bad


def empty_cells():
    """Rows in a table whose body has a blank cell."""
    bad = []
    for a in ARTIFACTS:
        p = os.path.join(B, a)
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p), 1):
            if not line.startswith("|"):
                continue
            if set(line.strip()) <= set("|-: "):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            blank = [j for j, c in enumerate(cells) if c == ""]
            if blank:
                bad.append(f"{a}:{i}: {len(blank)} blank cell(s) -> {line.strip()[:110]}")
    return bad


def label_check():
    """Labels referenced by the generators must exist in their JSONs."""
    import json
    bad = []
    src = open(os.path.join(PROJ, "scripts", "paper_table_route_a.py")).read()
    # crude but effective: find ("file.json", "label" ...) and ("file.json", ["l1","l2"] ...)
    for m in re.finditer(r'\("([A-Za-z0-9_]+\.json)",\s*(\[[^\]]*\]|"[^"]*")', src):
        fname, labels = m.group(1), m.group(2)
        path = os.path.join(B, fname)
        if not os.path.exists(path):
            bad.append(f"{fname}: file missing")
            continue
        d = json.load(open(path))
        have = {r["label"] for r in d.get("summary", [])}
        want = re.findall(r'"([^"]+)"', labels)
        miss = [w for w in want if w not in have]
        if miss:
            bad.append(f"{fname}: labels absent {miss} (have {sorted(have)})")
    return bad


def duplicate_rows():
    """A non-scripted arm whose metrics are bit-identical to a scripted anchor.

    This caught a real defect: `--policy sarl --arms '...:-:LSTM-RL'` silently
    skipped the controller (no ckpt -> tr is None) and the "LSTM-RL" row was a
    copy of scripted-beeline.  Any such coincidence is a label/value mismatch
    until proven otherwise.
    """
    import json
    import glob
    keys = ("success", "cvar_raw", "coll_rp", "coll_rr")
    bad = []

    def sig(r):
        out = []
        for k in keys:
            if k not in r:
                return None
            v = r[k][0] if isinstance(r[k], list) else r[k]
            out.append(round(float(v), 6))
        return tuple(out)

    for f in sorted(glob.glob(os.path.join(B, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        rows = d.get("summary") or []
        if not rows or not isinstance(rows[0], dict) or "label" not in rows[0]:
            continue
        for anchor in ("scripted-beeline", "scripted-detour", "scripted-stop",
                       "scripted-creep"):
            a = next((r for r in rows if r["label"] == anchor), None)
            if a is None or sig(a) is None:
                continue
            for r in rows:
                if r["label"].startswith("scripted") or r["label"] == a["label"]:
                    continue
                if sig(r) == sig(a):
                    bad.append(f"{os.path.basename(f)}: '{r['label']}' == '{anchor}' "
                               "(label/value mismatch?)")
    return bad


def broken_references():
    """Every `*.json` / `*.md` / `*.png` named in a bench document must exist.

    Hand-written documents drift: a renamed result file leaves a citation that
    looks authoritative and points at nothing.  This check only looks at names
    that appear inside backticks or as bare filenames in the bench docs.
    """
    import glob
    bad = []
    pat = re.compile(r"[A-Za-z0-9_./-]+\.(?:json|md|png)")
    # only complete, self-contained filenames: this skips line fragments such as
    # `_t075p.json` (the tail of `wtd_P*_t075p.json`) and shorthands like
    # `cls_orca_P5/9/16.json`, which are abbreviations rather than paths.
    whole = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:json|md|png)$")
    roots = [B, PROJ, os.path.join(PROJ, "results"),
             os.path.dirname(PROJ)]          # workspace root (plan*.md, HANDOVER.md)
    docs = sorted(glob.glob(os.path.join(B, "*.md")))
    docs.append(os.path.join(PROJ, "results", "NIGHT_REPORT.md"))
    for doc in docs:
        if not os.path.exists(doc):
            continue
        for i, line in enumerate(open(doc, encoding="utf-8"), 1):
            for name in pat.findall(line):
                if not whole.match(name):
                    continue
                if any(os.path.exists(os.path.join(r, name)) for r in roots):
                    continue
                if glob.glob(os.path.join(B, name)):
                    continue
                bad.append(f"{os.path.basename(doc)}:{i}: missing file `{name}`")
    return bad


def main():
    fails = 0
    for name, fn in (("stale", stale_check), ("empty-cell", empty_cells),
                     ("label", label_check), ("duplicate-row", duplicate_rows),
                     ("broken-ref", broken_references)):
        out = fn()
        print(f"[{name}] {'OK' if not out else f'{len(out)} problem(s)'}")
        for line in out:
            print("   -", line)
        fails += len(out)
    print("\nVERIFY_TABLES", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
