"""calibration_table.py -- render the tau-calibration JSONs as paper tables.

Reads results/bench/tau_calib_dens_P*.json produced by scripts/calibrate_tau.py and
prints one table per density with the knee marked, so the operating point is
chosen by the stated rule (max distance to the chord on the success/cvar curve)
rather than by eye.

Usage: python scripts/calibration_table.py > results/bench/TAU_CALIBRATION.md
"""
from __future__ import annotations

import glob
import json
import os


def run_name(ckpt):
    """runs/<name>/ckpt/seed<k>/final.pt -> <name>  (which POLICY was swept)."""
    if not ckpt:
        return "?"
    parts = str(ckpt).split(os.sep)
    if "runs" in parts:
        i = parts.index("runs")
        if i + 1 < len(parts):
            return parts[i + 1]
    return os.path.basename(os.path.dirname(os.path.dirname(str(ckpt))))


def main():
    files = sorted(glob.glob("results/bench/tau_calib_*_P*.json"))
    print("# τ 标定（密度课程策略 + 互惠同伴；3 环境种子）\n")
    if not files:
        print("_（尚无标定结果）_")
        return
    print("规则：`scripts/calibrate_tau.py` 在 (success, cvar) 曲线上取"
          "\"到首尾弦的最大距离\"点为 knee。\n")

    # ---- knee summary, grouped by (policy, solver): the operating point must
    # never be read off a table whose policy is unidentified.
    ds = [(f, json.load(open(f))) for f in files]

    def degenerate(d):
        """A knee on a flat/near-zero-success curve is an artefact, not a result.

        This is not hypothetical: `tau_calib_lex_P20.json` sweeps a policy whose
        success is 0.036 at every tau (it was trained up to P=16), so the
        max-distance-to-the-chord rule still returns a number.  Flagging it here
        stops that number being quoted as "the P=20 knee".
        """
        best = max(r["success"] for r in d["rows"])
        return best < 0.10

    groups: dict[tuple, dict] = {}
    for _, d in ds:
        groups.setdefault((run_name(d.get("ckpt")), d.get("mode", "joint")), {})[d["P"]] = d
    print("## 0. knee 汇总（按策略与求解器分组）\n")
    cols = sorted({P for g in groups.values() for P in g})
    print("| 策略 (run) | 求解器 | " + " | ".join(f"P={P}" for P in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for (run, mode), g in sorted(groups.items()):
        cells = []
        for P in cols:
            if P not in g:
                cells.append("—")
            elif degenerate(g[P]):
                cells.append(f"~~{g[P]['knee_tau']}~~ ⚠")
            else:
                cells.append(f"**{g[P]['knee_tau']}**")
        print(f"| `{run}` | {mode} | " + " | ".join(cells) + " |")
    print("\n读法：knee 随策略/密度/求解器在 **0.5–1.25 s** 间波动，且**不单调**"
          "（均匀交错策略是 1.25 → 0.75 → 1.0 → 0.75）——论文应报告整条曲线，"
          "并把 τ≈0.75 当作跨密度工作点，而不是把某个 knee 当作普适常数。")
    print("⚠ = 该点曲线退化（全程 success < 10%，策略在该密度超出能力边界），"
          "knee 是「到弦最大距离」规则的产物、**不可引用**。\n")

    for f, d in ds:
        P = d.get("P")
        print(f"## P = {P}（策略 `{run_name(d.get('ckpt'))}`，"
              f"solver={d.get('mode','joint')}，peer={d.get('peer')}，α={d.get('alpha')}）\n")
        if degenerate(d):
            print("> ⚠ **该曲线退化**（全程 success < 10%）：策略在此密度超出能力边界，"
                  "下面的 knee 只是规则输出，**不可作为工作点引用**。\n")
        print("| τ (s) | success | cvar ↓ | social@0.55 | R–P 接触 | R–R 接触 | min_rp | nav (s) |")
        print("|---|---|---|---|---|---|---|---|")
        for r in d["rows"]:
            mark = " **← knee**" if r["tau"] == d.get("knee_tau") else ""
            print(f"| {r['tau']:.2f}{mark} | {r['success']:.3f} | {r['cvar']:.2f} | "
                  f"{r['social']:.3f} | {r['coll_rp']:.3f} | {r['coll_rr']:.3f} | "
                  f"{r['min_rp']:.3f} | {r['nav']:.2f} |")
        print(f"\n**knee τ = {d.get('knee_tau')}**\n")


if __name__ == "__main__":
    main()
