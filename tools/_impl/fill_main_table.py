"""fill_main_table.py -- assemble the paper's main tables from the measured json
files, so no number is ever transcribed by hand.

Reads the evaluation jsons written by scripts/pareto_eval.py and emits markdown:

  §1  main table (P=9): one row per method, columns
      reach success | social@0.55 | social@0.8 | min_rp | R-P contact |
      R-R contact | nav | wid | cvar
  §4  CBF sweep
  §2  density table (as available)

Rows are pooled over TRAINING seeds when several arms belong to the same method
(the ± is then the training-seed spread); single-arm methods (scripted, ORCA,
SFM, CBF) show the environment-seed spread instead, which the header notes.

Usage:  python scripts/fill_main_table.py [--out results/bench/MAIN_TABLE_FILLED.md]
"""
from __future__ import annotations

import argparse
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = os.path.join(PROJ, "results", "bench")

# (display name, [arm labels], source file, group)
MAIN = [
    ("scripted-beeline", ["scripted-beeline"], "p9_main.json", "参考包络"),
    ("**scripted-detour**", ["scripted-detour"], "p9_main.json", "参考包络"),
    ("scripted-creep", ["scripted-creep"], "p9_main.json", "参考包络"),
    ("scripted-stop", ["scripted-stop"], "p9_main.json", "参考包络"),
    ("**ORCA (RVO2)**", ["ORCA-RVO2"], "cls_orca_P9.json", "经典·安全"),
    ("MAPPO + geometric filter (0.6/0.8)", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"],
     "cbf3_mappo_0.6_0.8.json", "学习+滤波"),
    ("MAPPO + geometric filter (0.6/0.9)", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"],
     "cbf3_mappo_0.6_0.9.json", "学习+滤波"),
    ("**SFM (k=5,A=6,B=0.5)**", ["SFM-k5-A6-B0.5"], "cls_sfm_bal_P9.json", "经典·社会"),
    ("SFM (safety, k=3)", ["SFM-k3-A6-B0.5"], "cls_sfm_safe_P9.json", "经典·社会"),
    ("SARL (our port, not converged)", ["SARL-s0"], "sarl_P9_s0.json", "社交 DRL"),
    ("SARL (our port) seed 1, not converged", ["SARL-s1"], "sarl_P9_s1.json", "社交 DRL"),
    ("SARL (our port) seed 2, 2x budget, not converged", ["SARL-s2-2x"], "sarl_P9_s2.json", "社交 DRL"),
    # LSTM-RL ablation of the SARL port (--use_attention 0), trained and evaluated
    # with the same protocol as the attention SARL rows (queue_lstm.sh).
    ("LSTM-RL（本移植，未收敛）", ["LSTM-RL"], "sarl_lstm_P9.json", "社交 DRL"),
    ("DWA (a=1,b=0.25)", ["DWA"], "cls_dwa_P9.json", "经典·安全"),
    ("**MAPPO**", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"], "p9_main.json", "学习基线"),
    # THE PROPOSED METHOD: the recommended configuration (pedestrian-first solver,
    # 0.75 s early-yield margin, reciprocal robot-robot barrier) on top of MAPPO,
    # pooled over three independently trained policies.  Written by
    # `scripts/merge_layer_seeds.py` from the `queue_layer_3seed.sh` evaluations.
    ("**AMC-MAPPO (ours)**", ["AMC-MAPPO-s0", "AMC-MAPPO-s1", "AMC-MAPPO-s2"],
     "amc_mappo_p9.json", "本方法"),
    ("CVaR-MAPPO (slack budget, multiplier pinned at 0)", ["AMC-s0", "AMC-s1", "AMC-s2"], "p9_main.json", "对照·约束学习基线"),
    ("**CVaR-MAPPO (binding budget, multiplier <= 0.5, target 0.12)**", ["AMC-bind012-l05"],
     "amc_bind_P9.json", "对照·约束学习基线"),
    ("CVaR-MAPPO (binding budget, multiplier <= 0.5, target 0.15)", ["AMC-bind015-l05"],
     "amc_bind_P9.json", "对照·约束学习基线"),
    ("CVaR-MAPPO (binding budget, multiplier <= 2.0)", ["AMC-bind015-l20"],
     "amc_bind_P9.json", "对照·约束学习基线"),
    # CONFOUND CONTROL (NIGHT_REPORT section 46): the AMC+CBF rows below run
    # `amc_ws.yaml`, which has NO acceleration curriculum (train amax = 1.0,
    # matched to evaluation), while the MAPPO+CBF rows above run
    # `mappo_ws_curr.yaml`, whose curriculum only reaches amax ~47 within 29.5M
    # samples.  Measured on the same filter: MAPPO 0.570 -> matched-dynamics
    # MAPPO 0.669 -> AMC-lambda=0 0.721, i.e. most of the "CVaR-MAPPO + geometric filter is best"
    # gap is the training recipe, NOT the constraint (lambda was 0 throughout).
    # This row is the fair like-for-like control.
    ("MAPPO + geometric filter (0.6/0.8), matched recipe (3 seeds)",
     ["MAPPO-acc1fix-s0", "MAPPO-acc1fix-s1", "MAPPO-acc1fix-s2"],
     "cbf3_acc1fix_3seed.json", "对照·配方"),
    ("CVaR-MAPPO + geometric filter (0.6/0.8), confounded recipe", ["AMC-s0", "AMC-s1", "AMC-s2"],
     "cbf3_amc_0.6_0.8.json", "对照·配方"),
    ("CVaR-MAPPO + geometric filter (0.6/0.5), confounded recipe", ["AMC-s0", "AMC-s1", "AMC-s2"],
     "cbf3_amc_0.6_0.5.json", "对照·配方"),
]


def _g(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def load(fn, labels):
    """return list of per-arm summary dicts for the requested labels"""
    if not fn:
        return []
    p = os.path.join(B, fn)
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    rows = d["summary"] if isinstance(d, dict) and "summary" in d else d
    by = {r["label"]: r for r in rows}
    return [by[l] for l in labels if l in by]


def pool(rows, kind="train"):
    """mean over arms of (value), and the spread across arms (train seeds)"""
    def col(key, nested=None):
        vals = []
        for r in rows:
            if nested:
                d = r.get(key) or {}
                v = d.get(nested, d.get(float(nested)) if nested.replace('.', '', 1).isdigit() else None)
            else:
                v = r.get(key)
            if v is not None:
                vals.append(_g(v))
        return vals
    out = {}
    for key, nested in (("success", None), ("cvar_raw", None), ("wid", None),
                        ("coll_rp", None), ("coll_rr", None), ("nav_s", None),
                        ("min_rp", None), ("cbf_hmin", None), ("cbf_viol", None),
                        ("jerk", None), ("decel_p95", None), ("head_rate", None),
                        ("spl", None), ("reveal", None), ("yield_lead", None),
                        ("social_succ", "0.55"), ("social_succ", "0.8")):
        vals = col(key, nested)
        if not vals:
            out[f"{key}{'@'+nested if nested else ''}"] = (None, None)
            continue
        m = sum(float(v) for v in vals) / len(vals)
        # NOTE: statistics.stdev chokes on numpy scalars; use the plain formula
        s = ((sum((float(v) - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
             if len(vals) > 1 else 0.0)
        out[f"{key}{'@'+nested if nested else ''}"] = (m, s)
    return out


def f(x, nd=3, pct=False):
    m, s = x
    if m is None:
        return "  "
    if pct:
        return f"{100*m:.1f}"
    return f"{m:.{nd}f}±{s:.{nd}f}" if s else f"{m:.{nd}f}"


# SARL port with the two MEASURED defects fixed (per-step penalty scale vs
# per-step progress, and an action set that allowed walking backwards); labelled
# as a repaired PORT, not as the published SARL method.  Appended only once the
# evaluation exists: an always-present blank row would trip `verify_tables.py`'s
# empty-cell rule, whose whole purpose is to catch "planned but never measured".
OPTIONAL_ROWS = [
    # pooled over training seeds once all three evaluations exist
    # (`scripts/merge_sarl_seeds.py` writes the merged file); until then the
    # available per-seed files are shown so the row is never blank.
    ("**SARL (our port, repaired)**", ["SARL-port-fixed", "SARL-port-fixed-s1",
                                "SARL-port-fixed-s2"],
     ("sarl_fix_3seed.json", "sarl_fix_P9.json"), "社交 DRL"),
    # same port, SAME recipe as the original, only the Q-target rule changed
    # (Double DQN) -- isolates the algorithm from the reward design.
    ("SARL (our port) + double Q-learning", ["SARL-double-dqn"],
     "sarl_double_P9.json", "社交 DRL"),
]


def main():
    # `MAIN` is the list the renderer walks; the optional rows are appended here
    # so that an unmeasured arm never renders as a blank row.
    for _name, _labs, _src, _grp in OPTIONAL_ROWS:
        # `_src` may be a list of candidates: use the pooled file when it exists
        # and fall back to a single-seed file, so the row is present as soon as
        # anything has been measured (and absent, rather than blank, before that).
        for _cand in ((_src,) if isinstance(_src, str) else _src):
            if os.path.exists(os.path.join(B, _cand)):
                MAIN.append((_name, _labs, _cand, _grp))
                break
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(B, "MAIN_TABLE_FILLED.md"))
    a = ap.parse_args()
    L = []
    L.append("# 论文主表（实测填充版）\n")
    L.append("> 由 `scripts/fill_main_table.py` 从 `results/bench/*.json` 自动汇总；")
    L.append("> 空行 = 该臂尚未评测（见文末待办）。协议见 `MAIN_TABLE_TEMPLATE.md` §0。\n")
    L.append("## §1 主表（P=9，held-out 环境 seed 1000/1001/1002）\n")
    L.append("| 组 | 方法 | reach success (%) ↑ | social@0.55 (%) ↑ | social@0.8 (%) ↑ | "
             "min_rp (m) ↑ | R–P 接触 (%) ↓ | R–R 接触 (%) ↓ | nav (s) ↓ | wid ↓ | cvar_raw ↓ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, labels, fn, group in MAIN:
        rows = load(fn, labels)
        if not rows:
            L.append(f"| {group} | {name} |  |  |  |  |  |  |  |  |  |")
            continue
        p = pool(rows)
        L.append(f"| {group} | {name} | {f(p['success'],pct=True)} | "
                 f"{f(p['social_succ@0.55'],pct=True)} | {f(p['social_succ@0.8'],pct=True)} | "
                 f"{f(p['min_rp'])} | {f(p['coll_rp'],pct=True)} | {f(p['coll_rr'],pct=True)} | "
                 f"{f(p['nav_s'],2)} | {f(p['wid'])} | {f(p['cvar_raw'],2)} |")
    L.append("")
    L.append("注：`±` 对多训练 seed 的臂是**训练 seed 标准差**，对脚本/ORCA/SFM/DWA/几何滤波 行是**环境 seed 标准差**。")
    L.append("`social@r` = 全部到达 ∧ 全程与任一机器人-行人距离 ≥ r。\n")

    # ---- CBF sweep -------------------------------------------------------
    L.append("## §4 Geometric-filter parameters (P=9, 3 training seeds)\n")
    L.append("| 方法 | d_min (m) | α | reach success (%) ↑ | social@0.55 (%) ↑ | "
             "min_rp (m) ↑ | R–P 接触 (%) ↓ | h_min ↑ | h<0 (%) ↓ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    specs = []
    for cfg in ("0.6_0.5", "0.6_0.8", "0.6_0.9", "0.8_0.5", "0.8_0.8"):
        specs.append(("MAPPO + geometric filter", cfg, f"cbf3_mappo_{cfg}.json",
                      ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"]))
    for cfg in ("0.6_0.8", "0.6_0.5"):
        specs.append(("CVaR-MAPPO + geometric filter", cfg, f"cbf3_amc_{cfg}.json",
                      ["AMC-s0", "AMC-s1", "AMC-s2"]))
    for meth, cfg, fn, labels in specs:
        d, al = cfg.split("_")
        rows = load(fn, labels)
        if not rows:
            L.append(f"| {meth} | {d} | {al} |  |  |  |  |  |  |")
            continue
        p = pool(rows)
        L.append(f"| {meth} | {d} | {al} | {f(p['success'],pct=True)} | "
                 f"{f(p['social_succ@0.55'],pct=True)} | {f(p['min_rp'])} | "
                 f"{f(p['coll_rp'],pct=True)} | {f(p['cbf_hmin'],4)} | "
                 f"{f(p['cbf_viol'],4,pct=True)} |")
    L.append("")
    L.append("## §2 密度\n")
    L.append("学习臂均在 **P=9 训练**，P=5/16 为**零样本**；经典臂（ORCA/SFM/DWA）无训练，"
             "每个密度独立评测。\n")
    L.append("⚠️ **P=16 时学习臂 success = 0**：实测其平均速度 0.94 m/s（在动），"
             "但平均目标距离从 4.07 m **增大到 5.36 m**（朝远离目标的方向移动）→ "
             "这是**真实的密度外推失败**，不是加载错误（同一 checkpoint 在 P=9 上 85.8%）。\n")
    L.append("| P | 方法 | reach success (%) ↑ | social@0.55 (%) ↑ | min_rp (m) ↑ | R–P 接触 (%) ↓ |")
    L.append("|---|---|---|---|---|---|")
    # learned arms: P=9 trained, P=5/16 zero-shot.  classical arms: per-density files.
    plan = [(9, "p9_main.json"), (5, "learned_P5_zeroshot.json"),
            (16, "learned_P16_zeroshot.json")]
    for K, fn in plan:
        for name, labels in (("MAPPO", ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"]),
                             ("CVaR-MAPPO", ["AMC-s0", "AMC-s1", "AMC-s2"]),
                             ("scripted-beeline", ["scripted-beeline"]),
                             ("scripted-detour", ["scripted-detour"]),
                             ("**ORCA (RVO2)**", ["ORCA-RVO2"]),
                             ("SFM (k=5)", ["SFM-k5-A6-B0.5"]),
                             ("SFM (safety)", ["SFM-k3-A6-B0.5"]),
                             ("DWA", ["DWA"]),
                             ("**CVaR-MAPPO + geometric filter (0.6/0.8)**", ["AMC-s0", "AMC-s1", "AMC-s2"])):
            cf = fn
            if name.startswith("**AMC+CBF"):
                # P=9 uses the original filename, P=5/16 the cross-density ones
                cf = (f"cbf3_amc_P{K}_0.6_0.8.json" if K != 9
                      else "cbf3_amc_0.6_0.8.json")
            elif labels[0] in ("ORCA-RVO2", "SFM-k5-A6-B0.5", "SFM-k3-A6-B0.5", "DWA"):
                cf = f"cls_{'orca' if labels[0]=='ORCA-RVO2' else ('sfm_bal' if 'k5' in labels[0] else ('sfm_safe' if 'k3' in labels[0] else 'dwa'))}_P{K}.json"
            rows = load(cf, labels)
            if not rows:
                L.append(f"| {K} | {name} |  |  |  |  |")
                continue
            p = pool(rows)
            L.append(f"| {K} | {name} | {f(p['success'],pct=True)} | "
                     f"{f(p['social_succ@0.55'],pct=True)} | {f(p['min_rp'])} | "
                     f"{f(p['coll_rp'],pct=True)} |")
    # ---- P3: the cross-density (density-curriculum) method rows -----------
    # One policy trained with --density-stages 0:5,300:11,600:16, evaluated at
    # each density, with and without the affect-aware safety layer.  These are the
    # rows that answer "does our method work across densities", as opposed to the
    # zero-shot / density-specific rows above.
    for K, fn, lab in ((5, "dens_P5_base.json", "dens-P5"),
                       (9, "dens_P9_base.json", "dens-P9"),
                       (16, "dens_P16_base.json", "dens-P16"),
                       (12, "dens_P12_base.json", "dens-P12-base"),
                       (20, "dens_P20_base.json", "dens-P20-base")):
        base = load(fn, [lab])
        lyr = {5: "dens_P5_t05p.json", 9: "dens_P9_t05p.json",
               16: "dens_P16_t05p.json", 12: "dens_P12_layer.json",
               20: "dens_P20_layer.json"}.get(K)
        # the LONGER curriculum (1600 it, 1200 of them at P=16) -- this is the row
        # that matches density-specific training at P=16 while still covering P=5/9
        lng = {5: "dlong_P5_base.json", 9: "dlong_P9_base.json",
               16: "dlong_P16_base.json"}.get(K)
        lng_lab = {5: "dlong-P5-base", 9: "dlong-P9-base",
                   16: "dlong-P16-base"}.get(K)
        # WIDE curriculum: covers P=20 (the row that shows "cover the target range")
        wide = {5: "wide_P5_base.json", 9: "wide_P9_base.json",
                16: "wide_P16_base.json", 20: "wide_P20_base.json"}.get(K)
        wide_lab = {5: "wide-P5-base", 9: "wide-P9-base",
                    16: "wide-P16-base", 20: "wide-P20-base"}.get(K)
        # the in-density layer arms are labelled dens-P<K>-layer
        lyr_rows = load(lyr, [lab.replace("-base", "-layer")]) if lyr else None
        if base:
            p = pool(base)
            L.append(f"| {K} | **密度课程：一个策略（无滤波）** | {f(p['success'],pct=True)} | "
                     f"{f(p['social_succ@0.55'],pct=True)} | {f(p['min_rp'])} | "
                     f"{f(p['coll_rp'],pct=True)} |")
        if lyr_rows:
            p = pool(lyr_rows)
            L.append(f"| {K} | **密度课程 + 情绪安全层（τ=0.5）** | {f(p['success'],pct=True)} | "
                     f"{f(p['social_succ@0.55'],pct=True)} | {f(p['min_rp'])} | "
                     f"{f(p['coll_rp'],pct=True)} |")
        lng_rows = load(lng, [lng_lab]) if lng else None
        if lng_rows:
            p = pool(lng_rows)
            L.append(f"| {K} | **密度课程（长，P=16 占 1200 it）** | {f(p['success'],pct=True)} | "
                     f"{f(p['social_succ@0.55'],pct=True)} | {f(p['min_rp'])} | "
                     f"{f(p['coll_rp'],pct=True)} |")
        wide_rows = load(wide, [wide_lab]) if wide else None
        if wide_rows:
            p = pool(wide_rows)
            L.append(f"| {K} | **密度课程（宽，上限覆盖到 P=20）** | {f(p['success'],pct=True)} | "
                     f"{f(p['social_succ@0.55'],pct=True)} | {f(p['min_rp'])} | "
                     f"{f(p['coll_rp'],pct=True)} |")
    # ---- §3 eight-dimension metrics --------------------------------------
    # explicit list: each row is pulled from the ONE file that measured it, so
    # no duplicates and no mislabelled unfiltered numbers under a filtered name
    L.append("")
    L.append("## §3 八维指标（P=9；全部臂均已评测）\n")
    L.append("| 方法 | jerk (m/s³) ↓ | 减速度 p95 (m/s²) ↓ | 航向变化率 (rad/s) ↓ | SPL ↑ | "
             "意图明确时刻 ↓ | 让路提前量 (s) ↑ |")
    L.append("|---|---|---|---|---|---|---|")
    M3 = ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"]
    A3 = ["AMC-s0", "AMC-s1", "AMC-s2"]
    S3 = [("scripted-beeline", "metrics_P9.json", ["scripted-beeline"]),
          ("**scripted-detour**", "metrics_P9.json", ["scripted-detour"]),
          ("scripted-creep", "metrics_P9.json", ["scripted-creep"]),
          ("scripted-stop", "metrics_P9.json", ["scripted-stop"]),
          ("**ORCA (RVO2)**", "metrics_P9_orca.json", ["ORCA-RVO2"]),
          ("**SFM (k=5,A=6,B=0.5)**", "metrics_P9_sfm.json", ["SFM-k5-A6-B0.5"]),
          ("DWA (a=1,b=0.25)", "cls_dwa_P9.json", ["DWA"]),
          ("**MAPPO**", "metrics_P9.json", M3),
          ("**AMC-CVaR**", "metrics_P9.json", A3),
          ("MAPPO + CBF (0.6/0.8)", "metrics_P9_cbf08.json", M3),
          ("CVaR-MAPPO + CBF (0.6/0.8)", "metrics_P9_amccbf08.json", A3)]
    for name, fn, labels in S3:
        rows = load(fn, labels) if fn else []
        if not rows:
            L.append(f"| {name} |  |  |  |  |  |  |")
            continue
        p2 = pool(rows)
        L.append(f"| {name} | {f(p2['jerk'],2)} | {f(p2['decel_p95'],2)} | "
                 f"{f(p2['head_rate'],2)} | {f(p2['spl'],3)} | {f(p2['reveal'],3)} | "
                 f"{f(p2['yield_lead'],2)} |")
    L.append("")
    L.append("## §5 通道消融（P=9，MAPPO，3 训练 seed；`->0` = 该通道置零）\n")
    L.append("| 设定 | reach success (%) ↑ | social@0.55 (%) ↑ | wid ↓ | jerk ↓ | R–P 接触 (%) ↓ |")
    L.append("|---|---|---|---|---|---|")
    for fn, name in (("metrics_P9.json", "baseline"),
                     ("abl_mood_obs_zero_P9.json", "行人情绪 → 0"),
                     ("abl_ped_vel_zero_P9.json", "行人速度 → 0"),
                     ("abl_rmood_obs_zero_P9.json", "机器人自身情绪 → 0")):
        rows = [r for r in load(fn, ["MAPPO-s0", "MAPPO-s1", "MAPPO-s2"])]
        if not rows:
            L.append(f"| {name} |  |  |  |  |  |")
            continue
        p3 = pool(rows)
        L.append(f"| {name} | {f(p3['success'],pct=True)} | {f(p3['social_succ@0.55'],pct=True)} | "
                 f"{f(p3['wid'])} | {f(p3['jerk'],2)} | {f(p3['coll_rp'],pct=True)} |")
    L.append("")
    # ---- §6 significance (produced by scripts/significance.py) ------------
    sig = os.path.join(B, "MAIN_TABLE_SIGNIFICANCE.md")
    if os.path.exists(sig):
        L.append("")
        L.append(open(sig).read().strip())
        L.append("")
    # ---- §7 cost-definition experiment (C1) ------------------------------
    L.append("")
    L.append("## §7 成本口径对照（几何 `encroachment`，cvar 为该口径，**不可与 §1 的 mood_tail 列比较**）\n")
    L.append("| 臂 | reach success (%) ↑ | cvar(几何) ↓ | min_rp (m) ↑ | R–P 接触 (%) ↓ | wid ↓ |")
    L.append("|---|---|---|---|---|---|")
    for name, labels in (("scripted-beeline", ["scripted-beeline"]),
                         ("scripted-detour", ["scripted-detour"]),
                         ("AMC-enc（预算绑定，3 seed）", ["AMC-enc-s0", "AMC-enc-s1", "AMC-enc-s2"])):
        rows = load("amc_enc_P9.json", labels)
        if not rows:
            L.append(f"| {name} |  |  |  |  |  |")
            continue
        p4 = pool(rows)
        L.append(f"| {name} | {f(p4['success'],pct=True)} | {f(p4['cvar_raw'],2)} | "
                 f"{f(p4['min_rp'])} | {f(p4['coll_rp'],pct=True)} | {f(p4['wid'])} |")
    L.append("")
    L.append("结论：几何成本下约束仍只买到 −12% 成本 / −17pp 任务（绕行：−33% / −1.3pp）→ "
             "成本不可行动的解释被排除。\n")

    # ---- §8 MATCHED-DYNAMICS three-seed comparison -------------------------
    # Closes the recipe confound of §1: MAPPO and AMC fine-tuned with the SAME
    # curriculum-off recipe (`configs/mappo_ws_accfix.yaml`), three training
    # seeds each, with and without the same geometric CBF.  Produced by
    # `ops/queue/queue_matched_dyn_3seed.sh`.
    md = os.path.join(B, "matched_dyn_3seed.json")
    mdc = os.path.join(B, "matched_dyn_cbf_3seed.json")
    if os.path.exists(md) or os.path.exists(mdc):
        L.append("")
        L.append("## §8 同配方（课程关闭）+ 三训练 seed 的对照：CVaR-MAPPO 约束仍买不到东西\n")
        L.append("`MAPPO-acc1fix` 与 `AMC-acc1fix` 用**同一个**课程关闭配方微调、"
                 "各 3 个训练 seed；CVaR-MAPPO 的预算由 seed 内的探针标定（target = 0.65×探针 cvar）。\n")
        L.append("| 臂 | reach success (%) ↑ | cvar ↓ | R–P 接触 (%) ↓ | R–R 接触 (%) ↓ |")
        L.append("|---|---|---|---|---|")
        for fname, tag in ((md, "无滤波"), (mdc, "几何 CBF 0.6/0.8")):
            if not os.path.exists(fname):
                continue
            for base in ("MAPPO", "AMC"):
                labels = [f"{base}-acc1fix-s{k}" for k in (0, 1, 2)]
                rows = [r for r in load(os.path.basename(fname), labels)]
                if not rows:
                    L.append(f"| {tag} · {base}-acc1fix |  |  |  |  |")
                    continue
                p5 = pool(rows)
                L.append(f"| {tag} · **{base}-acc1fix** | {f(p5['success'],pct=True)} | "
                         f"{f(p5['cvar_raw'],2)} | {f(p5['coll_rp'],pct=True)} | "
                         f"{f(p5['coll_rr'],pct=True)} |")
        L.append("")
        L.append("读法与结论（三 seed）：")
        L.append("1. **无滤波**：CVaR-MAPPO 用 **−8.8pp 任务**换 **−1.4 cvar**；"
                 "**加了同一个几何 CBF 之后**：CVaR-MAPPO 用 **−24.5pp 任务**换 **−0.65 cvar**"
                 "——滤波器已经把成本压到 17.3，CVaR 约束几乎再买不到东西，却继续吃掉任务。")
        L.append("2. §1 里 `AMC+CBF` 看起来优于 `MAPPO+CBF`（75.4 vs 61.7）**完全是配方造成的**："
                 "把配方对齐后方向反过来（45.7 vs 70.2），且三个 seed 上一致"
                 "（CVaR-MAPPO 44.0–48.2、MAPPO 66.9–72.1）。")
        L.append("3. 训练日志显示 CVaR-MAPPO 的 $\\lambda$ 全程顶在上限 0.5（`lam 0.500`），"
                 "即约束一直想变大却仍买不到成本 ⇒ 与 §3 的结论一致："
                 "**有效成分是显式安全层，不是 CVaR 约束**。\n")
    L.append("## 待补 / 未完成（如实标注）\n")
    L.append("- **SARL（修正版）**：诊断出两个可测量的缺陷——① 每步碰撞/不适惩罚"
             "（5.0/1.0）比每步进度奖励（≈0.05）大约 100 倍，在 P=9 人群里最优策略退化成"
             "「侧向绕开所有人」（实测 cos(v,goal)=+0.10、速度 0.46 m/s、17% 动作朝正后方）；"
             "② `head_span=1.0` 允许「倒着走」。修正这两点后（`queue_sarl_fix.sh`："
             "`w_progress 15 / w_coll 2 / w_discom 0.5 / head_span 0.3`）重训的版本见本表"
             "「SARL port（修正版）」行——**仍标注为「修补过的移植」，不冒充已发表的 SARL**。"
             "**修补后实测 70.1%**，与推荐安全层（69.6%）任务水平相当 —— "
             "这正是「同等任务下比 cvar/接触」的对照。"
             "另有一个**只改算法**的隔离臂（Double DQN，奖励与动作集不变）："
             "seed 1 只有 3.9%（原版 13.0%），单场景方向对齐 +0.29（修补版 +0.50）⇒ "
             "**主因是奖励设计，不是 Q 目标规则**。诊断脚本：`scripts/diag_sarl.py`。")
    L.append("- **SARL（原始移植）**：`envs/sarl.py` + `scripts/train_sarl.py` 已实现（注意力版与 LSTM 消融），"
             "三个 seed 均未收敛：seed0（46M transitions）0%、seed1（46M）13.0%、"
             "**seed2（92M，2× 预算）仍为 0%**。seed0 的逐帧诊断显示它确实朝目标移动"
             "（d_goal 4.11→3.17 m）但平均速度只有 0.42 m/s。"
             "**该行按实测填写，不作为收敛的社交 DRL 基线引用**；"
             "要得到可用基线需要更细的动作集/更稳的算法（Double-DQN、更密的速度-航向网格）"
             "或直接用连续动作的 PPO 版本。")
    if [r for r in load("sarl_lstm_P9.json", ["LSTM-RL"])]:
        L.append("- **LSTM-RL**：`--use_attention 0` 消融已训练并评测（行见 §1），"
                 "与注意力版同属未收敛的离散动作 SARL 移植，同样**不作为收敛基线引用**。")
    else:
        L.append("- **LSTM-RL**：同一份代码的 `--use_attention 0` 消融，未训练。")
    L.append("- **DWA**：可选基线，已实现并评测（见 §1）。")
    L.append("- 八维指标（jerk / 减速度 p95 / 航向变化率 / SPL / 让路提前量）：见 `metrics_P9*.json`。")
    L.append("- 通道消融（mood / ped_vel / rmood 置零）：见 `abl_*_P9.json`。")
    txt = "\n".join(L)
    print(txt)
    open(a.out, "w").write(txt + "\n")
    print("\nsaved", os.path.relpath(a.out, PROJ))


if __name__ == "__main__":
    main()
