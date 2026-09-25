"""affect_layer_table.py -- the candidate main-method table (route A).

Collects the affect-aware safety-layer rows measured on 09-22 into one table:
the barrier variants (geometric / anisotropic-emotion / +q-gain / +mood margin),
the reciprocal peer-robot barrier, and the same filters on top of the CVaR
policy, all at P=9 with 3 environment seeds against the shared scripted envelope.

Usage:  python scripts/affect_layer_table.py > results/bench/AFFECT_LAYER_TABLE.md
"""
from __future__ import annotations

import json
import os

ROWS = [
    # (file, label, human description)
    ("emocbf_mappo_geo06.json", "mappo_geo06",
     "MAPPO + 几何滤波 0.6（主表参考行）"),
    ("emocbf_mappo_geo09.json", "mappo_geo09",
     "MAPPO + 几何滤波 **0.9**（半径对齐对照）"),
    ("emocbf_mappo_aniso.json", "mappo_aniso",
     "MAPPO + **各向异性情绪个人空间**"),
    ("emocbf_mappo_emo03.json", "mappo_emo03",
     "MAPPO + 各向异性 + distress 余量 0.3"),
    ("qcbf_mappo_q01.json", "mappo_q01", "MAPPO + 各向异性 + q 余量 0.1"),
    ("qcbf_mappo_q03.json", "mappo_q03", "MAPPO + 各向异性 + q 余量 0.3"),
    ("qcbf_mappo_q10.json", "mappo_q10", "MAPPO + 各向异性 + q 余量 1.0"),
    ("peercbf_mappo_aniso.json", "mappo_aniso",
     "（同伴组的基线：各向异性 + distress 余量 0.3）"),
    ("peercbf_mappo_peer062f.json", "mappo_peer062f",
     "MAPPO + 各向异性 + 互惠同伴 0.62（前向门控）"),
    ("peercbf_mappo_peer070f.json", "mappo_peer070f",
     "MAPPO + 各向异性 + 互惠同伴 0.70（前向门控）"),
    ("peercbf_mappo_peer062all.json", "mappo_peer062all",
     "MAPPO + 各向异性 + 互惠同伴 0.62（**全向**）"),
    ("peercbf_mappo_peer070all.json", "mappo_peer070all",
     "**MAPPO + 各向异性 + 互惠同伴 0.70（全向）**"),
    ("emocbf_amc_geo06.json", "amc_geo06", "CVaR-MAPPO + 几何滤波 0.6"),
    ("emocbf_amc_aniso.json", "amc_aniso", "CVaR-MAPPO + 各向异性情绪空间"),
    ("peercbf_amc_peer070f.json", "amc_peer070f",
     "CVaR-MAPPO + 各向异性 + 互惠同伴 0.70（前向门控）"),
    ("qcbf_amc_q03.json", "amc_q03", "CVaR-MAPPO + 各向异性 + q 余量 0.3"),
    # ---- P0: joint QP solve + explicit early-yield margin (2026-09-22 late) ----
    ("joint_jnt_a08.json", "jnt_a08",
     "MAPPO + 各向异性 + **联合 QP**，τ=0（精确最小修正）"),
    ("ttc_t05.json", "t05", "MAPPO + 联合 QP + **提前量 τ=0.5 s**"),
    ("ttc_t10.json", "t10", "MAPPO + 联合 QP + **提前量 τ=1.0 s**"),
    ("ttc_t15.json", "t15", "MAPPO + 联合 QP + **提前量 τ=1.5 s**"),
    ("ttc_t10p.json", "t10p",
     "**MAPPO + 联合 QP + τ=1.0 + 互惠同伴（全向）**"),
    ("final_cyc_t10.json", "cyc-tau1.0",
     "对照：**循环投影** + 同样 τ=1.0（会塌进 stand-still 退化解）"),
    ("final_amc_t10.json", "amc-joint-tau1.0",
     "CVaR-MAPPO 策略 + 联合 QP + τ=1.0"),
    ("final_amc_t10p.json", "amc-joint-tau1.0-peer",
     "CVaR-MAPPO 策略 + 联合 QP + τ=1.0 + 互惠同伴"),
    ("final_orca_t10.json", "ORCA+filter",
     "**ORCA + 联合 QP + τ=1.0**（滤波器与规划器正交）"),
    # ---- P1: tau knee + the third magnitude formulation (negative) ----
    ("tau_t075.json", "t075", "MAPPO + 联合 QP + 提前量 τ=0.75 s"),
    ("tau_t125.json", "t125", "MAPPO + 联合 QP + 提前量 τ=1.25 s"),
    ("tau_t075p.json", "t075p", "MAPPO + 联合 QP + τ=0.75 + 互惠同伴"),
    ("tau_t125p.json", "t125p", "MAPPO + 联合 QP + τ=1.25 + 互惠同伴"),
    ("pred_k05.json", "k05", "MAPPO + 联合 QP + **预测式情绪余量 κ=5**"),
    ("pred_k10.json", "k10", "MAPPO + 联合 QP + 预测式情绪余量 κ=10"),
    ("pred_k20.json", "k20", "MAPPO + 联合 QP + 预测式情绪余量 κ=20"),
    # ---- P2: density-adaptive tau ----
    ("adapt_p9_a05.json", "p9_a05", "MAPPO + 联合 QP + **密度自适应 τ（a=0.5）**，P=9"),
    ("adapt_p9_a10.json", "p9_a10", "MAPPO + 联合 QP + 密度自适应 τ（a=1.0），P=9"),
    ("geo_matched_0.70.json", "geo-0.70", "MAPPO + **几何圆 d=0.70（真正半径对齐）**"),
    ("geo_matched_0.65.json", "geo-0.65", "MAPPO + 几何圆 d=0.65"),
    ("geo_matched_0.75.json", "geo-0.75", "MAPPO + 几何圆 d=0.75"),
    ("lex_p9_joint.json", "p9_joint", "MAPPO + 联合 QP + τ=1.0 + **同伴**（P=9）"),
    ("lex_p9_lex.json", "p9_lex", "MAPPO + **词法式** QP + τ=1.0 + 同伴（P=9）"),
]


def main():
    print("# 情绪感知安全层（P=9，3 个环境种子 1000/1001/1002）\n")
    print("> ⚠ **这些行是单个训练 seed（`runs/ws_p9_mappo_s0`）。**")
    print("> 头条行（各向异性屏障 / τ 扫描 / 互惠同伴 / 词法式 QP /")
    print("> 严格半径对齐对照）已经补到 **3 个训练 seed**，见 `LAYER_3SEED.md`——")
    print("> 引用数字时**优先用那一份**；本表保留作完整变体清单（含负面结果与")
    print("> CVaR-MAPPO 策略上的叠加），其中带 `±` 的仅为环境 seed 波动。\n")
    print("脚本包络（同文件、同种子）：`scripted-beeline` 0.891 / cvar 29.20 / "
          "social@0.55 25.3%；`scripted-detour` 0.878 / **20.17** / **41.7%**；"
          "ORCA-RVO2 0.839 / 19.18 / 51.6%（R–R 接触 0.5%）。\n")
    print("| 方法 | reach success (%) ↑ | cvar ↓ | social@0.55 (%) ↑ | "
          "R–P 接触 (%) ↓ | **R–R 接触 (%) ↓** | min_rp (m) ↑ | nav (s) ↓ |")
    print("|---|---|---|---|---|---|---|---|")
    for f, label, desc in ROWS:
        p = os.path.join("results/bench", f)
        if not os.path.exists(p):
            print(f"| {desc} | | | | | | | |")
            continue
        d = json.load(open(p))
        s = next((r for r in d["summary"] if r["label"] == label), None)
        if s is None:
            print(f"| {desc} | | | | | | | |")
            continue
        soc = s["social_succ"]["0.55"]
        print(f"| {desc} | {100*s['success'][0]:.1f} | {s['cvar_raw'][0]:.2f} | "
              f"{100*soc[0]:.1f} | {100*s['coll_rp'][0]:.1f} | "
              f"**{100*s['coll_rr'][0]:.1f}** | {s['min_rp'][0]:.3f} | "
              f"{s['nav_s'][0]:.2f} |")
    # ---- density section -------------------------------------------------
    print("\n## 密度（零样本，除非标注 in-P training）\n")
    print("| P | 配置 | success (%) ↑ | cvar ↓ | R–P 接触 (%) ↓ | R–R 接触 (%) ↓ | social@0.55 (%) ↑ | social@0.8 (%) ↑ |")
    print("|---|---|---|---|---|---|---|---|")
    DENS = [
        (5, "learned_P5_zeroshot.json", "MAPPO-s0", "无滤波（P=9 训练零样本）"),
        (5, "final_P5_t10.json", "mappo-joint-tau1.0-P5", "安全层 τ=1.0"),
        (9, "scene_base.json", "MAPPO-P9", "无滤波"),
        (9, "ttc_t10.json", "t10", "安全层 τ=1.0"),
        (9, "ttc_t10p.json", "t10p", "安全层 τ=1.0 + 互惠同伴"),
        (16, "learned_P16_zeroshot.json", "MAPPO-s0", "无滤波（零样本）"),
        (16, "final_P16_t10.json", "mappo-joint-tau1.0-P16", "安全层 τ=1.0（零样本）"),
        (16, "p16_base.json", "MAPPO-P16", "无滤波（**在 P=16 训练**）"),
        (16, "p16tau_t025.json", "t025", "安全层 τ=0.25 + 同伴（在 P=16 训练）"),
        (16, "p16tau_t050.json", "t050", "安全层 τ=0.5 + 同伴（在 P=16 训练）"),
        # ---- P3: density curricula (one policy across densities) ----
        (5, "dlong_P5_base.json", "dlong-P5-base", "**长课程**（P=16 占 1200 it）"),
        (9, "dlong_P9_base.json", "dlong-P9-base", "**长课程**"),
        (16, "dlong_P16_base.json", "dlong-P16-base", "**长课程**"),
        (5, "wide_P5_base.json", "wide-P5-base", "**宽课程**（上限覆盖到 P=20）"),
        (9, "wide_P9_base.json", "wide-P9-base", "**宽课程**"),
        (16, "wide_P16_base.json", "wide-P16-base", "**宽课程**"),
        (20, "wide_P20_base.json", "wide-P20-base", "**宽课程**"),
        (5, "wide_P5_t05p.json", "wide-P5-t05p", "宽课程 + 安全层 τ=0.75"),
        (9, "wide_P9_t05p.json", "wide-P9-t05p", "宽课程 + 安全层 τ=0.75"),
        (16, "wide_P16_t05p.json", "wide-P16-t05p", "宽课程 + 安全层 τ=0.75"),
        (20, "wide_P20_t05p.json", "wide-P20-t05p", "宽课程 + 安全层 τ=0.75"),
    ]
    for P, f, lab, desc in DENS:
        p2 = os.path.join("results/bench", f)
        if not os.path.exists(p2):
            print(f"| {P} | {desc} | | | | | | |")
            continue
        d = json.load(open(p2))
        row = next((r for r in d["summary"] if r["label"] == lab), None)
        if row is None:
            print(f"| {P} | {desc} | | | | | | |")
            continue
        ss = row["social_succ"]
        print(f"| {P} | {desc} | {100*row['success'][0]:.1f} | {row['cvar_raw'][0]:.2f} | "
              f"{100*row['coll_rp'][0]:.1f} | {100*row['coll_rr'][0]:.1f} | "
              f"{100*ss['0.55'][0]:.1f} | {100*ss['0.8'][0]:.1f} |")

    # ---- 8-dimension comfort/legibility table -----------------------------
    print("\n## 八维指标（P=9，3 环境种子；pareto_eval 同一 rollout 采集）\n")
    print("| 方法 | jerk (m/s³) ↓ | 减速度 p95 (m/s²) ↓ | 航向变化率 (rad/s) ↓ | SPL ↑ | 意图明确时刻 ↓ | 让路提前量 (s) ↑ |")
    print("|---|---|---|---|---|---|---|")
    EIGHT = [("scene_base.json", "MAPPO-P9", "无滤波 MAPPO"),
             ("ttc_t10.json", "t10", "安全层 τ=1.0"),
             ("ttc_t10p.json", "t10p", "安全层 τ=1.0 + 互惠同伴"),
             ("joint_cyc_a08.json", "cyc_a08", "对照：循环投影 α=0.8"),
             ("final_orca_t10.json", "ORCA+filter", "ORCA + 安全层")]
    for f, lab, desc in EIGHT:
        p2 = os.path.join("results/bench", f)
        if not os.path.exists(p2):
            print(f"| {desc} | | | | | | |"); continue
        d = json.load(open(p2))
        row = next((r for r in d["summary"] if r["label"] == lab), None)
        if row is None:
            print(f"| {desc} | | | | | | |"); continue
        g = lambda k: row[k][0] if isinstance(row[k], list) else row[k]
        print(f"| {desc} | {g('jerk'):.3f} | {g('decel_p95'):.3f} | {g('head_rate'):.3f} | "
              f"{g('spl'):.3f} | {g('reveal'):.3f} | {g('yield_lead'):.3f} |")

    print("\n读法（P0 之后的最终版）：\n"
          "1. **各向异性情绪个人空间**在半径对齐的对照上同时更优"
          "（任务 +8.6pp、cvar −0.47）——情绪模型在这里提供的是**信息**，不是保守度；\n"
          "2. **联合 QP 是「能用大余量」的前提**：同样的约束与同样的提前量 τ=1.0，"
          "循环投影塌进 stand-still（任务 15.9%），联合求解保持 67.2%；\n"
          "3. **提前量 τ 给出可控的 Pareto 曲线**（τ=0.5/1.0/1.5 → 任务 75.5/67.2/55.2%，"
          "cvar 21.9/18.9/17.9）；τ=1.0 的 cvar 已低于解析绕行（20.17）而任务仍有 67.2%；\n"
          "4. **互惠同伴项只在联合求解下真正生效**：R–R 接触 13.6%（无滤波）→ **2.9%**"
          "（联合+同伴）→ **1.8%**（CVaR-MAPPO 策略）→ **1.6%**（叠在 ORCA 上），接近 ORCA 的 0.5%；"
          "旧循环实现只能到 5.5%；\n"
          "5. 滤波器与规划器**正交**：叠在 ORCA 上把 cvar 19.18→8.93、接触 39.3%→20.8%（任务 0.839→0.583）；\n"
          "6. `q` / `mood` 这些**情绪量级**项只增加保守性，没有 Pareto 收益（§43）；\n"
          "7. 叠在 CVaR-MAPPO 策略上**始终不如**叠在 MAPPO 上——有效成分是显式安全层，不是 CVaR 约束；\n"
          "8. **情绪量级的三类写法（累计 mood / 瞬时 q / 预测情绪损失）全部只增加保守性**，"
          "同等任务成本下 cvar 与 social 都不如固定 τ——有效的情绪信息是**几何**（各向异性个人空间），不是量级。")


if __name__ == "__main__":
    main()
