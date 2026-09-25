"""robustness_table.py -- does the safety layer still help OUTSIDE the training
distribution, and where does the whole stack break?

Cells (all inference-only; the env runs in observation-capacity mode so the
trained P=9 / N=3 policy drops into every cell without retraining):

  clean    P=9, no perturbation
  noise*   observation noise std 1e-6 / 0.02 / 0.05 / 0.1
           (1e-6 is an ARTEFACT CONTROL: `get_obs` draws from the global RNG
            whenever obs_noise != 0, so this cell keeps the RNG-stream shift and
            removes the noise -- if it matches `clean`, the collapse at larger
            values is the noise itself)
  delay*   actions applied 1 / 2 / 4 steps late (0.05 / 0.1 / 0.2 s)
  n4       4 robots instead of the 3 trained
  obst     6 obstacles instead of 4
  beta     affect model misspecified: mood_beta halved

Arms: `base` (no filter) and `lex` (the recommended layer: affect shape +
tau=0.75 + lexicographic QP + reciprocal peers).  Each cell x arm is 3 policy
seeds x 3 held-out environment seeds; `±` is the range over policy seeds.

Usage: python scripts/robustness_table.py > results/bench/ROBUSTNESS.md
"""
from __future__ import annotations

import glob
import json
import os

B = "results/bench"

CELLS = [
    ("clean", "P=9，无扰动", None),
    ("noiseeps", "观测噪声 σ=1e-6（**对照：只保留 RNG 流位移**）", "noiseeps"),
    ("noise02", "观测噪声 σ=0.02", "noise02"),
    ("noise05", "观测噪声 σ=0.05", "noise05"),
    ("noise", "观测噪声 σ=0.10", "noise"),
    ("delay1", "控制延迟 1 步（0.05 s）", "delay1"),
    ("delay2", "控制延迟 2 步（0.10 s）", "delay2"),
    ("delay", "控制延迟 4 步（0.20 s）", "delay"),
    ("n4", "机器人 N=4（训练为 3）", "n4"),
    ("obst", "障碍物 6 个（训练为 4）", "obst"),
    ("beta", "情绪模型错配：mood_beta 减半", "beta"),
]

# clean-cell sources (same layer configuration as the `lex` arm)
CLEAN = {
    "base": [("p9_main.json", f"MAPPO-s{k}") for k in (0, 1, 2)],
    "lex": [(f"layer3s_lex-t0.75-s{k}.json", f"lex-t0.75-s{k}") for k in (0, 1, 2)],
}


def get(f, lab):
    p = os.path.join(B, f)
    if not os.path.exists(p):
        return None
    for r in json.load(open(p))["summary"]:
        if r["label"] == lab:
            g = lambda k: (r[k][0] if isinstance(r[k], list) else r[k])  # noqa: E731
            ss = r["social_succ"]["0.55"]
            ss = ss[0] if isinstance(ss, list) else ss
            return dict(success=100 * g("success"), cvar=g("cvar_raw"),
                        social=100 * ss, coll_rp=100 * g("coll_rp"),
                        coll_rr=100 * g("coll_rr"))
    return None


def cell_rows(axis):
    if axis is None:                     # clean
        out = {}
        for arm, specs in CLEAN.items():
            got = [get(f, l) for f, l in specs]
            out[arm] = [g for g in got if g] or None
        return out
    out = {}
    for arm in ("base", "lex"):
        got = [get(f"rb_{axis}_{arm}_s{k}.json", f"rb-{axis}-{arm}-s{k}") for k in (0, 1, 2)]
        out[arm] = [g for g in got if g] or None
    return out


def ms(rows, k, nd=1):
    if not rows:
        return "—"
    v = [r[k] for r in rows]
    m = sum(v) / len(v)
    return f"{m:.{nd}f} ± {max(v)-min(v):.{nd}f}" if len(v) > 1 else f"{m:.{nd}f}"


def main():
    print("# 安全层的分布外鲁棒性（P=9；每格 = 3 训练 seed × 3 环境 seed）\n")
    print("全部为**纯推理**：环境跑在观测容量模式（robots 4 / peds 24 / obstacles 6），"
          "所以训练好的 P=9、N=3 策略可以直接放进每个格子里，不需要重训。\n")
    print("`base` = 无滤波；`lex` = 推荐安全层（情绪各向异性 + τ=0.75 + 词法式 QP + 互惠同伴）。"
          "`Δ` = 安全层 − 无滤波。\n")
    print("| 扰动 | 臂 | reach success (%) | cvar ↓ | social@0.55 (%) | R–P 接触 | R–R 接触 |")
    print("|---|---|---|---|---|---|---|")
    summary = []
    for _name, desc, axis in CELLS:
        rows = cell_rows(axis)
        for arm in ("base", "lex"):
            r = rows.get(arm)
            print(f"| {desc} | {arm} | {ms(r,'success')} | "
                  f"{ms(r,'cvar',2)} | {ms(r,'social')} | {ms(r,'coll_rp')} | {ms(r,'coll_rr')} |")
        b, l = rows.get("base"), rows.get("lex")
        if b and l:
            ds = sum(x["success"] for x in l) / len(l) - sum(x["success"] for x in b) / len(b)
            dc = sum(x["cvar"] for x in l) / len(l) - sum(x["cvar"] for x in b) / len(b)
            summary.append((desc, ds, dc))
            # every cell must be non-empty: this table is checked by
            # `verify_tables.py`, whose blank-cell rule treats an empty cell as
            # "planned but never measured".
            print(f"| {desc} | **Δ（层 − 无层）** | **{ds:+.1f}** | **{dc:+.2f}** | "
                  f"— | — | — |")
    print()
    print("### 两个必须说明的口径问题\n")
    print("- **N=4 无法用这套 checkpoint 评测**（表中为 `—`）。`pareto_eval` 报 "
          "`size mismatch for vr.net.0.weight: shape [128, 135]`："
          "环境虽在观测容量模式（robots 4 / peds 24 / obstacles 6），"
          "但**只有 actor 的观测维度**被容量钳住（N=3 与 N=4 都是 166），"
          "**critic 的状态维度仍然依赖 N**（`_state_dim` 里 `2*N`（预测通道）与 `N`（机器人情绪）"
          "没有用容量上限）——实测 state_dim：干净/obst/beta/noise/delay 都是 **135**，"
          "N=4 是 **136**。⇒ 机器人数量泛化需要**重训**，不能用评测补。\n")
    print("- **`mood_beta` 那一格的 cvar 不可与其它格比较**：情绪成本本身由该参数定义"
          "（β 减半 ⇒ 同样的侵入产生的 mood 损失减半），所以 cvar 从 26.94 降到 19.14 "
          "**是定义使然，不是行为变好**。该格应只读 **success / social / 接触率**。\n")
    print("- **噪声与外障碍两格的任务崩溃发生在策略侧**：`base` 自己就已经 0% 成功"
          "（见 `diag_obs_noise.py` 的剂量-反应：σ=0.002 就把 success 从 0.90 打到 0.02），"
          "所以这两格**不能**用来评价安全层的抗噪/抗障能力，只能说明"
          "**层仍然把成本与接触压下去**（σ=0.05：cvar −9.4、R–P −22.6pp；障碍 6 个："
          "cvar −8.3、R–P −31.5pp、R–R 33.6% → 9.5%）。\n")
    # ---- dose-response diagnostic for the noise axis -----------------------
    p = os.path.join(B, "diag_obs_noise.json")
    if os.path.exists(p):
        dd = json.load(open(p))
        print("## 观测噪声的剂量-反应（`diag_obs_noise.py`，CPU，16 环境 × 1 episode）\n")
        print("| 噪声 σ | success | 平均速度 (m/s) | cos(速度, 目标方向) | 净接近目标 (m) |")
        print("|---|---|---|---|---|")
        for r in dd:
            print(f"| {r['sigma']:.5f} | {r['success']:.3f} | {r['mean_speed']:.3f} | "
                  f"{r['cos_vel_goal']:+.3f} | {r['progress']:+.3f} |")
        print()
        print("**这张表是本轮最刺眼的发现**：`σ = 1e-6` 与干净基线**完全一致**"
              "（0.896 / cos +0.390）⇒ 崩溃**不是**「开关噪声导致 RNG 流位移」造成的；"
              "而 `σ = 0.002`（约为观测尺度的 0.2%，折算位置约 1 cm）就把 success "
              "从 **0.90 打到 0.02**，速度-目标对齐从 `+0.39` 塌到 `0.09`。"
              "注意机器**仍在动**（0.20 m/s）只是不再朝目标——不是冻住，是**方向被噪声毁掉**。\n")
        print("⇒ 结论：**学习策略对观测噪声的裕度 < 观测尺度的 1%**。"
              "这是**策略**的脆弱性，不是安全层的；"
              "因此「安全层能否提升抗噪性」这个问题在本设定下**无法回答**——"
              "基线策略在任何可测的噪声下都已经失效。论文应把它写成局限，"
              "并把「带噪声训练/观测平滑」列为未来工作。\n")
    print("## 读法\n")
    for desc, ds, dc in summary:
        print(f"- **{desc}**：Δsuccess {ds:+.1f}pp、Δcvar {dc:+.2f}")
    print()
    print("要点：① `noiseeps` 与 `clean` 的差就是「噪声开关带来的 RNG 流位移」本身的大小；")
    print("② 观测噪声只影响学习臂（脚本臂用的是真实状态，噪声对它们无效），"
          "所以这一列同时是「学习策略的感知脆弱性」的度量；")
    print("③ 延迟是最对抗安全层的扰动（它的全部前提是**提前**让路）。\n")


if __name__ == "__main__":
    main()
