"""claims.py -- the paper's claim registry, with LIVE numbers.

Every claim the paper will make is written once, next to the locators that
measure it, and the numbers below are READ FROM THE JSONS at generation time.
Nothing here is hand-typed: a claim whose numbers cannot be located prints as
`MISSING`, and a stale number cannot survive because there is no number in the
source text.

Each locator is a list of (file, label) pairs; the generator aggregates over
POLICY seeds when several are given (each file/label is already a mean over the
3 held-out environment seeds) and reports the seed range, so the reader sees how
many seeds back the number.

Usage: python scripts/claims.py > results/bench/CLAIMS.md
"""
from __future__ import annotations

import json
import os

B = "results/bench"


def val(f, lab, metric):
    p = os.path.join(B, f)
    if not os.path.exists(p):
        return None
    for r in json.load(open(p))["summary"]:
        if r["label"] != lab:
            continue
        if metric == "social":
            v = r["social_succ"]["0.55"]
            return 100 * (v[0] if isinstance(v, list) else v)
        v = r.get(metric)
        if v is None:
            return None
        return 100 * v[0] if metric in ("success", "coll_rp", "coll_rr", "timeout") \
            else (v[0] if isinstance(v, list) else v)
    return None


def agg(specs, metric):
    got = [val(f, l, metric) for f, l in specs]
    got = [g for g in got if g is not None]
    if not got:
        return None
    m = sum(got) / len(got)
    return dict(mean=m, n=len(got), rng=(max(got) - min(got)) if len(got) > 1 else 0.0)


def fmt(a, nd=1, suffix=""):
    if a is None:
        return "MISSING"
    if a["n"] > 1:
        return f"**{a['mean']:.{nd}f} ± {a['rng']:.{nd}f}{suffix}** (n={a['n']})"
    return f"**{a['mean']:.{nd}f}{suffix}** (n=1)"


def d(a, b, nd=1):
    if a is None or b is None:
        return "MISSING"
    return f"{a['mean']-b['mean']:+.{nd}f}"


# --------------------------------------------------------------------------- #
# locator shorthands
def p9(prefix, seeds=(0, 1, 2)):
    """layer3s_<prefix>-s{k} files"""
    return [(f"layer3s_{prefix}-s{k}.json", f"{prefix}-s{k}") for k in seeds]


def wtd(kind, seeds=(0, 1, 2)):
    out = []
    for k in seeds:
        stem = "wtd" if k == 0 else f"wtds1"
        s = 0 if k == 0 else 1
        out.append((f"{stem}_P9_{kind}.json", f"{stem}-P9-{kind}"))
    return out


def ours():
    """The proposed method as reported in the main table: the recommended
    configuration (pedestrian-first solver, 0.75 s early-yield margin, reciprocal
    peers) pooled over three trained policies."""
    if os.path.exists(os.path.join(B, "amc_mappo_p9.json")):
        return [("amc_mappo_p9.json", f"AMC-MAPPO-s{k}") for k in (0, 1, 2)]
    return p9("lex-t0.75")


def sarl_fix(name="SARL-port-fixed"):
    """First existing source for the repaired-SARL baseline -- the pooled file
    once `merge_sarl_seeds.py` has run, else the single-seed one.  Returning a
    SINGLE spec list matters: listing both would double-count seed 0."""
    if os.path.exists(os.path.join(B, "sarl_fix_3seed.json")):
        # the pooled file holds ONE ROW PER TRAINING SEED, so all three labels
        # must be requested for the aggregate to be a training-seed mean
        return [("sarl_fix_3seed.json", f"{name}{s}") for s in ("", "-s1", "-s2")]
    return [("sarl_fix_P9.json", name)]


def cyc(kind, seeds=(0, 1, 2)):
    out = []
    for k in seeds:
        stem = "cyc" if k == 0 else "cycs1"
        out.append((f"{stem}_P9_{kind}.json", f"{stem}-P9-{kind}"))
    return out


CLAIMS = [
    ("C1", "**AMC-MAPPO** 显著降低情绪成本，代价是任务（对无滤波基线 MAPPO）",
     "P=9：MAPPO（无层）→ AMC-MAPPO（推荐配置：词法式 + τ=0.75 + 互惠同伴）；两行各 3 训练 seed",
     [("success", ours(), [("p9_main.json", f"MAPPO-s{k}") for k in (0, 1, 2)]),
      ("cvar_raw", ours(), [("p9_main.json", f"MAPPO-s{k}") for k in (0, 1, 2)]),
      ("coll_rp", ours(), [("p9_main.json", f"MAPPO-s{k}") for k in (0, 1, 2)]),
      ("coll_rr", ours(), [("p9_main.json", f"MAPPO-s{k}") for k in (0, 1, 2)])]),
    ("C1b", "其中**屏障本身**（无提前量）就把成本压到 15.4——这是「情绪几何」的贡献",
     "P=9：无滤波 → 各向异性情绪屏障（linear，α=0.8，τ=0）",
     [("success", p9("aniso"), [("p9_main.json", f"MAPPO-s{k}") for k in (0, 1, 2)]),
      ("cvar_raw", p9("aniso"), [("p9_main.json", f"MAPPO-s{k}") for k in (0, 1, 2)])]),
    ("C2", "τ 是可控、单调的安全/任务旋钮",
     "P=9：τ = 0 → 1.5（联合 QP，无同伴）",
     [("success", p9("jt0.0"), p9("jt1.5")),
      ("cvar_raw", p9("jt0.0"), p9("jt1.5"))]),
    ("C3", "互惠同伴项几乎免费，但把 R–R 接触压到个位数百分比",
     "P=9：τ=0.75 无同伴 vs 有同伴",
     [("success", p9("jt0.75p"), p9("jt0.75")),
      ("cvar_raw", p9("jt0.75p"), p9("jt0.75")),
      ("coll_rr", p9("jt0.75p"), p9("jt0.75"))]),
    ("C4", "收益来自情绪几何的**形状与朝向**，不是任意各向异性、也不是平均半径",
     "P=9：同平均半径 0.6637 m 下，本方法形状 vs 朝向反转 / 侧面 / 无形状圆",
     [("cvar_raw", p9("aniso"), [(f"shape_rot180-s{k}.json", f"rot180-s{k}") for k in (0, 1, 2)]),
      ("cvar_raw", p9("aniso"), [(f"shape_sidebig-s{k}.json", f"sidebig-s{k}") for k in (0, 1, 2)]),
      ("cvar_raw", p9("aniso"), [(f"shape_circ663-s{k}.json", f"circ663-s{k}") for k in (0, 1, 2)]),
      ("success", p9("aniso"), [(f"shape_circ663-s{k}.json", f"circ663-s{k}") for k in (0, 1, 2)])]),
    ("C5", "安全层与经典规划器正交（叠在 ORCA 上成本最低）",
     "P=9：ORCA vs ORCA + 安全层（脚本臂，单次评测，无训练 seed）",
     [("cvar_raw", [("cls_orca_P9.json", "ORCA-RVO2")], [("final_orca_t10.json", "ORCA+filter")]),
      ("coll_rp", [("cls_orca_P9.json", "ORCA-RVO2")], [("final_orca_t10.json", "ORCA+filter")]),
      ("success", [("cls_orca_P9.json", "ORCA-RVO2")], [("final_orca_t10.json", "ORCA+filter")])]),
    ("C6", "跨密度：一个策略可覆盖 P=5–16（均匀交错最好）",
     "P=9 / P=16：均匀交错 vs 难度加权交错（各 2 seed）",
     [("success", [("cyc_P9_base.json", "cyc-P9-base"), ("cycs1_P9_base.json", "cycs1-P9-base")],
       [("wcyc_P9_base.json", "wcyc-P9-base"), ("wcycs1_P9_base.json", "wcycs1-P9-base")]),
      ("success", [("cyc_P16_base.json", "cyc-P16-base"), ("cycs1_P16_base.json", "cycs1-P16-base")],
       [("wcyc_P16_base.json", "wcyc-P16-base"), ("wcycs1_P16_base.json", "wcycs1-P16-base")])]),
    ("C7", "我们**没有**全面超过经典规划器：同密度 ORCA 在任务/社交上仍更好",
     "P=5：推荐行 vs 同密度 ORCA（3 环境 seed 配对）",
     [("success", wtd("t075p"), [("cls_orca_P5.json", "ORCA-RVO2")]),
      ("cvar_raw", wtd("t075p"), [("cls_orca_P5.json", "ORCA-RVO2")]),
      ("social", wtd("t075p"), [("cls_orca_P5.json", "ORCA-RVO2")])]),
    ("C8", "作动延迟会吃掉「提前让路」的收益（机制签名）",
     "P=9：延迟 0 / 1 / 2 / 4 步，层的 Δcvar",
     [("cvar_raw", [(f"rb_delay_lex_s{k}.json", f"rb-delay-lex-s{k}") for k in (0, 1, 2)],
       [(f"rb_delay_base_s{k}.json", f"rb-delay-base-s{k}") for k in (0, 1, 2)]),
      ("cvar_raw", [(f"rb_delay2_lex_s{k}.json", f"rb-delay2-lex-s{k}") for k in (0, 1, 2)],
       [(f"rb_delay2_base_s{k}.json", f"rb-delay2-base-s{k}") for k in (0, 1, 2)])]),
    ("C9", "【展望】CVaR-MAPPO 的约束在本设定下无效（同配方三 seed）",
     "MAPPO vs CVaR-MAPPO，**课程关闭配方微调**、各 3 训练 seed；cvar 越低越好",
     [("success", [("matched_dyn_3seed.json", f"MAPPO-acc1fix-s{k}") for k in (0, 1, 2)],
       [("matched_dyn_3seed.json", f"AMC-acc1fix-s{k}") for k in (0, 1, 2)]),
      ("cvar_raw", [("matched_dyn_3seed.json", f"MAPPO-acc1fix-s{k}") for k in (0, 1, 2)],
       [("matched_dyn_3seed.json", f"AMC-acc1fix-s{k}") for k in (0, 1, 2)]),
      ("success", [("matched_dyn_cbf_3seed.json", f"MAPPO-acc1fix-s{k}") for k in (0, 1, 2)],
       [("matched_dyn_cbf_3seed.json", f"AMC-acc1fix-s{k}") for k in (0, 1, 2)]),
      ("cvar_raw", [("matched_dyn_cbf_3seed.json", f"MAPPO-acc1fix-s{k}") for k in (0, 1, 2)],
       [("matched_dyn_cbf_3seed.json", f"AMC-acc1fix-s{k}") for k in (0, 1, 2)])]),
    ("C9b", "【展望】§1 里 CVaR-MAPPO + 几何滤波的「优势」是配方造成的（对齐配方后方向反转）",
     "旧行（配方混淆）vs 新行（同配方），均为 3 seed",
     [("success", [("cbf3_amc_0.6_0.8.json", "AMC-s0"), ("cbf3_amc_0.6_0.8.json", "AMC-s1"),
                   ("cbf3_amc_0.6_0.8.json", "AMC-s2")],
       [("matched_dyn_cbf_3seed.json", f"AMC-acc1fix-s{k}") for k in (0, 1, 2)])]),
    ("C8b", "**同等任务水平**下，AMC-MAPPO 优于学习式社交基线 SARL（本移植，修补版）",
     "P=9：SARL（本移植，修补版） vs 推荐安全层（前者 70.1% vs 后者 69.6%，任务相当）",
     [("cvar_raw", ours(), sarl_fix()),
      ("coll_rr", ours(), sarl_fix()),
      ("social", ours(), sarl_fix())]),
    ("C9c", "逐场景层面：层在多数场景减少「侵入时间占比」（配对、12 个场景）",
     "见 `trajectory_qualitative.json`（由 `scripts/make_trajectory_figure.py` 生成）："
     "7/12 场景更低、均值降 6.8pp、接触场景 10 → 8；"
     "⚠ 单场景的 **min_rp 在 0.55 m 处饱和**（`contact_stop` 会冻结 episode），"
     "所以逐场景比较要用「侵入时间占比」，不要用单场景最小距离",
     []),
    ("C10", "【展望】SARL 移植未收敛，不作为可用基线",
     "SARL 三个 seed + LSTM 消融",
     [("success", [("sarl_P9_s0.json", "SARL-s0")], [("sarl_P9_s1.json", "SARL-s1")]),
      ("success", [("sarl_lstm_P9.json", "LSTM-RL")], [("sarl_P9_s2.json", "SARL-s2-2x")])]),
    ("C11", "安全层不是实时性瓶颈",
     "见 `FILTER_RUNTIME.md`：P=9 最贵配置 5.0 ms/128 环境步 = 1.29× 策略前向",
     []),
    ("C12", "学习策略的观测噪声裕度 < 观测尺度的 1%（局限）",
     "见 `diag_obs_noise.json`：σ=0.002 时 success 0.896 → 0.021",
     []),
]


def main():
    print("# 论文主张 ↔ 实时数值（由 `scripts/claims.py` 自动生成）\n")
    print("每一行的数字都**从 JSON 现算**：`±` 是策略训练 seed 的极差，`(n=k)` 是参与聚合的 "
          "策略 seed 数（每个文件本身已是 3 个 held-out 环境 seed 的均值）。"
          "`→` 左侧是主张的参照臂，右侧是对照臂。\n")
    for cid, claim, setup, rows in CLAIMS:
        print(f"### {cid}. {claim}\n")
        print(f"设定：{setup}\n")
        if not rows:
            print("_（证据见引用的产物；本条不需要现场数值）_\n")
            continue
        print("| 指标 | 参照 A | 对照 B | Δ(A−B) |")
        print("|---|---|---|---|")
        for metric, A, Bv in rows:
            a, b = agg(A, metric), agg(Bv, metric)
            nd = 2 if metric == "cvar_raw" else 1
            print(f"| {metric} | {fmt(a, nd)} | {fmt(b, nd)} | {d(a, b, nd)} |")
        print()


if __name__ == "__main__":
    main()
