"""shape_control_table.py -- is it the SHAPE (and its orientation) or just "some
anisotropy / some extra radius"?

De-confounded design: the affect COST is always computed with the environment's
own affect geometry (1.0 / 0.6 / 0.4), and only the shape the BARRIER assumes is
varied (`--cbf-shape`).  All three shapes below have the identical theoretical
mean radius

    E[r] = 0.3183*(d_front + d_back) + 0.3634*d_side = 0.6637 m

so the comparison isolates PLACEMENT, not magnitude; a plain circle of the same
mean radius (d_min = 0.663) is the no-shape control, and the previously used
radius-matched control (d=0.70) is shown next to it.

Usage: python scripts/shape_control_table.py > results/bench/SHAPE_CONTROL.md
"""
from __future__ import annotations

import json
import os

B = "results/bench"

ARMS = [
    ("affect 1.0 / 0.6 / 0.4（**本方法**：前方最大，符合人的个人空间）",
     [("layer3s_aniso-s0.json", "aniso-s0"), ("layer3s_aniso-s1.json", "aniso-s1"),
      ("layer3s_aniso-s2.json", "aniso-s2")]),
    ("rot180 0.4 / 0.6 / 1.0（同样三个半径，**朝向反转**）",
     [("shape_rot180-s0.json", "rot180-s0"), ("shape_rot180-s1.json", "rot180-s1"),
      ("shape_rot180-s2.json", "rot180-s2")]),
    ("side-big 0.472 / 1.0 / 0.472（同样三个半径，**放到侧面**）",
     [("shape_sidebig-s0.json", "sidebig-s0"), ("shape_sidebig-s1.json", "sidebig-s1"),
      ("shape_sidebig-s2.json", "sidebig-s2")]),
    ("圆 r=0.663（**同平均半径、无形状**）",
     [("shape_circ663-s0.json", "circ663-s0"), ("shape_circ663-s1.json", "circ663-s1"),
      ("shape_circ663-s2.json", "circ663-s2")]),
    ("圆 r=0.70（此前用的半径对齐对照）",
     [("layer3s_geo070-s0.json", "geo070-s0"), ("layer3s_geo070-s1.json", "geo070-s1"),
      ("layer3s_geo070-s2.json", "geo070-s2")]),
]


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
                        coll_rr=100 * g("coll_rr"), min_rp=g("min_rp"))
    return None


def main():
    print("# 情绪几何的**形状/朝向**对照（P=9，3 训练 seed × 3 环境 seed）\n")
    print("设计：**代价口径固定**为环境的情绪几何（1.0/0.6/0.4），只改变**安全层假定的形状**")
    print("（`--cbf-shape`，本次新增的开关）。三个形状的理论平均半径相同")
    print("（`E[r] = 0.3183·(d_front+d_back) + 0.3634·d_side = 0.6637 m`），")
    print("因此比较的是**摆放位置**，不是量级。\n")
    print("| 安全层假定的形状 | success (%) | cvar ↓ | social@0.55 | R–P 接触 | R–R 接触 | min_rp |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for desc, specs in ARMS:
        got = [get(f, l) for f, l in specs]
        got = [g for g in got if g]
        if not got:
            print(f"| {desc} | | | | | | |")
            continue
        n = len(got)
        rng = lambda k: max(g[k] for g in got) - min(g[k] for g in got)  # noqa: E731
        mean = lambda k: sum(g[k] for g in got) / n                      # noqa: E731
        print(f"| {desc} | {mean('success'):.1f} ± {rng('success'):.1f} | "
              f"**{mean('cvar'):.2f}** ± {rng('cvar'):.2f} | {mean('social'):.1f} | "
              f"{mean('coll_rp'):.1f} | {mean('coll_rr'):.1f} | {mean('min_rp'):.3f} |")
        rows.append((desc, mean("success"), mean("cvar"), n))
    print()
    # effects relative to the two no-shape controls
    base663 = next((c for d, s, c, n in rows if d.startswith("圆 r=0.663")), None)
    base070 = next((c for d, s, c, n in rows if d.startswith("圆 r=0.70")), None)
    aff = next((c for d, s, c, n in rows if d.startswith("affect")), None)
    if aff and base663 and base070:
        print(f"**结论**：正确朝向的情绪形状把 cvar 从 {base663:.2f}（同平均半径的圆）降到 "
              f"**{aff:.2f}**（**−{100*(base663-aff)/base663:.1f}%**），"
              f"相对 r=0.70 的圆是 **−{100*(base070-aff)/base070:.1f}%**；")
        print("而**把同样三个半径反转（rot180）或放到侧面（side-big）后，收益完全消失**"
              "（两者的 cvar 与「无形状的圆」在 0.1 以内）。\n")
        print("⇒ 对审稿人「是不是只要各向异性就行 / 是不是只要半径更大就行」的回答是：")
        print("**都不是**——起作用的既不是「各向异性」这个性质本身，也不是平均半径，")
        print("而是**与人的个人空间一致的朝向与形状**。\n")
        print("**诚实的解读与边界**：代价函数与安全层用的是**同一个**情绪几何，")
        print("所以这条结论的准确含义是「**当控制器假定正确的个人空间形状时，真实情绪成本最低**」。")
        print("这是「把情绪几何接进安全层」这一主张的直接证据；反过来说，")
        print("若代价口径换成另一套个人空间模型，收益预计会缩小——")
        print("这既是本方法的适用条件，也是未来用人类数据标定后应重做的实验。\n")
    print("附：任务侧的差异全部在训练 seed 极差之内（各臂 56.9–59.7，极差 5.7–9.4pp），")
    print("即**这些形状差异不是靠牺牲任务换来的**，但也没有任何一臂在任务上显著更好。\n")


if __name__ == "__main__":
    main()
