"""benchmark feasibility chart（torque× vs balance violation の散布図）。

各 (motion, robot) run を **torque×（横）× balance violation率（縦）**に打ち、PASS/REJECT で
色分けする。実行可能領域（torque×≤1.0 かつ balance≤閾値）を淡く塗ることで、**動作がなぜ落ちるか
（トルク律速か / バランス律速か）**が一目で分かる。motion 名を注記する。

matplotlib 必須（dev 依存・ヘッドレス Agg）。sim（mujoco）無しで verdict が None の run は描かない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# certificate の REJECT 閾値（balance violation率）。memory/real-video-demo-pipeline と整合。
_BALANCE_THRESH = 0.3
_TORQUE_THRESH = 1.0


def render_benchmark_chart(report: dict, path: str | Path, *,
                           title: str = "RobotDance feasibility") -> Path:
    """benchmark report を feasibility 散布図にして path に書き出す。書き出した Path を返す。"""
    rows = [r for r in report["rows"]
            if r.get("verdict") and r.get("torque_ratio") is not None
            and r.get("balance_violation_ratio") is not None]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=90)
    # 実行可能領域（torque×≤1.0 かつ balance≤閾値）を淡緑で塗る。
    ax.axvspan(0, _TORQUE_THRESH, ymin=0, ymax=1, color="#e8f5e9", zorder=0)
    ax.axhline(_BALANCE_THRESH, color="#bbb", ls="--", lw=0.8, zorder=1)
    ax.axvline(_TORQUE_THRESH, color="#bbb", ls="--", lw=0.8, zorder=1)

    markers = {}
    robots = report.get("robots", sorted({r["robot"] for r in rows}))
    marker_cycle = ["o", "s", "^", "D", "v", "P", "*"]
    for i, rb in enumerate(robots):
        markers[rb] = marker_cycle[i % len(marker_cycle)]

    for r in rows:
        passed = r["verdict"] == "PASS"
        ax.scatter(r["torque_ratio"], r["balance_violation_ratio"],
                   marker=markers.get(r["robot"], "o"), s=70,
                   c=("#2ca02c" if passed else "#d62728"), alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(r["motion_id"], (r["torque_ratio"], r["balance_violation_ratio"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points", color="#444")

    ax.set_xlabel("torque ratio  (× actuator limit;  ≤1.0 feasible)")
    ax.set_ylabel("balance violation ratio  (ZMP outside support)")
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # 凡例: 色（PASS/REJECT）と marker（robot）。
    handles = [plt.Line2D([], [], marker="o", ls="", color="#2ca02c", label="PASS"),
               plt.Line2D([], [], marker="o", ls="", color="#d62728", label="REJECT")]
    handles += [plt.Line2D([], [], marker=markers[rb], ls="", color="#666", label=rb)
                for rb in robots]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _has_plottable(report: dict[str, Any]) -> bool:
    return any(r.get("verdict") and r.get("torque_ratio") is not None for r in report["rows"])
