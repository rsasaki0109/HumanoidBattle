"""robotdance_unitree

G1/H1 configs, URDF mapping, SDK2/ROS2 bridge assumptions — Unitree を primary target とする embodiment 統合。
"""

import dataclasses

from robotdance_retarget.embodiment import RobotMorphology

from . import g1, h1

# robot 名 → 形態 の registry。新しい Unitree 機種はここに追加する。
EMBODIMENTS: dict[str, RobotMorphology] = {
    g1.ROBOT_NAME: g1.MORPHOLOGY,
    h1.ROBOT_NAME: h1.MORPHOLOGY,
}

# robot 名 → 実 URDF <inertial> 由来の per-bone 慣性テンソル（opt-in）。
# 既定 morphology は capsule 近似（inertia_tensors=None）だが、real_inertia=True で実テンソルを装着する。
EMBODIMENT_INERTIA: dict[str, dict] = {
    g1.ROBOT_NAME: g1.G1_INERTIA_TENSORS,
    h1.ROBOT_NAME: h1.H1_INERTIA_TENSORS,
}


def get_morphology(name: str, *, real_inertia: bool = False) -> RobotMorphology:
    """robot 名から RobotMorphology を返す。

    real_inertia: True で実 URDF <inertial> 由来の per-bone 慣性テンソルを装着する（sim が capsule
        近似でなく実機慣性で逆動力学・追従を行う）。既定 False は capsule 慣性。**PD 追従は実慣性でも
        安定**（survival 1.0, RMSE ほぼ不変 — tests/test_sim.py 参照）なので feasibility 検証に安全に
        使える。v0.37 で崩壊したのは PPO 学習ポリシーのみで、その再学習は別タスク。
    """
    if name not in EMBODIMENTS:
        raise KeyError(f"未知の robot: {name}（利用可能: {sorted(EMBODIMENTS)}）")
    morph = EMBODIMENTS[name]
    if real_inertia:
        return dataclasses.replace(morph, inertia_tensors=EMBODIMENT_INERTIA[name])
    return morph


__all__ = ["EMBODIMENTS", "EMBODIMENT_INERTIA", "get_morphology", "g1", "h1"]
