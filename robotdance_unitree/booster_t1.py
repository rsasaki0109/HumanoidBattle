"""Booster T1 embodiment（v0）。

canonical 19-joint と同一トポロジの Booster T1 形態。rest pose / 関節 limit / 質量分布は
**公式 Booster T1 URDF（BoosterRobotics booster_gym, Apache-2.0）の実値**から導いた数値のみを
採用（mesh/URDF 本体は同梱しない, license-safe）。T1 は G1/H1 より小型軽量（~0.98m / 31.6kg）で、
canonical skeleton への写像で機種非依存に汎用性を実証する 3 機種目。

⚠️ v0 注意:
- T1 は torso DOF が Waist 1 つ（pelvis/spine/chest の 3 canonical 関節へ写像、spine は中点・virtual）。
- 慣性テンソル（real_inertia）は未収載（follow-up）。real_inertia=True でも T1 は capsule にフォールバック。
  geometry / 位置・速度・トルク limit / 質量分布は実 URDF 値。

出典: BoosterRobotics/booster_gym resources/T1（T1_serial.urdf, Apache-2.0）。数値定数のみ抽出・attribution。
"""

from __future__ import annotations

import numpy as np

from robotdance_retarget.embodiment import RobotMorphology, SimDefaults

ROBOT_NAME = "booster_t1"

# 実 T1 URDF 由来の canonical 19-joint rest pose（z-up, x-forward, y-left, m）。
# pelvis=Waist body / chest=肩の高さの中線 / 足=接地 box の toe（前方）。pelvis を原点・足先 z≈0.03 へ平行移動。
T1_REST = np.array(
    [
        [0.000, 0.000, 0.588],   # 0 pelvis
        [-0.003, 0.000, 0.755],  # 1 spine（Waist DOF, pelvis-chest 中点）
        [-0.005, 0.000, 0.922],  # 2 chest
        [0.000, 0.000, 0.946],   # 3 neck
        [0.000, 0.000, 1.008],   # 4 head
        [-0.005, 0.106, 0.922],  # 5 left_shoulder
        [-0.005, 0.214, 0.922],  # 6 left_elbow
        [-0.005, 0.361, 0.922],  # 7 left_wrist
        [-0.005, -0.106, 0.922], # 8 right_shoulder
        [-0.005, -0.214, 0.922], # 9 right_elbow
        [-0.005, -0.361, 0.922], # 10 right_wrist
        [0.000, 0.106, 0.588],   # 11 left_hip
        [-0.014, 0.106, 0.352],  # 12 left_knee
        [-0.014, 0.106, 0.072],  # 13 left_ankle
        [0.107, 0.106, 0.030],   # 14 left_foot
        [0.000, -0.106, 0.588],  # 15 right_hip
        [-0.014, -0.106, 0.352], # 16 right_knee
        [-0.014, -0.106, 0.072], # 17 right_ankle
        [0.107, -0.106, 0.030],  # 18 right_foot
    ],
    dtype=np.float64,
)

# 実 T1 URDF 由来の canonical 関節 limit（位置 rad / 速度 rad·s⁻¹ / トルク[effort] N·m）。
# 1 canonical ball joint に複数 DOF が対応 → envelope 集約（位置=最広, 速度/トルク=min, G1 と同流儀）。
# 膝は屈曲のみ [0, 2.34]、足首は狭レンジ、トルクは膝60/股45/腕18 と関節ごとに異なる（実 effort）。
T1_JOINT_LIMITS: dict[str, dict[str, object]] = {
    "spine": {"position": [-1.5700, 1.5700], "velocity": 10.88, "torque": 30.0},
    "left_shoulder": {"position": [-3.3100, 1.5700], "velocity": 18.84, "torque": 18.0},
    "left_elbow": {"position": [-2.2700, 2.2700], "velocity": 18.84, "torque": 18.0},
    "left_wrist": {"position": [-2.4400, 0.0000], "velocity": 18.84, "torque": 18.0},
    "right_shoulder": {"position": [-3.3100, 1.7400], "velocity": 18.84, "torque": 18.0},
    "right_elbow": {"position": [-2.2700, 2.2700], "velocity": 18.84, "torque": 18.0},
    "right_wrist": {"position": [0.0000, 2.4400], "velocity": 18.84, "torque": 18.0},
    "left_hip": {"position": [-1.8000, 1.5700], "velocity": 10.9, "torque": 30.0},
    "left_knee": {"position": [0.0000, 2.3400], "velocity": 11.7, "torque": 60.0},
    "left_ankle": {"position": [-0.8700, 0.4400], "velocity": 12.4, "torque": 15.0},
    "right_hip": {"position": [-1.8000, 1.5700], "velocity": 10.9, "torque": 30.0},
    "right_knee": {"position": [0.0000, 2.3400], "velocity": 11.7, "torque": 60.0},
    "right_ankle": {"position": [-0.8700, 0.4400], "velocity": 12.4, "torque": 15.0},
}

# 実 T1 URDF の <inertial> mass 由来の canonical 質量分布（Σ=1, 数値のみで license-safe）。
# 各 link を kinematic セグメントで canonical bone へ割当て・左右対称化。胴体（chest=Trunk 11.7kg）が
# 最重量で 37%、脚（股+膝）が ~35%。総質量 31.614kg。
T1_MASS_FRACTION: dict[str, float] = {
    "pelvis": 0.0816, "spine": 0.0010, "chest": 0.3697, "neck": 0.0140, "head": 0.0199,
    "left_shoulder": 0.0218, "left_elbow": 0.0322, "left_wrist": 0.0103,
    "right_shoulder": 0.0218, "right_elbow": 0.0322, "right_wrist": 0.0103,
    "left_hip": 0.1129, "left_knee": 0.0556, "left_ankle": 0.0023, "left_foot": 0.0216,
    "right_hip": 0.1129, "right_knee": 0.0556, "right_ankle": 0.0023, "right_foot": 0.0216,
}

MORPHOLOGY = RobotMorphology(
    name=ROBOT_NAME,
    rest_pose=T1_REST,
    urdf_ref="BoosterRobotics booster_gym resources/T1/T1_serial.urdf（実寸由来, Apache-2.0, 本体は別途取得）",
    runtime_adapter="booster_sdk",
    per_joint_limits=T1_JOINT_LIMITS,
    mass_distribution=T1_MASS_FRACTION,
    # inertia_tensors は未収載（follow-up）。real_inertia でも capsule にフォールバック。
    # T1 は小型（0.98m/31.6kg）だが短い bone で実効慣性が小さく、PD 保持には G1（kp=150）より高い
    # kp=300 が要る（kp≤200 は転倒, 実測スイープ）。kd=6 で安定。torque_limit は膝相当の 60（fallback 用）。
    sim_defaults=SimDefaults(total_mass=31.614, kp=300.0, kd=6.0, torque_limit=60.0),
)

# 後方互換のモジュールレベル別名。
BONE_LENGTHS = MORPHOLOGY.bone_lengths
NOMINAL_HEIGHT = MORPHOLOGY.nominal_height


def embodiment_dict() -> dict:
    """RD-Embodiment v0 schema 適合の dict を返す。"""
    return MORPHOLOGY.to_rd_embodiment()
