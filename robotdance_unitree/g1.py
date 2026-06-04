"""Unitree G1 embodiment（v0）。

retarget とビューアを sim なしで動かすための、canonical 19-joint と同一トポロジの G1 形態。
rest pose は **公式 g1_23dof URDF の実寸**（Unitree unitree_ros / g1_description）から導いた
canonical joint 位置を採用（v0.26 で更新）。関節オフセット＝寸法の事実のみ使用し、mesh/URDF 本体は
同梱しない（license-safe）。実 actuator 写像・joint limits・慣性は actuator-space IK（retarget-ik /
import-urdf）が実 URDF から扱う。

⚠️ v0 注意: rest 寸法は実機相当だが、これは kinematic 形態であり実機慣性/アクチュエータモデルではない。
（旧 v0 手書きプロキシは nominal 1.12m・bone 平均相対誤差 ~26% で実機と乖離していた。）
"""

from __future__ import annotations

import numpy as np

from robotdance_retarget.embodiment import RobotMorphology, SimDefaults

ROBOT_NAME = "unitree_g1"

# 実 g1_23dof URDF 由来の canonical 19-joint rest pose（z-up, x-forward, y-left, m）。
# 接地に合わせ足先が z≈0.03 になるよう全体を +0.809 m シフト。nominal_height ≈ 1.291 m。
G1_REST = np.array(
    [
        [0.000, 0.000, 0.809],   # 0 pelvis
        [0.000, 0.000, 0.955],   # 1 spine
        [0.000, 0.000, 1.101],   # 2 chest
        [0.015, 0.000, 1.211],   # 3 neck
        [0.030, 0.000, 1.321],   # 4 head
        [0.000, 0.100, 1.101],   # 5 left_shoulder
        [0.016, 0.147, 0.914],   # 6 left_elbow
        [0.116, 0.149, 0.904],   # 7 left_wrist
        [0.000, -0.100, 1.101],  # 8 right_shoulder
        [0.016, -0.147, 0.914],  # 9 right_elbow
        [0.116, -0.149, 0.904],  # 10 right_wrist
        [0.000, 0.064, 0.706],   # 11 left_hip
        [0.000, 0.119, 0.370],   # 12 left_knee
        [0.000, 0.119, 0.070],   # 13 left_ankle
        [0.120, 0.119, 0.030],   # 14 left_foot
        [0.000, -0.064, 0.706],  # 15 right_hip
        [0.000, -0.119, 0.370],  # 16 right_knee
        [0.000, -0.119, 0.070],  # 17 right_ankle
        [0.120, -0.119, 0.030],  # 18 right_foot
    ],
    dtype=np.float64,
)

# 実 g1_23dof URDF 由来の canonical 関節 limit（位置 rad / 速度 rad·s⁻¹ / トルク N·m）。
# 1 canonical ball joint に複数 DOF が対応するため envelope 集約（位置=最広レンジ、速度/トルク=min）。
# 数値のみ（mesh/URDF 非同梱, license-safe）。test_real_g1_urdf が実 URDF と一致を検証。
# 膝は屈曲のみ（逆屈不可）、足首は狭レンジ、トルクは膝139/腕25 と関節ごとに大きく異なる。
G1_JOINT_LIMITS: dict[str, dict[str, object]] = {
    "left_hip": {"position": [-2.7576, 2.9671], "velocity": 32.0, "torque": 88.0},
    "right_hip": {"position": [-2.9671, 2.8798], "velocity": 32.0, "torque": 88.0},
    "left_knee": {"position": [-0.0873, 2.8798], "velocity": 20.0, "torque": 139.0},
    "right_knee": {"position": [-0.0873, 2.8798], "velocity": 20.0, "torque": 139.0},
    "left_ankle": {"position": [-0.8727, 0.5236], "velocity": 30.0, "torque": 35.0},
    "right_ankle": {"position": [-0.8727, 0.5236], "velocity": 30.0, "torque": 35.0},
    "left_shoulder": {"position": [-3.0892, 2.6704], "velocity": 37.0, "torque": 25.0},
    "right_shoulder": {"position": [-3.0892, 2.6704], "velocity": 37.0, "torque": 25.0},
    "left_elbow": {"position": [-1.0472, 2.0944], "velocity": 37.0, "torque": 25.0},
    "right_elbow": {"position": [-1.0472, 2.0944], "velocity": 37.0, "torque": 25.0},
    "left_wrist": {"position": [-1.9722, 1.9722], "velocity": 37.0, "torque": 25.0},
    "right_wrist": {"position": [-1.9722, 1.9722], "velocity": 37.0, "torque": 25.0},
    "spine": {"position": [-2.618, 2.618], "velocity": 32.0, "torque": 88.0},
}

# 実 g1_23dof URDF の <inertial> 由来の canonical 質量分布（Σ≈1, 数値のみで license-safe）。
# 各 link 質量を世界 COM 最近傍の canonical bone へ割当て・左右対称化。実機は股/膝アクチュエータで
# 脚が重く（脚~53%, 胴体~29%）、Winter 人体プライア（胴体~58%/脚~32%）とは別物。
# test_real_g1_urdf が実 URDF からの算出値と一致を検証。
G1_MASS_FRACTION: dict[str, float] = {
    "pelvis": 0.001, "spine": 0.0071, "chest": 0.2499, "neck": 0.001, "head": 0.0302,
    "left_shoulder": 0.001, "left_elbow": 0.0611, "left_wrist": 0.0279,
    "right_shoulder": 0.001, "right_elbow": 0.0611, "right_wrist": 0.0279,
    "left_hip": 0.0556, "left_knee": 0.1334, "left_ankle": 0.0585, "left_foot": 0.0177,
    "right_hip": 0.0556, "right_knee": 0.1334, "right_ankle": 0.0585, "right_foot": 0.0177,
}

# 実 g1_23dof URDF の <inertial> 由来の canonical bone 慣性（質量 kg / COM[3]=親 joint 相対 m /
# fullinertia[6]=COM まわり世界軸 ixx,iyy,izz,ixy,ixz,iyz）。各 link を最近傍 canonical bone へ
# 割当て・剛体合成（平行軸）。capsule 近似（軸対称・棒）と違い実機の三軸非対称（胴体は太い箱で
# 慣性 2-5 倍）を反映。数値のみで license-safe。test_real_g1_urdf が実 URDF と一致を検証。
#
# ⚠️ opt-in: 既定 MORPHOLOGY には付けない。real 慣性は物理的に正しいが、PPO tracking baseline
#   （capsule 慣性で調整済み）を不安定化させるため、controller 再チューニングが済むまで既定は capsule。
#   real 慣性 sim は `build_mjcf(morph, mass_fraction=None)` で morph.inertia_tensors を設定するか、
#   `urdf_to_morphology`（URDF-import）経由で使える。
G1_INERTIA_TENSORS: dict[str, dict] = {
    "pelvis": {"mass": 3.816, "com": [0.0, 0.0, -0.07599], "fullinertia": [0.010567, 0.009327, 0.007919, 0.0, 2e-06, 0.0]},
    "spine": {"mass": 0.244, "com": [0.00396, 0.0, 0.01877], "fullinertia": [0.0001, 0.000124, 0.000156, -2e-06, -1.3e-05, -0.0]},
    "chest": {"mass": 8.562, "com": [-0.00136, 0.00025, 0.06183], "fullinertia": [0.065675, 0.053535, 0.030808, -8.6e-05, -0.001737, 8.7e-05]},
    "head": {"mass": 1.036, "com": [-0.00973, 0.00029, 0.04809], "fullinertia": [0.004085, 0.004185, 0.001808, -3e-06, -6.9e-05, -4e-06]},
    "left_elbow": {"mass": 2.095, "com": [0.00371, 0.04283, -0.08309], "fullinertia": [0.013687, 0.013854, 0.001352, -9e-06, 0.000906, 0.000298]},
    "left_wrist": {"mass": 0.95693, "com": [0.11829, 0.00408, -0.00929], "fullinertia": [0.000486, 0.006995, 0.006938, 9.2e-05, -8.9e-05, -3e-06]},
    "right_elbow": {"mass": 2.095, "com": [0.00371, -0.04283, -0.08309], "fullinertia": [0.013687, 0.013854, 0.001352, 9e-06, 0.000906, -0.000298]},
    "right_wrist": {"mass": 0.95693, "com": [0.11829, -0.00408, -0.00929], "fullinertia": [0.000486, 0.006995, 0.006938, -9.2e-05, -8.9e-05, 3e-06]},
    "left_knee": {"mass": 4.572, "com": [0.02147, 0.04632, -0.15912], "fullinertia": [0.075349, 0.077239, 0.006669, -0.000559, -0.001848, -0.002532]},
    "left_ankle": {"mass": 2.006, "com": [0.00499, 0.00381, -0.12694], "fullinertia": [0.013353, 0.013322, 0.001542, 4.5e-05, -0.0002, -0.000763]},
    "left_foot": {"mass": 0.608, "com": [0.0265, 0.0, -0.03398], "fullinertia": [0.000223, 0.001616, 0.001667, 0.0, 8.9e-05, -0.0]},
    "right_knee": {"mass": 4.572, "com": [0.02147, -0.04632, -0.15912], "fullinertia": [0.075349, 0.077239, 0.006669, 0.000559, -0.001848, 0.002532]},
    "right_ankle": {"mass": 2.006, "com": [0.00499, -0.00381, -0.12694], "fullinertia": [0.013353, 0.013322, 0.001542, -4.5e-05, -0.00011, 0.000763]},
    "right_foot": {"mass": 0.608, "com": [0.0265, -0.0, -0.03398], "fullinertia": [0.000223, 0.001616, 0.001667, -0.0, 8.9e-05, 0.0]},
}

MORPHOLOGY = RobotMorphology(
    name=ROBOT_NAME,
    rest_pose=G1_REST,
    urdf_ref="unitree_ros g1_description/g1_23dof.urdf（実寸由来, 本体は別途取得）",
    runtime_adapter="unitree_sdk2",
    per_joint_limits=G1_JOINT_LIMITS,
    mass_distribution=G1_MASS_FRACTION,
    # inertia_tensors は opt-in（上記 ⚠️ 参照）。既定は capsule 慣性で controller baseline を安定維持。
    # G1（1.29m）の関節 PD で実寸を支える既定。total_mass は実 g1_23dof URDF 総質量（34.13kg）。
    # kd=6 で安定（H1 ほどの慣性は無い）。
    sim_defaults=SimDefaults(total_mass=34.13, kp=150.0, kd=6.0, torque_limit=80.0),
)

# 後方互換のモジュールレベル別名。
BONE_LENGTHS = MORPHOLOGY.bone_lengths
NOMINAL_HEIGHT = MORPHOLOGY.nominal_height


def embodiment_dict() -> dict:
    """RD-Embodiment v0 schema 適合の dict を返す。"""
    return MORPHOLOGY.to_rd_embodiment()
