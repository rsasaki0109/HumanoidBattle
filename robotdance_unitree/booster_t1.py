"""Booster T1 embodiment（v0）。

canonical 19-joint と同一トポロジの Booster T1 形態。rest pose / 関節 limit / 質量分布は
**公式 Booster T1 URDF（BoosterRobotics booster_gym, Apache-2.0）の実値**から導いた数値のみを
採用（mesh/URDF 本体は同梱しない, license-safe）。T1 は G1/H1 より小型軽量（~0.98m / 31.6kg）で、
canonical skeleton への写像で機種非依存に汎用性を実証する 3 機種目。

⚠️ v0 注意:
- T1 は torso DOF が Waist 1 つ（pelvis/spine/chest の 3 canonical 関節へ写像、spine は中点・virtual）。
- geometry / 位置・速度・トルク limit / 質量分布 / 慣性テンソル すべて実 URDF 値（v0.56 で慣性も収載・
  G1/H1 と同格の 7 軸フル実データ）。real_inertia=True で実慣性 sim。

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
# 各 link を**世界 COM 最近傍の canonical bone（区間中点）**へ割当て・左右対称化（v0.56 で是正:
# v0.55 はセグメントを 1 つ取り違え、大腿を hip・下腿を knee に置いていた → 大腿=knee bone, 下腿=
# ankle bone に修正）。胴体（chest=Trunk 11.7kg）が最重量 37%、脚は大腿(knee) > 下腿(ankle)。総 31.614kg。
T1_MASS_FRACTION: dict[str, float] = {
    "pelvis": 0.0813, "spine": 0.0010, "chest": 0.3686, "neck": 0.0010, "head": 0.0339,
    "left_shoulder": 0.0010, "left_elbow": 0.0217, "left_wrist": 0.0424,
    "right_shoulder": 0.0010, "right_elbow": 0.0217, "right_wrist": 0.0424,
    "left_hip": 0.0322, "left_knee": 0.0804, "left_ankle": 0.0554, "left_foot": 0.0239,
    "right_hip": 0.0322, "right_knee": 0.0804, "right_ankle": 0.0554, "right_foot": 0.0239,
}

# 実 T1 URDF の <inertial> 由来の canonical bone 慣性（質量 kg / COM[3]=親 joint 相対 m /
# fullinertia[6]=COM まわり世界軸 ixx,iyy,izz,ixy,ixz,iyz）。割当 link を剛体合成（平行軸）。
# T1 URDF は inertial frame に回転が無い（link 軸=世界軸 at rest）ので回転変換不要、並進のみ。
# 数値のみで license-safe。get_morphology("booster_t1", real_inertia=True) で使用。
T1_INERTIA_TENSORS: dict[str, dict] = {
    "pelvis": {"mass": 2.581, "com": [0.00228, 0.00000, 0.00730], "fullinertia": [0.005289, 0.005299, 0.004821, 0, 0.000207, 1e-06]},
    "chest": {"mass": 11.7, "com": [-0.00486, -0.00000, 0.05331], "fullinertia": [0.0915287, 0.0767787, 0.0556171, -4.2537e-07, 0.00064636, 5.8234e-07]},
    "head": {"mass": 1.0749, "com": [0.00437, 0.00016, 0.11793], "fullinertia": [0.00501808, 0.00494622, 0.00190885, -2.77711e-05, -0.000176153, -3.75647e-05]},
    "left_elbow": {"mass": 0.69, "com": [0.00038, 0.05425, 0.00000], "fullinertia": [0.00183466, 0.000472532, 0.00196719, -3.13146e-05, 0, 0]},
    "left_wrist": {"mass": 1.3472, "com": [-0.00003, 0.12696, 0.00007], "fullinertia": [0.0282994, 0.000836121, 0.0282003, 1.58382e-06, 1.84616e-08, -3.62855e-05]},
    "right_elbow": {"mass": 0.69, "com": [0.00038, -0.05425, 0.00000], "fullinertia": [0.00183466, 0.000472532, 0.00196719, -2.68542e-06, 0, 0]},
    "right_wrist": {"mass": 1.3472, "com": [-0.00003, -0.12696, 0.00007], "fullinertia": [0.0282994, 0.000836121, 0.0282003, -1.58382e-06, 1.84616e-08, 3.62855e-05]},
    "left_hip": {"mass": 1.021, "com": [0.00053, 0.09870, -0.01808], "fullinertia": [0.001805, 0.001421, 0.001292, 6e-06, -1.5e-05, 8e-05]},
    "left_knee": {"mass": 2.551, "com": [-0.00598, 0.00018, -0.17334], "fullinertia": [0.0311219, 0.0319956, 0.00332472, -6.50417e-06, 0.00179147, -4.3022e-05]},
    "left_ankle": {"mass": 1.73, "com": [-0.00601, 0.00026, -0.12432], "fullinertia": [0.034618, 0.034539, 0.001934, 1.1e-05, 0.001561, 0.000197]},
    "left_foot": {"mass": 0.758, "com": [-0.00058, 0.00023, -0.01987], "fullinertia": [0.00223743, 0.00242622, 0.0026968, -5.72782e-08, -0.000107985, 2.17024e-07]},
    "right_hip": {"mass": 1.021, "com": [0.00053, -0.09849, -0.01808], "fullinertia": [0.001805, 0.001421, 0.001292, -8e-06, -1.5e-05, -8.5e-05]},
    "right_knee": {"mass": 2.555, "com": [-0.00594, -0.00012, -0.17339], "fullinertia": [0.0311551, 0.0320286, 0.00332448, 5.53105e-06, 0.00178496, 3.7363e-05]},
    "right_ankle": {"mass": 1.79, "com": [-0.00574, -0.00054, -0.12260], "fullinertia": [0.035098, 0.034958, 0.002039, -9e-06, 0.001554, -8.6e-05]},
    "right_foot": {"mass": 0.758, "com": [-0.00058, -0.00023, -0.01987], "fullinertia": [0.00223743, 0.00242622, 0.0026968, 5.72946e-08, -0.000107984, -2.17024e-07]},
}

MORPHOLOGY = RobotMorphology(
    name=ROBOT_NAME,
    rest_pose=T1_REST,
    urdf_ref="BoosterRobotics booster_gym resources/T1/T1_serial.urdf（実寸由来, Apache-2.0, 本体は別途取得）",
    runtime_adapter="booster_sdk",
    per_joint_limits=T1_JOINT_LIMITS,
    mass_distribution=T1_MASS_FRACTION,
    # inertia_tensors は EMBODIMENT_INERTIA registry 経由で real_inertia=True 時に装着（既定 capsule）。
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
