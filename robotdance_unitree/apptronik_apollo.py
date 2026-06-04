"""Apptronik Apollo embodiment（v0）。

canonical 19-joint と同一トポロジの Apptronik Apollo 形態（4 機種目, full-size ~1.62m / 80.9kg）。
rest pose / 関節 limit（位置・**実 forcerange トルク**）/ 質量分布 / 慣性テンソルは **MuJoCo Menagerie の
公式 Apollo モデル（Apache-2.0）の実値**から抽出（数値定数のみ・mesh/MJCF 本体は非同梱, license-safe）。

抽出は MuJoCo にモデルを読み込ませ world frame / world 軸慣性を厳密計算（diaginertia+quat の回転を
MuJoCo が処理）→ 各 body を世界 COM 最近傍の canonical bone（区間中点）へ割当て・剛体合成（平行軸）。

⚠️ v0 注意:
- **velocity 限界は未収載**（menagerie MJCF に actuator velocity 情報が無い）→ velocity feasibility 軸は
  generic fallback。位置 ROM / トルク / 質量 / 慣性 / 寸法 / balance の 6 軸は実 Apollo 値。
- runtime adapter は実機 SDK 不明のため sim（mujoco）扱い。

出典: google-deepmind/mujoco_menagerie `apptronik_apollo`（Apache-2.0, Apptronik Apollo 由来）。
"""

from __future__ import annotations

import numpy as np

from robotdance_retarget.embodiment import RobotMorphology, SimDefaults

ROBOT_NAME = "apptronik_apollo"

# 実 Apollo モデル由来の canonical 19-joint rest pose（z-up, x-forward, y-left, m）。MuJoCo qpos0 立位から
# 抽出し pelvis を原点・足先 z≈0.03 へ平行移動。chest=肩の高さ中線、足=接地 toe（前方）。
APOLLO_REST = np.array(
    [
        [0.000, 0.000, 1.089],   # 0 pelvis
        [0.000, 0.000, 1.264],   # 1 spine
        [0.000, 0.000, 1.440],   # 2 chest
        [0.000, 0.000, 1.448],   # 3 neck
        [0.027, 0.000, 1.648],   # 4 head
        [-0.020, 0.200, 1.440],  # 5 left_shoulder
        [0.015, 0.233, 1.125],   # 6 left_elbow
        [-0.024, 0.243, 1.065],  # 7 left_wrist
        [-0.020, -0.200, 1.440], # 8 right_shoulder
        [0.015, -0.233, 1.125],  # 9 right_elbow
        [-0.024, -0.243, 1.065], # 10 right_wrist
        [-0.020, 0.110, 0.920],  # 11 left_hip
        [-0.070, 0.110, 0.495],  # 12 left_knee
        [-0.020, 0.110, 0.070],  # 13 left_ankle
        [0.110, 0.110, 0.030],   # 14 left_foot
        [-0.020, -0.110, 0.920], # 15 right_hip
        [-0.070, -0.110, 0.495], # 16 right_knee
        [-0.020, -0.110, 0.070], # 17 right_ankle
        [0.110, -0.110, 0.030],  # 18 right_foot
    ],
    dtype=np.float64,
)

# 実 Apollo モデル由来の canonical 関節 limit（位置 rad / トルク[forcerange] N·m）。複数 DOF →
# envelope（位置=最広, トルク=min）。**velocity は menagerie MJCF に無く未収載**（follow-up）。
# 膝は屈曲のみ [0, 2.618]・トルク 336（full-size の強脚）、肘は屈曲、足首は狭レンジ。左右の実非対称は保持。
APOLLO_JOINT_LIMITS: dict[str, dict[str, object]] = {
    "spine": {"position": [-0.8290, 1.3526], "torque": 120.0},
    "neck": {"position": [-1.6581, 1.6581], "torque": 10.6},
    "head": {"position": [-0.2618, 0.5236], "torque": 34.2},
    "left_shoulder": {"position": [-2.1817, 1.6057], "torque": 67.0},
    "left_elbow": {"position": [-2.6180, 0.1745], "torque": 114.0},
    "left_wrist": {"position": [-1.6581, 1.6581], "torque": 10.6},
    "right_shoulder": {"position": [-2.1817, 0.6109], "torque": 67.0},
    "right_elbow": {"position": [-2.6180, 0.1745], "torque": 114.0},
    "right_wrist": {"position": [-1.6581, 1.6581], "torque": 10.6},
    "left_hip": {"position": [-1.8500, 1.0908], "torque": 120.0},
    "left_knee": {"position": [0.0000, 2.6180], "torque": 336.0},
    "left_ankle": {"position": [-1.5708, 0.4363], "torque": 120.0},
    "right_hip": {"position": [-1.8500, 0.5672], "torque": 120.0},
    "right_knee": {"position": [0.0000, 2.6180], "torque": 336.0},
    "right_ankle": {"position": [-1.5708, 0.6545], "torque": 120.0},
}

# 実 Apollo モデルの <inertial> mass 由来の canonical 質量分布（Σ=1）。世界 COM 最近傍 bone 割当・左右
# 対称化。胴体（chest 19.3kg）が最重量 24%、脚は大腿(knee) > 下腿(ankle)。総質量 80.898kg。
APOLLO_MASS_FRACTION: dict[str, float] = {
    "pelvis": 0.1073, "spine": 0.0014, "chest": 0.2388, "neck": 0.0010, "head": 0.0310,
    "left_shoulder": 0.0068, "left_elbow": 0.0433, "left_wrist": 0.0203,
    "right_shoulder": 0.0068, "right_elbow": 0.0433, "right_wrist": 0.0203,
    "left_hip": 0.0238, "left_knee": 0.1434, "left_ankle": 0.0565, "left_foot": 0.0161,
    "right_hip": 0.0238, "right_knee": 0.1434, "right_ankle": 0.0565, "right_foot": 0.0161,
}

# 実 Apollo モデルの <inertial>（diaginertia+quat）を MuJoCo で world 軸へ展開し canonical bone へ平行軸
# 合成（質量 kg / COM[3]=親 joint 相対 m / fullinertia[6]=COM まわり世界軸）。
# get_morphology("apptronik_apollo", real_inertia=True) で使用。
APOLLO_INERTIA_TENSORS: dict[str, dict] = {
    "pelvis": {"mass": 8.6881, "com": [-0.04181, -0.00002, -0.05938], "fullinertia": [0.0682946, 0.0417759, 0.0628248, 2.13512e-05, -0.00928448, 9.31719e-06]},
    "chest": {"mass": 19.341, "com": [-0.01850, 0.00143, 0.04116], "fullinertia": [0.272265, 0.303549, 0.220417, -0.000339538, -0.00116578, 0.0018574]},
    "head": {"mass": 2.5075, "com": [0.03700, -0.00045, 0.18946], "fullinertia": [0.0245975, 0.0320922, 0.0124901, 4.63737e-05, -0.00442314, 0.000108088]},
    "left_shoulder": {"mass": 0.54929, "com": [-0.02654, 0.21316, -0.02243], "fullinertia": [0.000633794, 0.000542785, 0.000569715, 0.000121465, 6.55438e-05, -0.000174463]},
    "left_elbow": {"mass": 3.5127, "com": [0.01521, 0.03925, -0.14356], "fullinertia": [0.0236274, 0.0280969, 0.0076553, 0.00129928, -0.000114085, 0.000543343]},
    "left_wrist": {"mass": 1.6421, "com": [-0.03829, 0.01407, -0.07176], "fullinertia": [0.00810231, 0.00820218, 0.00173001, 3.18307e-05, 0.000236653, 0.000612533]},
    "right_shoulder": {"mass": 0.54908, "com": [-0.02654, -0.21228, -0.02244], "fullinertia": [0.000644975, 0.000543961, 0.000582543, -0.000125923, 6.50793e-05, 0.000169564]},
    "right_elbow": {"mass": 3.5036, "com": [0.01487, -0.03925, -0.14298], "fullinertia": [0.0234474, 0.0278661, 0.00764543, -0.00134613, -9.16697e-05, -0.000448066]},
    "right_wrist": {"mass": 1.6428, "com": [-0.03822, -0.01405, -0.07177], "fullinertia": [0.00810127, 0.00820459, 0.00173121, -2.85719e-05, 0.00023962, -0.000619412]},
    "left_hip": {"mass": 1.9296, "com": [-0.04880, 0.10478, -0.16165], "fullinertia": [0.00445457, 0.0040131, 0.00320706, 0.000274089, -0.000732374, 0.000162259]},
    "left_knee": {"mass": 11.609, "com": [0.00605, 0.01169, -0.20106], "fullinertia": [0.141479, 0.13363, 0.0630767, 0.00191769, -0.00632823, -0.0105087]},
    "left_ankle": {"mass": 4.5751, "com": [0.04353, 0.00505, -0.15242], "fullinertia": [0.0634508, 0.0635494, 0.0108498, 0.000445155, -0.00149439, 0.000433275]},
    "left_foot": {"mass": 1.3056, "com": [0.04312, 0.00679, -0.03907], "fullinertia": [0.00374049, 0.00918728, 0.0073152, -0.00041831, 0.00273413, 0.000345913]},
    "right_hip": {"mass": 1.9296, "com": [-0.04878, -0.10478, -0.16164], "fullinertia": [0.00441779, 0.00403057, 0.0032198, -0.000317174, -0.000716724, -0.000169667]},
    "right_knee": {"mass": 11.618, "com": [0.00684, -0.01177, -0.20110], "fullinertia": [0.140877, 0.133957, 0.0630468, -0.00143526, -0.00677352, 0.0102102]},
    "right_ankle": {"mass": 4.5743, "com": [0.04329, -0.00516, -0.15244], "fullinertia": [0.0634341, 0.0635329, 0.0108394, -0.000419543, -0.0014567, -0.000385789]},
    "right_foot": {"mass": 1.3056, "com": [0.04312, -0.00679, -0.03907], "fullinertia": [0.00379247, 0.00908467, 0.00736582, 0.000985563, 0.00263034, -0.000485779]},
}

MORPHOLOGY = RobotMorphology(
    name=ROBOT_NAME,
    rest_pose=APOLLO_REST,
    urdf_ref="mujoco_menagerie apptronik_apollo（Apache-2.0, Apptronik Apollo 由来, 本体は別途取得）",
    runtime_adapter="mujoco",
    per_joint_limits=APOLLO_JOINT_LIMITS,
    mass_distribution=APOLLO_MASS_FRACTION,
    # inertia_tensors は EMBODIMENT_INERTIA registry 経由で real_inertia=True 時に装着（既定 capsule）。
    # Apollo は full-size 重量級（1.62m/80.9kg）。PD 既定は実測スイープで決定。
    sim_defaults=SimDefaults(total_mass=80.898, kp=400.0, kd=12.0, torque_limit=120.0),
)

BONE_LENGTHS = MORPHOLOGY.bone_lengths
NOMINAL_HEIGHT = MORPHOLOGY.nominal_height


def embodiment_dict() -> dict:
    """RD-Embodiment v0 schema 適合の dict を返す。"""
    return MORPHOLOGY.to_rd_embodiment()
