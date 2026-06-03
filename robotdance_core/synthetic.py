"""合成モーション生成（pose モデル不要のデモ用）。

簡易 forward kinematics で「ダンス風」の canonical 3D モーションを手続き的に生成し、RD-MIR を返す。
v0.1 のパイプライン/ビューアを、外部モデルや権利付き動画なしで end-to-end に検証するための種データ。
生成物は合成データなので privacy_flags.synthetic=true / license_state=redistributable。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .rd_mir import RdMir, Skeleton
from .skeleton import FOOT_JOINTS, JOINT_NAMES, NUM_JOINTS, PARENTS, index_of

# 立ち姿（rest pose）の world 座標 [J, 3]（z-up, x-forward, y-left, 単位 m）。
# 腕は自然に下げた状態。肩を x 軸回りに回すと腕が横〜頭上へ上がる。
_REST = np.array(
    [
        [0.00, 0.00, 0.95],   # 0 pelvis
        [0.00, 0.00, 1.05],   # 1 spine
        [0.00, 0.00, 1.25],   # 2 chest
        [0.00, 0.00, 1.45],   # 3 neck
        [0.00, 0.00, 1.58],   # 4 head
        [0.00, 0.18, 1.42],   # 5 left_shoulder
        [0.00, 0.20, 1.16],   # 6 left_elbow
        [0.00, 0.21, 0.92],   # 7 left_wrist
        [0.00, -0.18, 1.42],  # 8 right_shoulder
        [0.00, -0.20, 1.16],  # 9 right_elbow
        [0.00, -0.21, 0.92],  # 10 right_wrist
        [0.00, 0.10, 0.92],   # 11 left_hip
        [0.00, 0.10, 0.52],   # 12 left_knee
        [0.00, 0.10, 0.10],   # 13 left_ankle
        [0.15, 0.10, 0.06],   # 14 left_foot
        [0.00, -0.10, 0.92],  # 15 right_hip
        [0.00, -0.10, 0.52],  # 16 right_knee
        [0.00, -0.10, 0.10],  # 17 right_ankle
        [0.15, -0.10, 0.06],  # 18 right_foot
    ],
    dtype=np.float64,
)

# 親→子の bone ベクトル（rest）。root は使わない。
_OFFSET = _REST - _REST[np.array([max(p, 0) for p in PARENTS])]


def _fk(local_rot: list[Rot], root_pos: np.ndarray) -> np.ndarray:
    """各 joint の local 回転と root 位置から world 位置 [J, 3] を求める。"""
    world_rot: list[Rot] = [Rot.identity()] * NUM_JOINTS
    pos = np.zeros((NUM_JOINTS, 3))
    for j, parent in enumerate(PARENTS):
        if parent < 0:
            world_rot[j] = local_rot[j]
            pos[j] = root_pos
        else:
            world_rot[j] = world_rot[parent] * local_rot[j]
            pos[j] = pos[parent] + world_rot[parent].apply(_OFFSET[j])
    return pos


def generate_dance(
    *,
    duration: float = 4.0,
    fps: float = 30.0,
    beats_per_second: float = 1.0,
    motion_id: str = "rdmir-synth-dance-0001",
) -> RdMir:
    """ダンス風の合成 RD-MIR を生成する。"""
    n_frames = round(fps * duration)
    t = np.arange(n_frames) / fps
    phase = 2.0 * np.pi * beats_per_second * t

    i_spine, i_chest = index_of("spine"), index_of("chest")
    i_lsh, i_rsh = index_of("left_shoulder"), index_of("right_shoulder")
    i_lel, i_rel = index_of("left_elbow"), index_of("right_elbow")
    i_lhip, i_rhip = index_of("left_hip"), index_of("right_hip")
    i_lknee, i_rknee = index_of("left_knee"), index_of("right_knee")

    keypoints = np.zeros((n_frames, NUM_JOINTS, 3))
    for f in range(n_frames):
        ph = phase[f]
        local = [Rot.identity() for _ in range(NUM_JOINTS)]

        # 体幹: 左右の sway（forward 軸 x 回り）と軽い yaw（z 回り）。
        sway = 0.18 * np.sin(ph)
        local[i_spine] = Rot.from_euler("x", sway)
        local[i_chest] = Rot.from_euler("xz", [sway, 0.10 * np.sin(ph)])

        # 腕: 左右交互に横〜頭上へ。肩を x 軸回りに回すと下向きの腕が上がる。
        local[i_lsh] = Rot.from_euler("x", 1.6 * (0.5 + 0.5 * np.sin(ph)))
        local[i_rsh] = Rot.from_euler("x", -1.6 * (0.5 + 0.5 * np.sin(ph + np.pi)))
        local[i_lel] = Rot.from_euler("y", -0.5 * (0.5 + 0.5 * np.sin(ph)))
        local[i_rel] = Rot.from_euler("y", -0.5 * (0.5 + 0.5 * np.sin(ph + np.pi)))

        # 脚: 交互に膝を曲げて足踏み（hip/knee を y 軸回りに）。
        lift_l = 0.5 + 0.5 * np.sin(ph)
        lift_r = 0.5 + 0.5 * np.sin(ph + np.pi)
        local[i_lhip] = Rot.from_euler("y", -0.5 * lift_l)
        local[i_lknee] = Rot.from_euler("y", 0.9 * lift_l)
        local[i_rhip] = Rot.from_euler("y", -0.5 * lift_r)
        local[i_rknee] = Rot.from_euler("y", 0.9 * lift_r)

        # root: 上下のバウンスと微小な左右移動。
        root_pos = _REST[0] + np.array([0.0, 0.04 * np.sin(ph), 0.04 * np.abs(np.sin(ph))])
        keypoints[f] = _fk(local, root_pos)

    # 接地: ankle の高さが閾値以下なら接地とみなす。
    ankle_thresh = 0.16
    contacts: dict[str, list[bool]] = {}
    for side, (ankle_idx, _toe_idx) in FOOT_JOINTS.items():
        z = keypoints[:, ankle_idx, 2]
        contacts[f"{side}_foot"] = (z < ankle_thresh).tolist()

    root_traj = {"position": keypoints[:, 0, :].tolist()}

    return RdMir(
        motion_id=motion_id,
        source_ref={"dataset_name": "robotdance-synthetic", "generator": "synthetic.generate_dance"},
        license_state="redistributable",
        fps=fps,
        duration=duration,
        skeleton=Skeleton(joint_names=JOINT_NAMES, parents=PARENTS),
        root_trajectory=root_traj,
        keypoints_3d=keypoints.tolist(),
        contacts=contacts,
        privacy_flags={"synthetic": True, "face_visible": False},
        semantics={"action_label": "dance", "style_tag": "synthetic_demo"},
        extractor_versions={"generator": "robotdance.synthetic.v0"},
    )
