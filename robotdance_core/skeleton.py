"""RobotDance canonical human skeleton (v0).

RD-MIR の skeleton-first 原則の中核。embodiment 非依存の標準スケルトンを定義する。
pose/HMR adapter はこの canonical skeleton に出力をマップし、retarget はここから各ロボットへ写像する。

座標規約: z-up, x-forward, y-left（right-handed）。単位 m。
"""

from __future__ import annotations

from dataclasses import dataclass

# canonical joint 名（index 順）。v0 は body のみ（hand/face は将来の拡張 spec で追加）。
JOINT_NAMES: list[str] = [
    "pelvis",          # 0  (root)
    "spine",           # 1
    "chest",           # 2
    "neck",            # 3
    "head",            # 4
    "left_shoulder",   # 5
    "left_elbow",      # 6
    "left_wrist",      # 7
    "right_shoulder",  # 8
    "right_elbow",     # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "left_knee",       # 12
    "left_ankle",      # 13
    "left_foot",       # 14 (toe)
    "right_hip",       # 15
    "right_knee",      # 16
    "right_ankle",     # 17
    "right_foot",      # 18 (toe)
]

# 各 joint の親 index（root は -1）。
PARENTS: list[int] = [
    -1,  # pelvis
    0,   # spine
    1,   # chest
    2,   # neck
    3,   # head
    2,   # left_shoulder
    5,   # left_elbow
    6,   # left_wrist
    2,   # right_shoulder
    8,   # right_elbow
    9,   # right_wrist
    0,   # left_hip
    11,  # left_knee
    12,  # left_ankle
    13,  # left_foot
    0,   # right_hip
    15,  # right_knee
    16,  # right_ankle
    17,  # right_foot
]

NUM_JOINTS: int = len(JOINT_NAMES)

# 描画・距離計算に使う bone（親→子）の index ペア。
BONES: list[tuple[int, int]] = [(j, p) for j, p in enumerate(PARENTS) if p >= 0]

_NAME_TO_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

# 接地判定に使う末端 joint。
FOOT_JOINTS = {
    "left": (_NAME_TO_INDEX["left_ankle"], _NAME_TO_INDEX["left_foot"]),
    "right": (_NAME_TO_INDEX["right_ankle"], _NAME_TO_INDEX["right_foot"]),
}


def index_of(name: str) -> int:
    """joint 名から index を返す。"""
    return _NAME_TO_INDEX[name]


@dataclass(frozen=True)
class CanonicalSkeleton:
    """RD-MIR skeleton フィールドに対応する canonical 構造。"""

    joint_names: list[str]
    parents: list[int]

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def bones(self) -> list[tuple[int, int]]:
        return [(j, p) for j, p in enumerate(self.parents) if p >= 0]


CANONICAL_SKELETON = CanonicalSkeleton(joint_names=JOINT_NAMES, parents=PARENTS)
