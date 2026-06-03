"""RobotDance runtime のメッセージ契約（ROS2 非依存の dataclass, v0）。

設計方針 §5.2 の message に対応する純 Python 表現。ROS2 ノード（motion_server_node）は
これらを sensor/visualization メッセージへマップする。コア（safety_guard / motion_server）は
ROS2 に依存せず、これらの dataclass だけで動く（テスト容易）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SafetyStatus(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    ABORT = "ABORT"


@dataclass
class MotionFrame:
    """time-indexed な robot 目標フレーム。"""

    index: int
    time: float                  # クリップ先頭からの秒
    keypoints: np.ndarray        # [J, 3] robot link 位置
    base_position: np.ndarray    # [3] base/pelvis 位置
    contacts: dict[str, bool] = field(default_factory=dict)
    phase: float = 0.0           # 0..1 の再生位相


@dataclass
class SafetyState:
    """safety guard の出力状態。"""

    status: SafetyStatus
    speed_scale: float
    reasons: list[str] = field(default_factory=list)

    @property
    def is_abort(self) -> bool:
        return self.status is SafetyStatus.ABORT
