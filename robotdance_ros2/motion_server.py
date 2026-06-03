"""Motion Server コア（ROS2 非依存, v0）。

RD-Motion(.rdmotion) を MotionFrame の系列に展開し、SafetyGuard を通して安全なフレームを
逐次供給する。speed scaling / pause / phase 制御を持つ。ROS2 ノードはこのコアを駆動するだけ。
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from robotdance_core.rd_motion import RdMotion
from robotdance_core.skeleton import index_of

from .messages import MotionFrame, SafetyState, SafetyStatus
from .safety_guard import SafetyGuard


class MotionServer:
    """certified RD-Motion を安全に再生するサーバ（Mode B: motion playback）。"""

    def __init__(self, motion: RdMotion, guard: SafetyGuard | None = None) -> None:
        self.motion = motion
        self.guard = guard or SafetyGuard()
        self._kps = motion.keypoints_3d_array()  # [T, J, 3]
        self._pelvis = index_of("pelvis")
        self.paused = False
        # アクチュエータ関節角（actuator-space IK の出力があれば）。
        jr = motion.joint_rotations or {}
        self._joint_names: list[str] = list(jr.get("actuated_joint_names", []))
        self._joint_angles = (
            np.asarray(jr["angles_rad"], dtype=np.float64) if jr.get("angles_rad") else None
        )

    def _frame_at(self, i: int) -> MotionFrame:
        cs = self.motion.contact_schedule or {}
        contacts = {
            k: bool(np.asarray(v, dtype=bool)[i]) if i < len(v) else False
            for k, v in cs.items()
        }
        return MotionFrame(
            index=i,
            time=i / self.motion.fps,
            keypoints=self._kps[i],
            base_position=self._kps[i, self._pelvis],
            contacts=contacts,
            phase=i / max(self._kps.shape[0] - 1, 1),
            joint_names=self._joint_names,
            joint_angles=self._joint_angles[i] if self._joint_angles is not None else None,
        )

    def precheck(self) -> SafetyState:
        """再生前の certificate ゲート。"""
        return self.guard.check_certificate(self.motion)

    def stream(self) -> Iterator[tuple[MotionFrame, SafetyState]]:
        """安全フレームを逐次 yield する。ABORT が出たら停止する。

        speed_scale により実時間 dt は変わるが、フレーム列自体は等間隔（time は元クリップ基準）。
        pause 中は最後のフレームを保持して yield し続ける（呼び出し側が pause を解除するまで）。
        """
        pre = self.precheck()
        if pre.is_abort:
            return
        n = self._kps.shape[0]
        base_dt = 1.0 / self.motion.fps
        prev_safe: MotionFrame | None = None
        for i in range(n):
            target = self._frame_at(i)
            dt = base_dt / max(self.guard.speed_scale, 1e-3)
            safe, state = self.guard.filter_frame(target, prev_safe, dt)
            yield safe, state
            if state.status is SafetyStatus.ABORT:
                return
            prev_safe = safe

    def export_frames(self) -> list[tuple[MotionFrame, SafetyState]]:
        """Mode A: 全フレームを安全整形してリストで返す（offline export / bag 用）。"""
        return list(self.stream())
