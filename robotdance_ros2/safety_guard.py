"""Safety Guard（§5.6, ROS2 非依存, v0）。

「動画を入れたら即ロボットが踊る」を防ぐ最後の gate。certified でない motion を弾き、
過大な速度/加速度をクランプし、転倒を検知し、E-stop / speed scaling を提供する。

⚠️ v0 は Cartesian（link 位置）空間で動作する。joint 空間の limit clamp は actuator-space
retarget が入る Phase 4+ で追加する。物理 sim（sim_certificate）は別途 robotdance_sim が担う。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotdance_core.rd_motion import RdMotion

from .messages import MotionFrame, SafetyState, SafetyStatus


@dataclass
class SafetyLimits:
    """安全包絡線。"""

    max_link_speed: float = 6.0       # m/s（link 位置の許容速度）
    max_link_accel: float = 120.0     # m/s^2
    warn_link_speed: float = 4.0      # 警告閾値
    max_base_tilt_drop: float = 0.35  # base がこの割合以上沈むと転倒とみなす
    require_certificate: bool = True   # sim_certificate PASS を必須にするか


class SafetyGuard:
    """motion frame を安全に整形する gate。"""

    def __init__(self, limits: SafetyLimits | None = None, *, speed_scale: float = 1.0) -> None:
        self.limits = limits or SafetyLimits()
        self.speed_scale = float(np.clip(speed_scale, 0.0, 1.0))
        self._estopped = False
        self._nominal_base_z: float | None = None

    # --- 制御 ---

    def estop(self) -> None:
        """緊急停止。以降のフレームは ABORT になる。"""
        self._estopped = True

    def reset(self) -> None:
        self._estopped = False
        self._nominal_base_z = None

    def set_speed_scale(self, scale: float) -> None:
        self.speed_scale = float(np.clip(scale, 0.0, 1.0))

    # --- gate ---

    def check_certificate(self, motion: RdMotion) -> SafetyState:
        """再生前チェック: sim_certificate が PASS でなければ ABORT。"""
        if self._estopped:
            return SafetyState(SafetyStatus.ABORT, self.speed_scale, ["E-stop 作動中"])
        cert = motion.sim_certificate
        if not self.limits.require_certificate:
            return SafetyState(SafetyStatus.OK, self.speed_scale, [])
        if cert is None:
            return SafetyState(SafetyStatus.ABORT, self.speed_scale,
                               ["sim_certificate 無し（物理検証されていない）"])
        if not cert.get("passed", False):
            reasons = ["sim_certificate REJECT"] + list(cert.get("reasons", []))
            return SafetyState(SafetyStatus.ABORT, self.speed_scale, reasons)
        return SafetyState(SafetyStatus.OK, self.speed_scale, [])

    def filter_frame(
        self, target: MotionFrame, prev: MotionFrame | None, dt: float
    ) -> tuple[MotionFrame, SafetyState]:
        """1 フレームを安全に整形し、(safe_frame, state) を返す。"""
        if self._estopped:
            held = prev or target
            return held, SafetyState(SafetyStatus.ABORT, self.speed_scale, ["E-stop 作動中"])

        reasons: list[str] = []
        status = SafetyStatus.OK
        kp = target.keypoints.astype(np.float64).copy()

        if prev is not None and dt > 0:
            # 速度クランプ（link ごと）。
            delta = kp - prev.keypoints
            speed = np.linalg.norm(delta, axis=1) / dt  # [J]
            peak = float(speed.max()) if speed.size else 0.0
            if peak > self.limits.max_link_speed:
                scale = self.limits.max_link_speed / peak
                kp = prev.keypoints + delta * scale
                reasons.append(f"link 速度クランプ {peak:.1f}→{self.limits.max_link_speed:.1f} m/s")
                status = SafetyStatus.WARNING
            elif peak > self.limits.warn_link_speed:
                reasons.append(f"link 速度 {peak:.1f} m/s（警告）")
                status = SafetyStatus.WARNING

        # 転倒検知（base が nominal から大きく沈む）。
        bz = float(target.base_position[2])
        if self._nominal_base_z is None:
            self._nominal_base_z = bz
        elif self._nominal_base_z > 0 and bz < self._nominal_base_z * (1 - self.limits.max_base_tilt_drop):
            reasons.append(f"base 沈下 {bz:.2f} < {self._nominal_base_z:.2f}（転倒検知）")
            return (prev or target), SafetyState(SafetyStatus.ABORT, self.speed_scale, reasons)

        safe = MotionFrame(
            index=target.index, time=target.time, keypoints=kp,
            base_position=target.base_position, contacts=target.contacts, phase=target.phase,
        )
        return safe, SafetyState(status, self.speed_scale, reasons)
