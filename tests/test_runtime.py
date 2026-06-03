"""ROS2 runtime コア（safety guard / motion server）の検証。

コアは ROS2 非依存なので rclpy なしでテストできる。rclpy ノードは importorskip でスモーク。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotdance_core.rd_motion import RdMotion
from robotdance_core.skeleton import NUM_JOINTS
from robotdance_core.synthetic import generate_dance
from robotdance_retarget.kinematic import retarget
from robotdance_ros2.messages import MotionFrame, SafetyStatus
from robotdance_ros2.motion_server import MotionServer
from robotdance_ros2.safety_guard import SafetyGuard, SafetyLimits
from robotdance_unitree import get_morphology


def _certified(passed: bool) -> RdMotion:
    motion = retarget(generate_dance(duration=1.0), get_morphology("unitree_g1"))
    motion.sim_certificate = {"passed": passed, "verdict": "PASS" if passed else "REJECT",
                              "reasons": [] if passed else ["airborne"]}
    return motion


# --- safety guard: certificate gate ---

def test_guard_blocks_missing_certificate() -> None:
    motion = retarget(generate_dance(duration=1.0), get_morphology("unitree_g1"))
    assert motion.sim_certificate is None
    state = SafetyGuard().check_certificate(motion)
    assert state.status is SafetyStatus.ABORT


def test_guard_blocks_rejected_certificate() -> None:
    assert SafetyGuard().check_certificate(_certified(False)).is_abort


def test_guard_passes_certified() -> None:
    assert SafetyGuard().check_certificate(_certified(True)).status is SafetyStatus.OK


def test_guard_allow_uncertified_when_configured() -> None:
    motion = retarget(generate_dance(duration=1.0), get_morphology("unitree_g1"))
    guard = SafetyGuard(SafetyLimits(require_certificate=False))
    assert guard.check_certificate(motion).status is SafetyStatus.OK


def test_estop_aborts() -> None:
    guard = SafetyGuard()
    guard.estop()
    assert guard.check_certificate(_certified(True)).is_abort


# --- safety guard: frame filtering ---

def _frame(i: int, z: float = 0.7, x: float = 0.0) -> MotionFrame:
    kp = np.zeros((NUM_JOINTS, 3))
    kp[:, 0] = x
    kp[:, 2] = z
    return MotionFrame(index=i, time=i * 0.033, keypoints=kp, base_position=np.array([0.0, 0.0, z]))


def test_velocity_clamp() -> None:
    guard = SafetyGuard(SafetyLimits(max_link_speed=6.0))
    prev = _frame(0, x=0.0)
    target = _frame(1, x=10.0)  # 10m を 0.033s → 約 300 m/s（過大）
    safe, state = guard.filter_frame(target, prev, dt=0.033)
    peak = np.linalg.norm(safe.keypoints - prev.keypoints, axis=1).max() / 0.033
    assert peak <= 6.0 + 1e-6
    assert state.status is SafetyStatus.WARNING


def test_fall_detection_aborts() -> None:
    guard = SafetyGuard(SafetyLimits(max_base_tilt_drop=0.35))
    guard.filter_frame(_frame(0, z=0.7), None, dt=0.033)        # nominal z=0.7
    _, state = guard.filter_frame(_frame(1, z=0.3), _frame(0, z=0.7), dt=0.033)  # 0.3 < 0.7*0.65
    assert state.is_abort


# --- motion server ---

def test_server_streams_certified() -> None:
    server = MotionServer(_certified(True))
    frames = server.export_frames()
    assert len(frames) == 30  # 1s @ 30fps
    assert all(f.keypoints.shape == (NUM_JOINTS, 3) for f, _ in frames)


def test_server_blocks_uncertified() -> None:
    motion = retarget(generate_dance(duration=1.0), get_morphology("unitree_g1"))
    assert MotionServer(motion).export_frames() == []


def test_speed_scale_clamped() -> None:
    guard = SafetyGuard(speed_scale=5.0)
    assert guard.speed_scale == 1.0
    guard.set_speed_scale(-1.0)
    assert guard.speed_scale == 0.0


# --- ROS2 node smoke ---

def test_ros2_node_publishes() -> None:
    rclpy = pytest.importorskip("rclpy")
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray

    from robotdance_ros2.motion_server_node import MotionServerNode

    motion = _certified(True)
    motion.keypoints_3d = motion.keypoints_3d[:10]
    motion.duration = 10 / motion.fps

    rclpy.init()
    try:
        node = MotionServerNode(motion)
        listener = rclpy.node.Node("test_listener")
        got = {"skel": 0}
        listener.create_subscription(
            MarkerArray, "/robotdance/skeleton", lambda m: got.__setitem__("skel", got["skel"] + 1), 10)
        listener.create_subscription(String, "/robotdance/safety", lambda m: None, 10)
        exe = SingleThreadedExecutor()
        exe.add_node(node)
        exe.add_node(listener)
        for _ in range(40):
            try:
                exe.spin_once(timeout_sec=0.05)
            except SystemExit:
                break
        assert got["skel"] >= 5
        node.destroy_node()
        listener.destroy_node()
    finally:
        rclpy.shutdown()
