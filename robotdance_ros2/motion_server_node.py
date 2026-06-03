"""ROS2 motion server ノード（rclpy, ROS2 Jazzy 想定, v0）。

certified RD-Motion を読み、MotionServer + SafetyGuard を通して安全フレームを ROS2 に配信する。
RViz で可視化できる（sim-first 再生）。実機 bridge は安全レビュー後に別途接続する。

publish:
  /robotdance/skeleton  visualization_msgs/MarkerArray  （bone の LINE_LIST）
  /robotdance/safety    std_msgs/String                 （SafetyState の JSON）
subscribe:
  /robotdance/estop     std_msgs/Bool                   （True で緊急停止）
"""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from robotdance_core.rd_motion import RdMotion
from robotdance_core.skeleton import BONES

from .messages import MotionFrame, SafetyState
from .motion_server import MotionServer
from .safety_guard import SafetyGuard


def frame_to_marker_array(frame: MotionFrame, *, robot_name: str, frame_id: str = "map") -> MarkerArray:
    """MotionFrame を skeleton の MarkerArray（bone LINE_LIST）に変換する。"""
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = robot_name
    m.id = 0
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = 0.02
    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
    for child, parent in BONES:
        for j in (child, parent):
            p = Point()
            p.x, p.y, p.z = (float(v) for v in frame.keypoints[j])
            m.points.append(p)
    return MarkerArray(markers=[m])


class MotionServerNode(Node):
    """RD-Motion を安全ゲート越しに ROS2 へ配信するノード。"""

    def __init__(self, motion: RdMotion, *, guard: SafetyGuard | None = None) -> None:
        super().__init__("robotdance_motion_server")
        self.server = MotionServer(motion, guard=guard)
        self.robot_name = motion.robot_name
        self._skel_pub = self.create_publisher(MarkerArray, "/robotdance/skeleton", 10)
        self._safety_pub = self.create_publisher(String, "/robotdance/safety", 10)
        self.create_subscription(Bool, "/robotdance/estop", self._on_estop, 10)

        pre = self.server.precheck()
        self._publish_safety(pre)
        if pre.is_abort:
            self.get_logger().error(f"再生中止: {pre.reasons}")
            self._iter = iter(())
        else:
            self.get_logger().info(f"{self.robot_name} 再生開始（certificate PASS）")
            self._iter = self.server.stream()

        period = 1.0 / motion.fps / max(self.server.guard.speed_scale, 1e-3)
        self.create_timer(period, self._tick)

    def _on_estop(self, msg: Bool) -> None:
        if msg.data:
            self.server.guard.estop()
            self.get_logger().warn("E-stop 受信 → 停止")

    def _publish_safety(self, state: SafetyState) -> None:
        self._safety_pub.publish(String(data=json.dumps(
            {"status": state.status.value, "speed_scale": state.speed_scale,
             "reasons": state.reasons}, ensure_ascii=False)))

    def _tick(self) -> None:
        try:
            frame, state = next(self._iter)
        except StopIteration:
            self.get_logger().info("再生完了")
            raise SystemExit(0) from None
        self._skel_pub.publish(frame_to_marker_array(frame, robot_name=self.robot_name))
        self._publish_safety(state)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="robotdance_motion_server")
    parser.add_argument("rdmotion", type=Path, help="certified .rdmotion JSON")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--allow-uncertified", action="store_true",
                        help="sim_certificate 無しでも再生（危険・sim 専用）")
    args = parser.parse_args(argv)

    motion = RdMotion.load(args.rdmotion)
    from .safety_guard import SafetyLimits

    guard = SafetyGuard(
        SafetyLimits(require_certificate=not args.allow_uncertified), speed_scale=args.speed
    )
    rclpy.init()
    node = MotionServerNode(motion, guard=guard)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
