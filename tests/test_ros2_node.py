"""ROS2 motion server ノードの pause/seek/estop 配線を rclpy モックで検証する。

rclpy は CI に無いので、ROS2 関連モジュールを sys.modules に偽物として注入してから
motion_server_node を import する。これでノードの subscribe コールバックがコアの
MotionServer.pause/resume/seek_phase / guard.estop を正しく駆動することを CI でも確認できる。
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from robotdance_core.synthetic import generate_dance
from robotdance_retarget.kinematic import retarget
from robotdance_unitree import get_morphology


def _certified():
    motion = retarget(generate_dance(duration=1.0), get_morphology("unitree_g1"))
    motion.sim_certificate = {"passed": True, "verdict": "PASS", "reasons": []}
    return motion


class _FakeNode:
    """rclpy.node.Node の最小スタブ（publisher/subscription/timer/logger/clock を no-op 化）。"""

    def __init__(self, *_a, **_k) -> None:
        self.subs: dict = {}

    def create_publisher(self, *_a, **_k):
        return SimpleNamespace(publish=lambda *a, **k: None)

    def create_subscription(self, _type, topic, cb, _qos):
        self.subs[topic] = cb
        return SimpleNamespace()

    def create_timer(self, *_a, **_k):
        return SimpleNamespace()

    def get_logger(self):
        return SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None,
                               error=lambda *a, **k: None)

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: None))


@pytest.fixture()
def node_mod(monkeypatch):
    """ROS2 依存を偽モジュールで差し替えて motion_server_node を import する。"""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    def msg(*_a, **k):  # 構築可能な汎用 msg スタブ（属性は kwargs から）。
        return SimpleNamespace(**k)

    fakes = {
        "rclpy": mod("rclpy", init=lambda *a, **k: None, shutdown=lambda *a, **k: None,
                     spin=lambda *a, **k: None),
        "rclpy.node": mod("rclpy.node", Node=_FakeNode),
        "geometry_msgs": mod("geometry_msgs"),
        "geometry_msgs.msg": mod("geometry_msgs.msg", Point=msg),
        "sensor_msgs": mod("sensor_msgs"),
        "sensor_msgs.msg": mod("sensor_msgs.msg", JointState=msg),
        "std_msgs": mod("std_msgs"),
        "std_msgs.msg": mod("std_msgs.msg", Bool=msg, Float32=msg, String=msg),
        "visualization_msgs": mod("visualization_msgs"),
        "visualization_msgs.msg": mod("visualization_msgs.msg", Marker=msg, MarkerArray=msg),
    }
    for k, v in fakes.items():
        monkeypatch.setitem(sys.modules, k, v)
    sys.modules.pop("robotdance_ros2.motion_server_node", None)
    m = importlib.import_module("robotdance_ros2.motion_server_node")
    yield m
    sys.modules.pop("robotdance_ros2.motion_server_node", None)


def test_node_pause_seek_estop_wired_to_server(node_mod) -> None:
    node = node_mod.MotionServerNode(_certified())

    # pause トピックでコアが paused になる / resume で戻る。
    node.subs["/robotdance/pause"](SimpleNamespace(data=True))
    assert node.server.paused is True
    node.subs["/robotdance/pause"](SimpleNamespace(data=False))
    assert node.server.paused is False

    # seek トピック（phase 0..1）で cursor が末尾へ動く。
    n = node.server._kps.shape[0]
    node.subs["/robotdance/seek"](SimpleNamespace(data=1.0))
    assert node.server._cursor == n - 1
    node.subs["/robotdance/seek"](SimpleNamespace(data=0.0))
    assert node.server._cursor == 0

    # estop トピックで guard が停止状態になる。
    node.subs["/robotdance/estop"](SimpleNamespace(data=True))
    assert node.server.guard.check_certificate(node.server.motion).is_abort


def test_node_subscribes_expected_topics(node_mod) -> None:
    node = node_mod.MotionServerNode(_certified())
    assert {"/robotdance/estop", "/robotdance/pause", "/robotdance/seek"} <= set(node.subs)
