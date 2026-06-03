# robotdance_ros2

messages, motion server, safety guard, robot adapters — 安全な motion runtime としての ROS2 層。

## 実装状況（ROS2 Jazzy 想定）

| module | 役割 | ROS2 依存 |
| --- | --- | --- |
| `messages.py` | MotionFrame / SafetyState 等の dataclass 契約 | なし |
| `safety_guard.py` | **Safety Guard（§5.6）**: certificate gate, 速度クランプ, 転倒検知, E-stop, speed scaling | なし |
| `motion_server.py` | **Motion Server**: .rdmotion → 安全フレーム逐次供給（pause/speed/phase） | なし |
| `motion_server_node.py` | rclpy ノード: MarkerArray(skeleton) + SafetyState を配信、E-stop 購読 | rclpy |

```bash
robotdance serve g1.rdmotion.json --speed 0.5        # dry-run（ROS2 不要）で安全ゲート検証
robotdance serve g1.rdmotion.json --ros2             # ROS2 配信（RViz で可視化）
robotdance demo-runtime                              # certificate PASS は再生 / REJECT は遮断
```

ROS2 topic（`--ros2`）:
- `/robotdance/skeleton` `visualization_msgs/MarkerArray`（bone LINE_LIST, RViz 可視化）
- `/robotdance/safety` `std_msgs/String`（SafetyState JSON）
- `/robotdance/estop` `std_msgs/Bool`（True で緊急停止）

## 安全設計（§5.6）

motion artifact は **certificate gate → 速度/転倒チェック → E-stop** を通って初めて配信される。
`sim_certificate` が無い / REJECT の motion は **再生前に ABORT**（`demo-runtime` で実演）。

> ⚠️ **v0:** core は ROS2 非依存で完全テスト可能。ノードは ROS2 Jazzy（primary target, §5.1）で動作。
> **sim-first** — 実機 bridge（unitree_sdk2）は安全レビュー後に別途接続。v0 は Cartesian（link 位置）
> 空間で動作し、joint-space limit clamp / ros2_control 連携 / 実機再生は今後。
> 本パッケージは pip monorepo に同居するが、custom .msg の colcon パッケージ化も今後の課題。
