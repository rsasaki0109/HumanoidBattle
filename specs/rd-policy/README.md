# RD-Policy

> **status:** v0 (draft, スキーマは Phase 3 で確定)

Motion Policy の I/O 規約。policy は 2 種類:

- **Tracking Policy** — robot state + target robot motion + phase + contacts + latent を入力に、
  joint targets / PD targets / torque / residual を出力。retargeted motion の安定追従・sim-to-real。
- **Skill Policy** — robot state + motion latent + language/VLA intent + environment を入力に、
  whole-body action を出力。skill prior として VLA の action space を豊かにする補完レイヤー。

予定メッセージ（[`robotdance_ros2/`](../../robotdance_ros2/) と整合）:

| 型 | 内容 |
| --- | --- |
| `PolicyObservation` | robot state + target |
| `PolicyAction` | joint command / residual |
| `MotionLatent` | embedding / token |

RL framework は自前で作らず、Isaac Lab / MuJoCo を backend にし、RobotDance は motion data /
motion prior / retargeting target / benchmark を供給する frontend として接続します。
