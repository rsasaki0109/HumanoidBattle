# RD-Policy

> **status:** v0 (draft) — スキーマ確定: [`rd-policy.schema.json`](rd-policy.schema.json)

学習済み motion policy の **配布 artifact**（`.rdpolicy`）。policy の I/O 規約・アーキテクチャ・
学習来歴・**安全制約**・**weights 参照**を 1 つの spec 適合 JSON にまとめる。weights 本体は
埋め込まず参照する（license/容量 safe）。`robotdance export-policy` で tracking policy checkpoint
から生成でき、任意で **ONNX**（実機ランタイム向け）も書き出す。

必須フィールド: `rd_policy_version` / `policy_id` / `policy_type`（tracking|skill）/ `robot_name` /
`observation`（dim, components）/ `action`（dim, space, base_actuated）/ `weights`（format, ref, sha256）。
任意: `control` / `architecture` / `training` / `safety_limits` / `failure_modes` / `provenance` /
`license_state` / `runtime_adapter`。


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
