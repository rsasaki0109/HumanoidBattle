# RD-Embodiment

> **status:** v0 (draft)

retarget の対象となるロボット形態を記述します。一つの [RD-MIR](../rd-mir/) を複数の embodiment へ retarget でき、
人間 motion はそのまま流せない（link 長・joint limits・足裏形状・torque・重心が違う）ため、
contact-preserving / dynamics-aware / embodiment-conditioned retargeting の入力になります。

| フィールド | 内容 |
| --- | --- |
| `robot_name` | `unitree_g1`, `unitree_h1`, `digit`, `figure` 等 |
| `urdf_ref` | URDF/SDF/MJCF |
| `joint_names` / `joint_limits` | joint と position/velocity/torque 制限 |
| `link_lengths` | morphology |
| `end_effectors` / `contact_surfaces` | feet, hands / sole geometry |
| `control_modes` | position / velocity / torque / policy |
| `nominal_pose` | default standing pose |
| `safety_limits` | allowed motion envelope |
| `runtime_adapter` | ROS2 / unitree_sdk2 / ros2_control / isaac_lab / mujoco |

primary target は Unitree G1（最優先）と H1。config 実体は [`robotdance_unitree/`](../../robotdance_unitree/) に置きます。

スキーマ本体: [`rd-embodiment.schema.json`](rd-embodiment.schema.json)
