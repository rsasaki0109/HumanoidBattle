# RD-MIR — RobotDance Motion Intermediate Representation

> **status:** v0 (draft) · **中核標準**

RD-MIR は RobotDance の OSS moat です。人間動画・mocap・合成データを、一つの canonical な運動表現に集約します。

- 設計原則は **skeleton-first**: `skeleton` + `contacts` + `root_trajectory` が第一級表現。
- SMPL/SMPL-X は `smpl_params` の **optional field**（core を SMPL 必須依存にしない → license friction 回避）。
- `license_state` を schema レベルで保持し、`unknown` の派生 motion は公開しない。

## 主要フィールド

| フィールド | 内容 |
| --- | --- |
| `motion_id` | 安定 ID（UUID + provenance hash） |
| `source_ref` | URL, platform ID, local path, dataset name |
| `license_state` | `redistributable` / `trainable` / `commercial_allowed` / `research_only` / `unknown` |
| `fps`, `duration`, `world_frame` | canonical サンプリングと座標規約（z-up, x-forward 等） |
| `root_trajectory` | pelvis/root の world position, orientation, velocity |
| `skeleton`, `joint_rotations`, `keypoints_3d` | canonical skeleton と運動 |
| `contacts` | left/right foot, toe, heel, hand の接地 |
| `confidence` | frame-level / joint-level の信頼度 |
| `quality_metrics` | occlusion, foot skating, smoothness 等 |
| `semantics` | action label, caption, music tempo, style |
| `privacy_flags` | minor-risk, face-visible, consented, synthetic |
| `extractor_versions` | pose / HMR / smoothing のバージョン（再現性） |
| `retarget_certificates` | robot 別 feasibility result |

スキーマ本体: [`rd-mir.schema.json`](rd-mir.schema.json)

> ⚠️ v0 は draft。フレーム長・配列の厳密 shape 検証（`T = round(fps * duration)` 等）は
> `robotdance_core` のバリデータ側で段階的に追加します（JSON Schema だけでは表現しきれないため）。
