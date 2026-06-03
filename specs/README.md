# RobotDance Specifications

> **仕様は実装より偉い。** `specs/` はこのリポジトリの最上位概念です。
> 各 `robotdance_*` パッケージは、ここで定義された schema / 規約に従って実装されます。

RobotDance の OSS moat は巨大モデルではなく、**Motion IR の標準化**です。
動画フォーマット・SMPL・2D/3D keypoints・motion embeddings・robot URDF・ROS2 trajectory・RL policy data を、
一つの中間表現に集約することで、RobotDance は「モデル repo」ではなく「運動データの OS」になります。

## 仕様一覧

| Spec | バージョン | 役割 |
| --- | --- | --- |
| [rd-manifest](rd-manifest/) | v0 (draft) | どの動画の・どの時間範囲を・どの条件で・どの処理で再構築するかを記述（動画そのものではない） |
| [rd-mir](rd-mir/) | v0 (draft) | **中核標準。** canonical skeleton, root trajectory, contacts, metadata を持つ motion IR |
| [rd-embodiment](rd-embodiment/) | v0 (draft) | ロボット形態（joint, limits, morphology, contact surface, runtime adapter） |
| [rd-motion](rd-motion/) | v0 (draft) | robot-specific な実行可能モーション artifact（`.rdmotion`） |
| [rd-policy](rd-policy/) | v0 (draft) | policy の observation / action I/O |

## 設計原則

1. **skeleton-first。** canonical skeleton + contacts + root trajectory が第一級表現。SMPL は optional field（license friction を避ける）。
2. **license state を schema レベルで持つ。** `redistribution_allowed` / `derived_motion_allowed` / `training_allowed` / `commercial_allowed` を明示。`unknown` の派生 motion は公開しない。
3. **provenance を保持する。** `source_ref`, `extractor_versions`, `retarget_certificates` で来歴と再現性を担保する。
4. **embodiment 非依存。** 一つの RD-MIR から複数ロボットへ retarget でき、結果は RD-Motion に robot ごとの certificate として残る。

## バージョニング

- 各 spec は独立してバージョンを持つ（`*_version` フィールド）。
- v0 は **draft**。破壊的変更があり得る。v1.0 で stable 化（ロードマップ参照）。
- schema 本体（JSON Schema）の license は CC0 / Apache-2.0。manifest や motion の**中身**の利用許諾は source ごとに別管理。
