<div align="center">

# 🕺 RobotDance

**Drop a human motion video. Get a humanoid motion dataset, embedding, and G1 simulation replay.**

*RobotDance は、権利管理された人間動画を、ヒューマノイドロボットの運動データ・運動埋め込み・学習 policy・実行可能モーションへ変換する OSS モーションコンパイラです。*

<!-- TODO(v0.1): assets/readme/ に 10 秒 side-by-side GIF を配置して下に埋め込む -->
<!-- ![RobotDance demo](assets/readme/shorts_to_g1.gif) -->

</div>

---

## 一言で

> **RobotDance = "Human Video to Humanoid Motion Compiler"**

単なる pose estimation ラッパーでも、単なる retargeting ツールでも、単なる robot policy repo でもありません。
人間の全身運動を、ヒューマノイドロボットが**学習・検索・模倣・実行**できる形に変換する基盤です。

```
Input:  a short human video
Output: Unitree G1 simulation motion + RD-MIR dataset + motion embedding
```

## What RobotDance is / is not

| ✅ RobotDance は | ❌ RobotDance ではない |
| --- | --- |
| 人間動画 → ヒューマノイド運動資産の **motion compiler** | TikTok/Instagram scraper |
| **Motion IR (RD-MIR)** を標準化するデータの OS | 単なる pose estimation ラッパー |
| Unitree G1/H1 を primary target にした **sim-first** 基盤 | 「動画を入れたら即実機が踊る」危険ツール |
| Isaac Lab 等に motion prior を供給する **frontend** | Isaac Lab / GR00T の competitor |
| ダンスは強いデモの一つ（対象はスポーツ・武道・日常動作・リハビリ等） | ダンス専用ツール |

## 4 つの出力

1. **Humanoid Motion Dataset** — ライセンス管理済みの全身運動データセット
2. **Motion Embeddings** — 検索・クラスタリング・VLA 接続・RL conditioning 用の運動表現
3. **Robot Policies** — G1/H1 が物理的に追従できる policy
4. **Robot Executable Motions** — ROS2 / Unitree SDK / sim runtime で再生可能な artifact

## アーキテクチャ

```
[Video Source]            ローカル / 許諾済み / URL manifest
      ↓
[Source Manifest Layer]   RD-Manifest: URL, license, provenance, rebuild recipe
      ↓
[Video Processing]        decode, segmentation, tracking, quality scoring
      ↓
[Human Motion Recovery]   2D pose → 3D pose / SMPL / world-grounded motion
      ↓
[RD-MIR]  ◀── 中核標準 ──  canonical skeleton, root trajectory, contacts, metadata
      ↓
[Motion Understanding]    embeddings, action tags, retrieval, quality classifier
      ↓
[Retargeting]             canonical human motion → robot embodiment motion
      ↓
[Physics Validation]      kinematic / contact / sim tracking, fall / torque / slip
      ↓
[Learning]                motion encoder, foundation model, RL tracker
      ↓
[Runtime]                 ROS2 / Unitree SDK2 / simulation / motion server
```

中核となる内部標準は **RD-MIR (RobotDance Motion Intermediate Representation)** です。詳細は [`specs/`](specs/) を参照。

## リポジトリ構成

```
specs/                  ◀── 仕様は実装より偉い（最上位に配置）
  rd-manifest/          URL/ライセンス/再構築手順
  rd-mir/               中核 motion IR
  rd-embodiment/        ロボット形態記述
  rd-motion/            robot-specific 実行可能モーション
  rd-policy/            policy I/O
robotdance_core/        schemas, validators, CLI, config
robotdance_data/        manifests, source adapters, dataset builder, license firewall
robotdance_perception/  pose / HMR adapters, tracking, smoothing
robotdance_motion/      canonicalization, contacts, embeddings, retrieval
robotdance_retarget/    contact-preserving retargeting
robotdance_sim/         Isaac Lab / MuJoCo backend adapters
robotdance_models/      tokenizer, encoder, foundation model, policy training
robotdance_ros2/        messages, motion server, safety guard
robotdance_unitree/     G1/H1 configs, URDF mapping, SDK2/ROS2 bridge
robotdance_benchmarks/  extraction / retarget / sim tracking benchmark
robotdance_viewer/      side-by-side video/motion/robot visualization
```

## データ & ライセンス安全性

- **raw video を再配布しない。** URL/manifest + ローカル再構築を基本にする。
- source license が `unknown` の派生 motion は公開しない。
- SMPL/SMPL-X は **必須依存にしない**（skeleton-first core、SMPL は optional plugin）。
- モデルは `robotdance-open-*` / `robotdance-research-*` / `robotdance-private-*` に分離。

| ライセンス対象 | 方針 |
| --- | --- |
| Code | Apache-2.0 |
| Schema / manifest | CC0 or Apache-2.0（中身の利用許諾は source ごとに分離） |
| Model weights | 学習データ構成に応じて open / research-only / 非配布 |

## 対応ロボット

| Robot | 状態 |
| --- | --- |
| Unitree G1 | 最優先（README demo, sim retarget） |
| Unitree H1 | full-size humanoid benchmark |
| R1 / H2 / Figure / Digit / Booster / NEO | future adapter |

## ロードマップ

| Version | テーマ |
| --- | --- |
| **v0.1** | Video → G1 Sim（local video → 3D motion → viewer → G1 retarget） |
| v0.2 | Dataset Builder（RD-Manifest, license firewall, HF export） |
| v0.3 | Motion Embeddings（encoder, retrieval, motion map） |
| v0.4 | Humanoid Retarget Benchmark（G1/H1 metrics, leaderboard） |
| v0.5 | ROS2 Runtime（motion server, safety guard, Unitree bridge / sim-first） |
| v1.0 | Stable Specs（RD-MIR / RD-Manifest / RD-Embodiment 安定化） |

実装 workstream の詳細は [`docs/ROADMAP.md`](docs/ROADMAP.md)。

## ステータス

🚧 **Pre-v0.1 / scaffold。** 現在はリポジトリ雛形と仕様の初版（v0 schema）を整備中です。

## License

Code は [Apache-2.0](LICENSE)。データセット/モデルの利用許諾は source ごとに別途確認してください。
