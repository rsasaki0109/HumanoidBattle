# Contributing to RobotDance

> 🚧 Pre-v0.1。仕様と雛形を整備中です。大きな実装に入る前に Issue で方針合意を推奨します。

## 基本原則

1. **仕様は実装より偉い。** schema を変える PR は [`specs/`](specs/) と該当 README を必ず更新する。
2. **ライセンス安全性は最優先。** raw video を repo に入れない。source license が `unknown` の派生 motion を公開しない。
3. **skeleton-first。** SMPL/SMPL-X を core 必須依存にしない（optional plugin）。
4. **sim-first / safety-first。** 実機 path は安全 gate の後ろにのみ置く。

## 開発環境

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## CLI の動作確認

```bash
robotdance validate manifest examples/minimal_manifest.json
robotdance validate mir examples/minimal_mir.json
```

## ライセンス

- Code: Apache-2.0。コントリビューションは同ライセンス下で提供したものとみなされます。
- データセット・モデル weights の利用許諾は source ごとに別管理。Data Bill of Materials を伴わない
  データ contribution は受け付けません。
