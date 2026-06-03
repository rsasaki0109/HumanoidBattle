# RD-Manifest

> **status:** v0 (draft)

URL-only dataset を扱うための標準。**manifest は動画そのものではない。**
「どの動画の・どの時間範囲を・どの条件で・どの処理で再構築するか」を記述します。

## 重要な設計ルール

- 動画ファイルは repo に置かない。
- SNS の非公式 downloader を公式 path にしない（`rebuild_method` に `official_api` / `manual_download` / `local_file` / `prohibited`）。
- 公式 API・ユーザー持ち込み・研究 API・許諾済みデータだけを primary path にする。
- source license が `unknown` の派生 motion は公開しない。
- manifest build 時に license / availability を再検証する（`license_verified_at`, `status`）。
- モデルカードに Data Bill of Materials を必ず出す。

## ライセンス Tier

| Tier | データ種別 | raw 再配布 |
| --- | --- | --- |
| A: Open / Consented | 自前撮影, 明示許諾, CC BY/CC0 相当, AIST++ annotations | 可能な範囲で可 |
| B: Research / Restricted | AMASS, BABEL, Motion-X, SMPL 系 | 条件付き |
| C: URL-only Internet Videos | TikTok, YouTube Shorts, Reels | raw も派生 motion も原則配らない |

スキーマ本体: [`rd-manifest.schema.json`](rd-manifest.schema.json)
