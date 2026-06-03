# robotdance_data

manifests, source adapters, dataset builder, dedupe, license firewall — URL/manifest 駆動のデータパイプライン。raw video は再配布しない。

## 実装状況

| module | 役割 |
| --- | --- |
| `smpl.py` | SMPL/SMPL-H body skeleton の FK（**SMPL model file 不要**の skeleton-first）+ canonical 19 へのマップ |
| `amass.py` | `load_amass_npz(path) -> RdMir`。AMASS の SMPL pose を canonical RD-MIR 化（fps ダウンサンプル付き） |
| `manifest.py` | RD-Manifest 読込・schema 検証 + **license firewall**（`evaluate(manifest) -> FirewallDecision`） |
| `dataset.py` | manifest 駆動ビルダー。firewall を通し公開可のみ書き出し、**Data Bill of Materials** を出力 |

```bash
robotdance build-dataset manifests.json --data-root /path/to/amass -o build/
# → build/<clip>.rdmir.json と build/DATA_CARD.md（Data Bill of Materials）
```

## ライセンスファイアウォール

- `license_declared=unknown` または `derived_motion_allowed=false` → 派生 motion を**書き出さない**。
- 公開可の clip には manifest の権利フラグから `license_state`（redistributable/trainable/research_only…）を付与。
- raw source（動画・mocap）は再配布しない。manifest（URL/再構築手順）と派生 motion のみ扱う。
- ビルドのたびに **Data Bill of Materials** を出力し、どの source が・どの権利で・公開されたかを明示。

> ⚠️ **v0 注意:** SMPL FK の rest offset は近似（正確な shape-conditioned joint regressor は未使用）。
> retarget は direction-preserving なので下流に影響は小さい。AIST++ / Motion-X 等の adapter、
> 重複除去（perceptual/motion hash）は今後。実 AMASS は登録制で同梱せず、利用者が各自取得する。
