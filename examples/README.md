# Examples

雛形段階のサンプル。spec が実際にバリデーションを通ることを示す最小例です。

| ファイル | 内容 |
| --- | --- |
| `minimal_manifest.json` | RD-Manifest v0 の最小有効例 |
| `minimal_mir.json` | RD-MIR v0 の最小有効例（skeleton-first, SMPL なし） |
| `manifests/amass_example.json` | dataset ビルド用 manifest 配列の例（license firewall のデモ用） |

```bash
# manifest からデータセットを構築（local source があれば RD-MIR 化、無ければ withheld）
robotdance build-dataset examples/manifests/amass_example.json -o build/
```

```bash
robotdance validate manifest examples/minimal_manifest.json
robotdance validate mir examples/minimal_mir.json
```

> 実データ（動画・mocap）は repo に含めません。AIST++ 等は各自の利用規約に従って取得してください。
