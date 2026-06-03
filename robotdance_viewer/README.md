# robotdance_viewer

3D skeleton / side-by-side / 原動画 overlay 可視化 — RD-MIR・RD-Motion を「映える」形で見せる。

## 主な関数（`skeleton_view`）

- `render_gif(mir, out, *, caption=None, show_meta=True)` — RD-MIR を回転 3D スケルトン GIF に。
  **caption**（None なら `semantics.action_label` を自動）を上部バナーに、license_state/fps/frames を
  下部メタ行に重ねる。
- `render_side_by_side(panels, out, *, verdicts=None, title=None)` — 複数スケルトンを同一スケールで
  横並び（human ↔ robot, before ↔ after）。`verdicts` で PASS/REJECT 等のバッジ、`title` で図全体の
  タイトル（検索クエリ等）。
- `render_search_montage(query, results, out)` — **text 検索の top-k 結果**を、クエリをタイトルに・
  類似度（cosine）をバッジにして横並び描画（§6 見せ場）。

```bash
robotdance view dance.rdmir.json -o dance.gif                # caption 付き 3D GIF
robotdance search-text "fast dancing" --gif search.gif      # 検索結果モンタージュ
robotdance demo-multi -o many_humanoids.gif                 # same motion, many humanoids
```

> ⚠️ v0: matplotlib バックエンドの GIF レンダラ。web viewer / インタラクティブ検索 UI は今後。
> `pip install -e ".[demo]"` で matplotlib / imageio を入れる。
