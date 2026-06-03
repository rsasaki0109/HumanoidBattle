# robotdance_perception

pose adapters, human tracking, HMR adapters, smoothing — 2D/3D pose・人間メッシュ復元を adapter 方式で束ねる。

## 実装状況

- `mediapipe_adapter.py` — **MediaPipe Pose による local 動画 → RD-MIR**。
  `extract_motion(video) -> RdMir` が pose_world_landmarks（33点・メートル3D）を canonical
  19-joint へマップする。`mp_world_landmarks_to_canonical` は純関数で単体テスト可能。
- `hmr.py` — **HMR（Human Mesh Recovery）adapter: SMPL 出力 → RD-MIR**。4DHumans（HMR2.0/PHALP,
  rotmat）/ GVHMR（axis-angle・world-grounded）が回帰した **per-frame SMPL パラメータ**
  （global_orient / body_pose / transl / **betas**）を、既存の **skeleton-first SMPL FK**
  （`robotdance_data.smpl`）で canonical 19-joint に変換する。MediaPipe（2D→近似 3D）よりオクルージョン・
  奥行き・world trajectory に強い入口。`from_gvhmr(dict)` / `from_4dhumans(dict)` / `from_dict(dict)`
  / 共通 core `hmr_smpl_to_mir(...)`。axis-angle / rotation-matrix は形状から自動判別。
  - **betas（shape conditioning, v0 近似）**: betas があれば rest offset を身長(β0)/体幅(β1)の
    **粗い線形プロキシ**でスケールし個体差を first-order で反映（真の SMPL blend shapes ではない）。
  - **native loader**: `load_hmr_file(path)` が `.npz/.npy/.pkl/.pt` を読み、dict 構造から
    GVHMR/4DHumans/汎用を自動判別（`load_hmr_npz` は汎用 .npz 専用）。モデル weight は不要。

```python
from robotdance_perception.hmr import from_gvhmr, load_hmr_file
mir = from_gvhmr(gvhmr_result)             # GVHMR の出力 dict → RD-MIR（betas で shape-conditioning）
mir = load_hmr_file("track.pkl")           # native .npz/.npy/.pkl/.pt → 構造を自動判別

from robotdance_perception.mediapipe_adapter import extract_motion
mir = extract_motion("my_clip.mp4")        # → RD-MIR（license_state="unknown"）
```

```bash
robotdance import-hmr track.pkl -o clip.rdmir.json   # native(.npz/.npy/.pkl/.pt) → RD-MIR
```

- 座標変換: MediaPipe world（x:右, y:下, z:手前負, 腰原点）→ canonical（x:前, y:左, z:上）= `(-z, x, -y)`。
- HMR: SMPL frame（x:左, y:上, z:前）→ canonical = `(z, x, y)`（`robotdance_data.smpl` と共通）。
- モデル（Google 配布 Apache-2.0 の `.task`）は `~/.cache/robotdance/models/` へ自動 DL（`ROBOTDANCE_POSE_MODEL` で上書き可）。

> ⚠️ **ライセンス:** 入力動画の権利はユーザー責任。アダプタは動画を再配布せず、抽出 RD-MIR の
> `license_state` は既定で `"unknown"`（source 未確認 → 派生 motion を公開しない）。
> 検証は landmark→canonical の単体テスト + scikit-image の astronaut（NASA, public domain）実写で行う。
> **HMR adapter（v0）:** モデル weight / SMPL body model file は**同梱・実行しない**（HMR 推論は
> ツール側）。本 adapter は出力 SMPL パラメータ → canonical の変換のみを担い、numpy/scipy だけで
> CI 検証する。skeleton-first で、**betas は身長/体幅の粗いプロキシ**（真の SMPL blend shapes ではない）。
> 特定モデル版に pin した精度検証ではなく**文書化された出力構造**に対する検証。multi-person tracking・
> temporal smoothing 強化は今後。
> `pip install -e ".[perception]"` で mediapipe / opencv を入れる（HMR adapter は不要）。
