# robotdance_models

tokenizer, encoder, diffusion/autoregressive model, policy training — Motion Encoder / Foundation Model / Policy 学習。

## 実装状況

- `encoder.py` — **Masked Motion Modeling encoder**（小型 Transformer）。canonical motion window を
  マスク再構成で自己教師あり学習する。
- `train.py` — 学習ループ + checkpoint + `LearnedMotionEncoder`。手作り特徴量と**同じ前処理**
  （`robotdance_motion.normalized_keypoints`）・**同じ `embed(mir)` interface**で `MotionIndex` に差し込める。
- `text.py` — **決定的ハッシュ n-gram テキスト特徴**（依存なし）。caption → 固定長ベクトル。
- `contrastive.py` — **Contrastive text-motion アライメント**（CLIP 風）。motion encoder と text MLP を
  共有埋め込み空間に射影し、(motion, caption) を multi-positive InfoNCE で整合させる。学習後は
  `embed_text` / `embed_motion` が同じ単位球面に乗り、**テキスト → モーション検索**が可能。
- `tokenizer.py` — **Motion VQ-VAE**（離散トークナイザ）。motion window を時間方向に圧縮した潜在列に
  符号化し、EMA codebook の最近傍コードに量子化して**離散トークン列**にする。decoder で復元。
  `MotionTokenizer.encode(mir) -> tokens` / `decode_to_mir(tokens) -> RD-MIR`。将来の生成・補完・
  テキスト条件付け（VLA 接続）の足場。dead-code 復活 + データ依存初期化で codebook collapse を回避。
- `prior.py` — **Motion token prior**（GPT 風 causal Transformer）。VQ-VAE トークン列上で next-token
  予測を学習し、`MotionGenerator.generate()` で**新規モーション生成**、`complete()` で**補完**を行う。
  tokenizer（符号化⇄復号）と prior（並びの確率モデル）が揃って初めて生成が動く。
  `generate(length=...)` は seq_len を超える長さを **sliding-window 自己回帰**で **長尺生成**できる。
- `denoiser.py` — **Motion denoiser / in-betweening**（双方向 Transformer, masked token modeling）。
  causal prior が「続きを作る」のに対し、双方向 denoiser は **全体の文脈で埋める/直す**:
  `MotionDenoiser.denoise()` が尤度の低い外れトークンを mask→双方向充填で**ノイズ除去**し、
  `inbetween()` が両端を残し中間を埋めて**補間（中割り）**する。foundation model スタックが
  生成（prior）+ 補間/除去（denoiser）を備える。
- `text2motion.py` — **Text-conditioned 生成**。token prior を**テキスト特徴で条件付け**し、
  `TextToMotion.generate(caption)` で **caption → モーション**を生成する（"a backflip" → バックフリップ）。
  `text.py`（テキスト特徴）+ `tokenizer.py`（VQ-VAE）+ `prior.py`（生成）を 1 本に繋ぐ集大成。
- `policy_export.py` — **RD-Policy export**（§3/§4.5）。学習済み tracking policy checkpoint を
  配布 artifact（`.rdpolicy`, `robotdance_core.rd_policy`）にまとめる: I/O 規約・アーキテクチャ・
  学習来歴・**安全制約**・**weights 参照**（format/ref/sha256, 本体は非埋め込み）。任意で **ONNX**
  （決定論方策, onnxruntime 実行可能 = 実機ランタイム橋渡し）を書き出す。`export-policy` CLI。
- `tracking_policy.py` — **RL tracking policy baseline**（§4.5）。
  [`robotdance_sim.TrackingEnv`](../robotdance_sim/)（base 非駆動の物理 env）で参照運動を
  **倒れずに追従する方策**を小型 **PPO** で学習する。学習表現の次にある制御スタックの足場で、
  retarget→sim_certificate の「物理的に妥当か」の判定の先にある「**実際にバランスを取って動かせるか**」を扱う。
  `TrackingPolicy.rollout()` が物理ロールアウトを RD-Motion（`control_mode="policy"`）として返し、
  viewer / sim_certificate / ROS2 の既存パイプラインに流せる。`train_multi_tracking_policy` +
  [`MultiTrackingEnv`](../robotdance_sim/) で参照スイートを **1 方策に汎化**（round-robin・
  reference-conditioned 観測）。

```bash
pip install -e ".[learn]"          # torch を入れる
robotdance train-encoder -o motion_encoder.pt --epochs 40
robotdance demo-motion-map --checkpoint motion_encoder.pt -o map_learned.png

# テキスト → モーション検索（contrastive）
robotdance train-text-motion -o text_motion.pt --epochs 200
robotdance search-text "a person doing a backflip" --checkpoint text_motion.pt

# モーション → 離散トークン（VQ-VAE）
robotdance train-tokenizer -o motion_tokenizer.pt --epochs 150
robotdance demo-tokenizer --checkpoint motion_tokenizer.pt -o tokenizer_recon.gif

# トークン生成 prior でモーション生成・補完（--length で長尺 sliding-window 生成）
robotdance train-prior --tokenizer motion_tokenizer.pt -o motion_prior.pt --epochs 300
robotdance demo-generate --checkpoint motion_prior.pt -o generated.gif --length 64

# masked denoiser でノイズ除去・in-betweening（双方向, §4.2 拡張）
robotdance train-denoiser --tokenizer motion_tokenizer.pt -o motion_denoiser.pt --epochs 300
robotdance demo-denoise --checkpoint motion_denoiser.pt -o denoise.gif

# テキストからモーションを生成（text → motion）
robotdance train-text2motion --tokenizer motion_tokenizer.pt -o text2motion.pt --epochs 400
robotdance generate-text "a person doing a backflip" --gif backflip.gif

# RL tracking policy（参照を物理上で追従, §4.5）— sim + learn extra が必要
robotdance train-tracking -o tracking_policy.pt --iterations 40
robotdance demo-track --iterations 40 -o tracking.gif         # 参照 vs 物理追従 を side-by-side
robotdance train-tracking --suite -o tracking_multi.pt        # 1 方策で複数運動を汎化
robotdance demo-track-multi --iterations 60 -o tracking_multi.gif  # スイートを横並び描画
```

```python
from robotdance_models.train import LearnedMotionEncoder
from robotdance_motion.embeddings import MotionIndex
enc = LearnedMotionEncoder("motion_encoder.pt")
idx = MotionIndex(embed_fn=enc.embed)   # 検索・重複除去・Motion Map が学習表現で動く

from robotdance_models.contrastive import TextMotionModel
model = TextMotionModel("text_motion.pt")
model.search("flipping backwards in the air", suite)   # → backflip が top-1
```

> ⚠️ **v0:** 学習**基盤**の提供が目的。
> - **masked encoder**: 合成 corpus で再構成 loss が下がり（例: 0.36→0.02）dance/backflip を分離できることを示すが、
>   **手作り baseline を超えると主張するものではない**（要・実データ規模）。
> - **contrastive text-motion**: 小さな合成 corpus・ハッシュ n-gram テキスト特徴（**事前学習言語モデルなし**）で
>   caption→motion を **action 群レベル top-1 100%**（exact は variant が可換なため低い）で引けることを示す。
>   実キャプション・データ規模・CLIP/sentence-transformers への差し替えは今後。
> - **motion VQ-VAE**: 合成 corpus で再構成 MSE が下がり（例: 0.055→0.0007）・codebook が健全に使われる
>   （collapse 回避）ことを示す。本モジュールは符号化⇄復号のみ。
> - **motion token prior**: VQ-VAE トークン列で next-token 精度 ~92% に達し、生成（滑らかな新規モーション,
>   jitter ~0.03）・補完（prefix を保持して継続）・**長尺生成**（seq_len 超を sliding-window で,
>   256 frames でも jitter ~0.035）が動くことを示す。
> - **motion denoiser / in-betweening**: 双方向 masked modeling で masked-token 復元精度がランダム
>   （~0.8%）を大きく上回る（合成 corpus で ~50%）。破損トークンの**ノイズ除去**・両端固定の
>   **補間（中割り）**が動くことを示す。bidirectional ゆえ生成 prior と相補的（長尺/betas は今後）。
> - **text-conditioned 生成**: caption の **action 群**（dance / idle / backflip）に応じて生成が変わる
>   （"a backflip" → energy ~0.26 vs "standing still" → ~0.02 の高/低エネルギー）。小さな合成 corpus の
>   ため語彙・多様性・新規 caption 汎化は限定的。**生成物は物理的に妥当とは限らない** —
>   retarget → sim_certificate（MuJoCo）の安全パイプラインを必ず通す。
> - **RL tracking policy**: base 非駆動の物理 env で PPO を学習し、gentle 参照を **survival 100%** で
>   追従（pose RMSE ~0.37）・物理ロールアウトを RD-Motion で返すことを示す。短い feasible クリップでは
>   関節 PD だけで概ねバランスするため、v0 の残差 PPO は **PD を壊さず追従する足場**であって PD 超えの
>   tracking 精度を主張しない。摂動頑健性・AMP/敵対報酬・実機転移は今後。
> - **multi-motion tracking**: `MultiTrackingEnv`（round-robin 参照切替）+ `train_multi_tracking_policy` で
>   **1 つの方策**が合成 4 運動スイート（gentle/normal/fast dance + idle）を **全運動 survival 100%** で
>   追従できることを示す（reference-conditioned 観測）。実 motion データ規模での汎化は今後。
>
> weights は repo に同梱しない（`robotdance-*` weight family の方針）。motion foundation model・
> 高度な RL tracking（AMP/摂動/実機転移）・contrastive **video**-text-motion は今後。
