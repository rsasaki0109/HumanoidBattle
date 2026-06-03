"""Motion token prior（VQ-VAE トークン列の生成モデル）の検証。torch 無しは skip。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from robotdance_core.synthetic import generate_dance  # noqa: E402
from robotdance_models.prior import MotionGenerator, MotionPrior, train_prior  # noqa: E402
from robotdance_models.tokenizer import train_tokenizer  # noqa: E402


def test_prior_causal_shapes() -> None:
    import torch

    net = MotionPrior(vocab=40, d_model=32, nhead=2, nlayers=1, max_len=16)
    tokens = torch.zeros(4, 16, dtype=torch.long)
    logits = net(tokens)
    assert logits.shape == (4, 16, 40)


def _train(tmp_path):
    tok = tmp_path / "tok.pt"
    pri = tmp_path / "prior.pt"
    train_tokenizer(out_path=tok, epochs=120, num_codes=64, seed=0)
    train_prior(tokenizer_ckpt=tok, out_path=pri, seq_len=16, epochs=300, seed=0)
    return pri


def test_prior_learns_token_grammar(tmp_path) -> None:
    tok = tmp_path / "tok.pt"
    pri = tmp_path / "prior.pt"
    train_tokenizer(out_path=tok, epochs=120, num_codes=64, seed=0)
    res = train_prior(tokenizer_ckpt=tok, out_path=pri, seq_len=16, epochs=300, seed=0)
    assert res["loss_history"][-1] < 0.3 * res["loss_history"][0]
    assert res["next_token_acc"] > 0.6        # 合成 corpus のトークン文法を学習


def test_generate_produces_valid_motion(tmp_path) -> None:
    gen = MotionGenerator(_train(tmp_path))

    m = gen.generate(length=16, temperature=1.0, seed=0)
    assert m.keypoints_3d is not None
    kp = m.keypoints_3d_array()
    assert kp.shape[0] == m.num_frames and kp.shape[1:] == (19, 3)
    assert np.isfinite(kp).all()

    # 生成は滑らか（学習分布に沿う）: フレーム間加速度が小さい。
    jitter = float(np.linalg.norm(np.diff(kp, n=2, axis=0), axis=2).mean())
    assert jitter < 0.3

    # 異なる seed は異なるトークン列を生む（決定論的でない）。
    a = gen.generate(length=16, temperature=1.0, seed=0)
    b = gen.generate(length=16, temperature=1.0, seed=5)
    assert not np.array_equal(a.keypoints_3d_array(), b.keypoints_3d_array())


def test_completion_keeps_prefix(tmp_path) -> None:
    gen = MotionGenerator(_train(tmp_path))
    dance = generate_dance(beats_per_second=1.0)
    keep = 4
    comp, tokens = gen.complete(dance, keep=keep, temperature=0.8, seed=0)
    # 先頭 keep トークンは元 motion のものを保持。
    assert tokens[:keep].tolist() == gen.tok.encode(dance)[:keep].tolist()
    assert comp.keypoints_3d is not None
