"""Motion VQ-VAE（離散トークナイザ）の検証。torch 無しは skip。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from robotdance_core.synthetic import generate_backflip, generate_dance  # noqa: E402
from robotdance_models.encoder import INPUT_DIM  # noqa: E402
from robotdance_models.tokenizer import (  # noqa: E402
    DOWNSAMPLE,
    MotionTokenizer,
    MotionVQVAE,
    train_tokenizer,
)


def test_vqvae_shapes_and_downsample() -> None:
    import torch

    net = MotionVQVAE(num_codes=32)
    x = torch.zeros(4, 32, INPUT_DIM)
    recon, idx, commit = net(x)
    assert recon.shape == (4, 32, INPUT_DIM)
    assert idx.shape == (4, 32 // DOWNSAMPLE)     # 時間方向に DOWNSAMPLE 圧縮
    assert commit.ndim == 0


def test_training_reduces_loss_and_uses_codebook(tmp_path) -> None:
    res = train_tokenizer(out_path=tmp_path / "tok.pt", epochs=120, num_codes=64, seed=0)
    h = res["loss_history"]
    assert h[-1] < 0.2 * h[0]                       # 再構成が大きく進む
    assert res["recon_mse"] < 0.01                  # 正規化空間で十分小さい
    assert res["codes_used"] >= 8                   # codebook collapse していない


def test_tokenize_reconstruct_roundtrip(tmp_path) -> None:
    train_tokenizer(out_path=tmp_path / "tok.pt", epochs=120, num_codes=64, seed=0)
    tok = MotionTokenizer(tmp_path / "tok.pt")

    dance = generate_dance(beats_per_second=1.0)
    ids = tok.encode(dance)
    assert ids.ndim == 1 and ids.dtype.kind in "iu"
    assert ids.min() >= 0 and ids.max() < 64

    orig, rec = tok.reconstruct(dance)
    assert orig.shape == rec.shape
    rmse = float(np.sqrt(((orig - rec) ** 2).mean()))
    assert rmse < 0.1                               # トークンからの再構成が妥当

    # decode_to_mir: トークン列 → RD-MIR（フレーム数はタイル長に一致）。
    mir = tok.decode_to_mir(ids)
    assert mir.keypoints_3d is not None
    assert mir.num_frames == orig.shape[0]

    # backflip と dance はトークン列が異なる（別の運動 → 別のコード）。
    flip_ids = tok.encode(generate_backflip())
    assert set(ids.tolist()) != set(flip_ids.tolist())
