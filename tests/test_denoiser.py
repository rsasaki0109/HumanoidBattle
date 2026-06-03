"""Motion denoiser / in-betweening + 長尺生成（§4.2 拡張）の縦スライス。

torch 未インストール環境では skip する（CI では skip）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import jsonschema  # noqa: E402

from robotdance_core.synthetic import generate_dance  # noqa: E402
from robotdance_models.denoiser import MotionDenoiser, train_denoiser  # noqa: E402
from robotdance_models.prior import MotionGenerator, train_prior  # noqa: E402
from robotdance_models.tokenizer import train_tokenizer  # noqa: E402

_SCHEMA = json.loads(
    (Path(__file__).resolve().parent.parent / "specs" / "rd-mir" / "rd-mir.schema.json")
    .read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def tokenizer_ckpt(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("den") / "tok.pt"
    train_tokenizer(out_path=p, epochs=60)
    return p


def test_denoiser_trains_above_chance(tokenizer_ckpt: Path, tmp_path: Path) -> None:
    out = tmp_path / "den.pt"
    res = train_denoiser(tokenizer_ckpt=tokenizer_ckpt, out_path=out, epochs=150, seed=0)
    assert out.exists()
    assert res["loss_history"][-1] < res["loss_history"][0]
    # ランダム（1/num_codes ≈ 0.008）を大きく上回る masked-token 復元精度。
    assert res["masked_token_acc"] > 0.1


def test_denoise_and_inbetween_produce_valid_mir(tokenizer_ckpt: Path, tmp_path: Path) -> None:
    out = tmp_path / "den.pt"
    train_denoiser(tokenizer_ckpt=tokenizer_ckpt, out_path=out, epochs=150, seed=0)
    den = MotionDenoiser(out)

    clean = generate_dance(beats_per_second=1.0)
    ids = den.tok.encode(clean)
    assert len(ids) > 4

    # 破損 → denoise。
    rng = np.random.default_rng(0)
    pos = rng.choice(len(ids), size=max(1, len(ids) // 4), replace=False)
    corrupt = ids.copy()
    corrupt[pos] = rng.integers(0, den.num_codes, size=len(pos))
    corrupt_mir = den.tok.decode_to_mir(corrupt, motion_id="corrupt")
    denoised, info = den.denoise(corrupt_mir, detect_ratio=0.3)
    assert info["masked"] >= 1
    assert denoised.num_frames > 0
    jsonschema.Draft202012Validator(_SCHEMA).validate(denoised.to_dict())

    # in-betweening: 両端を残し中間を埋める → 有効な RD-MIR。
    ib, toks = den.inbetween(clean, keep=2)
    assert len(toks) == len(ids)
    assert ib.num_frames > 0
    jsonschema.Draft202012Validator(_SCHEMA).validate(ib.to_dict())


def test_long_form_generation_via_sliding_window(tokenizer_ckpt: Path, tmp_path: Path) -> None:
    """prior が seq_len を超える長さを sliding-window で生成し、滑らかさを保つ。"""
    prior = tmp_path / "prior.pt"
    train_prior(tokenizer_ckpt=tokenizer_ckpt, out_path=prior, epochs=120, seed=0)
    gen = MotionGenerator(prior)

    long = gen.generate(length=4 * gen.seq_len, seed=0)
    # 長尺（seq_len 超）が生成される。
    assert long.num_frames > gen.seq_len
    kp = long.keypoints_3d_array()
    jitter = float(np.linalg.norm(np.diff(kp, n=2, axis=0), axis=2).mean())
    assert np.isfinite(jitter) and jitter < 1.0  # 発散せず滑らか
    jsonschema.Draft202012Validator(_SCHEMA).validate(long.to_dict())
