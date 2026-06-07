"""テキスト特徴の概念正規化 / ステミングの検証（torch 非依存・CI で実行される）。"""

from __future__ import annotations

import numpy as np

from robotdance_models.text import (
    TEXT_DIM,
    normalize_token,
    normalize_tokens,
    text_features,
    tokenize,
)


def test_tokenize() -> None:
    assert tokenize("Fast, energetic DANCE!") == ["fast", "energetic", "dance"]


def test_normalize_token_folds_synonyms() -> None:
    assert normalize_token("jogging") == normalize_token("run") == "concept:run"
    assert normalize_token("twirl") == normalize_token("spinning") == "concept:spin"
    assert normalize_token("somersault") == normalize_token("backflip") == "concept:flip"
    assert normalize_token("motionless") == normalize_token("idle") == "concept:still"
    assert normalize_token("energetically") == normalize_token("upbeat") == "concept:energetic"
    assert normalize_token("gently") == normalize_token("slowly") == "concept:slow"
    # 拡充した動作語彙。
    assert normalize_token("pivoting") == normalize_token("turn") == "concept:turn"
    assert normalize_token("tossing") == normalize_token("throw") == "concept:throw"
    assert normalize_token("lifting") == normalize_token("raise") == "concept:raise"
    assert normalize_token("applauding") == normalize_token("clap") == "concept:clap"


def test_normalize_token_morphology_for_unknown_words() -> None:
    # 辞書に無い語でも屈折形同士は同じ語幹へ畳まれる（決定的）。
    assert normalize_token("wiggles") == normalize_token("wiggling") \
        == normalize_token("wiggled") == "wiggl"
    # 短すぎる語幹は剥がさない。
    assert normalize_token("is") == "is"
    # 未知語はそのまま。
    assert normalize_token("xyzzy") == "xyzzy"


def test_normalize_tokens_sentence() -> None:
    assert "concept:flip" in normalize_tokens("an acrobatic somersault")
    assert "concept:still" in normalize_tokens("standing perfectly motionless")
    assert "concept:run" in normalize_tokens("jogging in place")


def test_text_features_contract_preserved() -> None:
    a = text_features("a person doing a backflip")
    b = text_features("a person doing a backflip")
    assert a.shape == (TEXT_DIM,)
    assert np.allclose(a, b)                       # 決定的
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5     # L2 正規化
    assert np.allclose(text_features(""), 0.0)     # 空文字は zero
    assert not np.allclose(a, text_features("standing still"))


def test_text_features_paraphrases_correlate() -> None:
    # 同義の言い換え同士は共通バケットへ落ちて強く相関する。
    flip_a = text_features("flipping backwards")
    flip_b = text_features("a backward somersault")
    assert float(flip_a @ flip_b) > 0.5

    still_a = text_features("standing motionless")
    still_b = text_features("an idle resting pose")
    assert float(still_a @ still_b) > 0.3

    # 異なる概念同士は相関が低い。
    assert float(flip_a @ still_a) < 0.3
