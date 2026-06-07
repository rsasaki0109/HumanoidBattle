"""決定的ハッシュ n-gram テキスト特徴（v0, 依存なし）。

caption / action label を固定長の bag-of-features ベクトルに符号化する。事前学習言語モデルを
使わず、`hashlib` による安定ハッシュで unigram + bigram を固定次元へ振り分ける（プロセス間で
決定的: Python 組み込み `hash()` は PYTHONHASHSEED で攪乱されるため使わない）。

⚠️ v0: contrastive text-motion の **テキスト側の足場**。事前学習エンコーダ（CLIP/
sentence-transformers 等）への差し替えは将来。

語の同義性は本来学習側の MLP に任せる設計だが、ハッシュ特徴は「同じ綴り → 同じ次元」しか
保証しないため、**未学習の言い換え**（"jog" vs "run"、"twirl" vs "spin"、"flipping" vs
"somersault"）は別バケットに落ちて retrieval が外れやすい。そこで hashing の前段に
**curated な概念正規化 + 軽量ステミング**を挟み、同義語を共通コンセプト・トークンへ畳む。
事前学習 LM ではなく決定的な手書き辞書なので依存ゼロ・プロセス間決定的を維持する。
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

TEXT_DIM = 256

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# 概念 → その表層形（同義語）の一覧。RobotDance の合成 action 群（dance_fast / dance_slow /
# idle / backflip）と、それに付く caption の語彙を中心に、よくある言い換えを共通トークンへ畳む。
# canonical は "concept:" prefix 付きにして通常トークンと衝突しないようにする。
_CONCEPTS: dict[str, list[str]] = {
    # 動作の種類。
    "dance": ["dance", "dancing", "danced", "dancer", "groove", "grooving",
              "boogie", "bop"],
    "flip": ["flip", "backflip", "somersault", "somersaults", "frontflip",
             "tumble", "tumbling", "flipping", "flipped", "aerial"],
    "spin": ["spin", "spinning", "twirl", "twirling", "whirl", "whirling",
             "pirouette", "rotate", "rotating"],
    "jump": ["jump", "jumping", "leap", "leaping", "hop", "hopping", "bound",
             "bounding"],
    "run": ["run", "running", "jog", "jogging", "sprint", "sprinting", "dash"],
    "walk": ["walk", "walking", "stroll", "strolling", "march", "marching",
             "step", "stepping", "pace", "pacing"],
    "wave": ["wave", "waving", "waved"],
    "kick": ["kick", "kicking", "kicked"],
    "punch": ["punch", "punching", "jab", "jabbing"],
    "squat": ["squat", "squatting", "crouch", "crouching", "kneel", "kneeling"],
    "sway": ["sway", "swaying", "swayed", "rock", "rocking"],
    "stand": ["stand", "standing", "stood"],
    "bow": ["bow", "bowing", "bowed"],
    "clap": ["clap", "clapping", "clapped", "applaud", "applauding"],
    "turn": ["turn", "turning", "turned", "pivot", "pivoting"],
    "lunge": ["lunge", "lunging", "lunged"],
    "balance": ["balance", "balancing", "balanced", "steady"],
    "stretch": ["stretch", "stretching", "stretched"],
    "throw": ["throw", "throwing", "threw", "thrown", "toss", "tossing"],
    "reach": ["reach", "reaching", "reached"],
    "crawl": ["crawl", "crawling", "crawled"],
    "roll": ["roll", "rolling", "rolled"],
    "raise": ["raise", "raising", "raised", "lift", "lifting", "lifted"],
    "bend": ["bend", "bending", "bent"],
    "twist": ["twist", "twisting", "twisted"],
    # 様態。
    "fast": ["fast", "quick", "quickly", "rapid", "rapidly", "speedy", "swift",
             "swiftly", "brisk"],
    "energetic": ["energetic", "energetically", "energy", "upbeat", "lively",
                  "vigorous", "vigorously", "dynamic", "explosive", "intense",
                  "powerful"],
    "slow": ["slow", "slowly", "gentle", "gently", "calm", "calmly", "relaxed",
             "relaxing", "leisurely", "smooth", "smoothly", "soft", "softly",
             "mellow"],
    "still": ["still", "motionless", "idle", "resting", "rest", "stationary",
              "immobile", "static", "frozen", "barely"],
    "acrobatic": ["acrobatic", "acrobatically", "acrobatics", "gymnastic",
                  "gymnastics", "athletic"],
    # 方向。
    "backward": ["backward", "backwards", "back", "rearward"],
    "forward": ["forward", "forwards", "ahead"],
    # 身体部位。
    "arm": ["arm", "arms"],
    "leg": ["leg", "legs"],
}

# 表層形 → "concept:<canonical>" の逆引き辞書（決定的・読み取り専用）。
_SYNONYM: dict[str, str] = {
    surface: f"concept:{canon}"
    for canon, surfaces in _CONCEPTS.items()
    for surface in surfaces
}

# ステミングで剥がす接尾辞（長い順に試す）。over-stem を避けるため最小限。
_SUFFIXES = ("ingly", "ing", "edly", "ed", "ly", "es", "s")


def tokenize(text: str) -> list[str]:
    """小文字化して英数字トークンに分割する（記号は区切り）。"""
    return _TOKEN_RE.findall(text.lower())


def _stem(token: str) -> str:
    """ごく軽量な接尾辞ステミング。短い語幹は残す（"is" → "is" 等）。"""
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def normalize_token(token: str) -> str:
    """トークンを概念トークンへ正規化する。

    1. 表層形が同義語辞書にあればその概念へ（"jogging" → "concept:run"）。
    2. 無ければステミングして再度辞書を引く（"dancing" → 辞書直引き、"twirled" → "twirl" 経由）。
    3. それでも未知なら語幹をそのまま返す（綴り違いを少しでも畳む: "kicks" → "kick"）。
    """
    if token in _SYNONYM:
        return _SYNONYM[token]
    stem = _stem(token)
    if stem in _SYNONYM:
        return _SYNONYM[stem]
    return stem


def normalize_tokens(text: str) -> list[str]:
    """text をトークン化して概念正規化したトークン列を返す（デバッグ / 検査用）。"""
    return [normalize_token(t) for t in tokenize(text)]


def _bucket(token: str) -> int:
    """トークンを安定ハッシュで [0, TEXT_DIM) のバケットに写す。"""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % TEXT_DIM


def text_features(text: str) -> np.ndarray:
    """caption を L2 正規化済みの bag-of-(unigram, bigram) 特徴ベクトル [TEXT_DIM] にする。

    トークンを概念正規化してから unigram + bigram をハッシュ加算する。これにより未学習の
    言い換え（"flipping backwards" ≈ "a backward somersault"）が共通バケットへ落ちて、
    contrastive head が見たことのある表現と整合しやすくなる。
    空文字や未知語のみでも zero ベクトルを返す（学習側で扱える）。
    """
    toks = [normalize_token(t) for t in tokenize(text)]
    vec = np.zeros(TEXT_DIM, dtype=np.float32)
    if not toks:
        return vec
    for tok in toks:
        vec[_bucket(tok)] += 1.0
    for a, b in zip(toks[:-1], toks[1:]):
        vec[_bucket(f"{a}_{b}")] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec
