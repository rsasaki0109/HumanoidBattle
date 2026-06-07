#!/usr/bin/env python3
"""2 本の GIF を共通の高さに揃え、ラベル付きで横並び結合して 1 本の hero GIF を作る。

README 冒頭の hero（実 karate overlay ｜ 実 G1）を**再現可能**にするためのユーティリティ。
入力 GIF は同一 extract・同一 stride で生成された同期済みのものを想定（フレーム数が揃っている）。
出力は PIL の adaptive palette で減色して軽量化する（実写を含む左パネルは GIF 圧縮が効きにくいため）。

⚠️ 入力 GIF はパイプライン出力（overlay は CC-BY 出典明記で利用可）。生動画は同梱しない。

使い方:
    python scripts/make_hero_gif.py \
        assets/readme/real/karate3_g1_overlay.gif "real video + skeleton overlay" \
        assets/readme/real/karate3_g1_robot.gif   "Unitree G1 reproduces it" \
        -o assets/readme/karate_hero.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# 左右パネルのラベル色（BGR ではなく RGB。cv2.putText には RGB 画像にそのまま描く）。
_LEFT_COLOR = (120, 200, 255)
_RIGHT_COLOR = (160, 230, 160)


def _panel(img: np.ndarray, label: str, color, height: int, banner: int) -> np.ndarray:
    import cv2

    img = img[:, :, :3]
    h, w = img.shape[:2]
    w2 = int(round(w * height / h))
    resized = cv2.resize(img, (w2, height), interpolation=cv2.INTER_AREA)
    bar = np.full((banner, w2, 3), 28, np.uint8)
    cv2.putText(bar, label, (5, banner - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return np.vstack([bar, resized])


def make_hero(left_gif: Path, left_label: str, right_gif: Path, right_label: str,
              out: Path, *, height: int = 300, gap: int = 10, colors: int = 48,
              banner: int = 24, duration: float = 0.1) -> Path:
    """2 本の GIF を横並び結合して out に書き出す。書き出した Path を返す。"""
    import imageio.v2 as imageio
    from PIL import Image

    left = imageio.mimread(str(left_gif))
    right = imageio.mimread(str(right_gif))
    n = min(len(left), len(right))
    frames = []
    for i in range(n):
        lp = _panel(left[i], left_label, _LEFT_COLOR, height, banner)
        rp = _panel(right[i], right_label, _RIGHT_COLOR, height, banner)
        col = np.full((lp.shape[0], gap, 3), 255, np.uint8)
        frames.append(np.hstack([lp, col, rp]))
    ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=colors) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    ims[0].save(out, save_all=True, append_images=ims[1:],
                duration=int(duration * 1000), loop=0, optimize=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("left_gif", type=Path)
    ap.add_argument("left_label")
    ap.add_argument("right_gif", type=Path)
    ap.add_argument("right_label")
    ap.add_argument("-o", "--out", type=Path, default=Path("assets/readme/karate_hero.gif"))
    ap.add_argument("--height", type=int, default=300)
    ap.add_argument("--colors", type=int, default=48)
    args = ap.parse_args()

    out = make_hero(args.left_gif, args.left_label, args.right_gif, args.right_label,
                    args.out, height=args.height, colors=args.colors)
    print(f"✓ hero GIF → {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
