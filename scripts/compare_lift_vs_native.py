#!/usr/bin/env python3
"""native 3D（MediaPipe）と coarse planar lift（2D 検出器→3D）を同一動画で定量比較する。

同じクリップに 2 経路を当て、canonical 3D を比較する:
- native:  MediaPipe world landmarks（深度あり）
- lift:    YOLO11-pose / RTMPose の COCO-17 2D を解析的 planar lift（深度なし・x=0 平面）

正直な狙い: lift は **冠状面（正面 y-z 平面）では native に近いが、深度（前後 x）はゼロに潰れる**
ことを数値で示す。出力は (1) 指標表、(2) native|lift の canonical skeleton 横並び GIF（任意）。

⚠️ 入力動画は repo に同梱しない。GIF はパイプライン出力（抽出 keypoints）の可視化で動画ピクセルを
含まないため license-safe。

使い方:
    python scripts/compare_lift_vs_native.py karate.mp4 --detector yolo11-pose -o out.gif
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from robotdance_perception.lifting import extract_via_lift  # noqa: E402
from robotdance_perception.mediapipe_adapter import extract_motion  # noqa: E402


def _pelvis_center(kps: np.ndarray) -> np.ndarray:
    """各フレームで pelvis（index 0）を原点へ。"""
    return kps - kps[:, 0:1, :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--detector", default="yolo11-pose", help="lift 元の 2D 検出器")
    ap.add_argument("-o", "--out", type=Path, default=None, help="native|lift skeleton GIF 出力先")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    print(f"native(MediaPipe) 抽出: {args.video.name} ...")
    native = extract_motion(args.video, smooth=True)
    print(f"lift({args.detector}+planar) 抽出 ...")
    lift = extract_via_lift(args.video, detector=args.detector, smooth=True)

    a = _pelvis_center(native.keypoints_3d_array())
    b = _pelvis_center(lift.keypoints_3d_array())
    t = min(len(a), len(b))
    a, b = a[:t], b[:t]

    # 深度（前後 x）のレンジ: native は非ゼロ、lift は 0 に潰れる。
    fwd_native = float(a[:, :, 0].std())
    fwd_lift = float(b[:, :, 0].std())
    # 冠状面（y,z）と全体の MPJPE（native を基準に lift がどれだけ離れるか）。
    mpjpe_full = float(np.linalg.norm(a - b, axis=-1).mean())
    mpjpe_frontal = float(np.linalg.norm(a[:, :, 1:] - b[:, :, 1:], axis=-1).mean())

    print(f"\n{'metric':28s} {'value':>10s}")
    print(f"{'frames (min of two)':28s} {t:10d}")
    print(f"{'native depth-x std [m]':28s} {fwd_native:10.4f}")
    print(f"{'lift depth-x std [m]':28s} {fwd_lift:10.4f}  ← 平面なので 0")
    print(f"{'MPJPE full [m]':28s} {mpjpe_full:10.4f}")
    print(f"{'MPJPE frontal y-z [m]':28s} {mpjpe_frontal:10.4f}  ← 正面では近い")
    if mpjpe_full > 1e-9:
        print(f"{'frontal/full ratio':28s} {mpjpe_frontal / mpjpe_full:10.2f}  "
              "← 小さいほど『差は深度由来』")

    if args.out is not None:
        from robotdance_viewer.skeleton_view import render_side_by_side

        render_side_by_side(
            [(a, "native (MediaPipe 3D)", "#1f77b4"),
             (b, f"lift ({args.detector}, planar)", "#d62728")],
            args.out, fps=native.fps, stride=args.stride,
            title="native 3D vs planar lift (depth collapses)",
        )
        print(f"\n✓ skeleton GIF: {args.out}")


if __name__ == "__main__":
    main()
