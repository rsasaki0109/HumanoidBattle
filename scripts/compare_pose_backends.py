#!/usr/bin/env python3
"""複数の OSS 2D pose 検出器を同じ実動画で走らせ、横並び overlay GIF で比較する。

RobotDance の抽出は MediaPipe Pose（3D world landmarks）が既定だが、pose 検出器は色々ある。
本スクリプトは **MediaPipe / YOLO11-pose(Ultralytics) / RTMPose(rtmlib)** を同一クリップに当て、
各検出器の骨格を **共通の COCO-17 表現**に揃えて原フレームへ重ね、3 パネルの比較 GIF を書き出す。
検出率・平均 confidence・推論時間も集計して表示する。

⚠️ 入力動画は repo に同梱しない（license-safe）。出力は overlay（ソース動画ピクセルを含む派生物 →
CC-BY 等の出典明記で利用）。MediaPipe のみ 3D world landmarks を返し robot retarget に使える。
YOLO/RTMPose は 2D で、3D 化には別途 lifting が要る（本比較は検出品質の確認が目的）。

依存（dev のみ・パッケージ依存ではない）: mediapipe, ultralytics, rtmlib, opencv-python, imageio。

使い方:
    python scripts/compare_pose_backends.py clip.mp4 -o assets/readme/pose_compare.gif
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

# COCO-17 の骨格エッジと色。
_COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6), (0, 1), (0, 2), (1, 3), (2, 4),
]
# MediaPipe BlazePose 33 → COCO 17 の対応 index。
_MP33_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def _largest_person(kxy: np.ndarray, kconf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """多人数検出から bbox 面積が最大の人物（前景被写体）を選ぶ。"""
    best, area = 0, -1.0
    for i in range(len(kxy)):
        pts = kxy[i][kconf[i] > 0.2]
        if len(pts) < 4:
            continue
        a = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        if a > area:
            area, best = a, i
    return kxy[best], kconf[best]


def _draw(frame: np.ndarray, xy: np.ndarray, conf: np.ndarray, color, thr: float = 0.3) -> None:
    import cv2

    for a, b in _COCO_EDGES:
        if conf[a] > thr and conf[b] > thr:
            cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), color, 2,
                     cv2.LINE_AA)
    for i in range(17):
        if conf[i] > thr:
            cv2.circle(frame, tuple(xy[i].astype(int)), 3, (0, 0, 255), -1, cv2.LINE_AA)


def _label(frame: np.ndarray, text: str, color) -> None:
    import cv2

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 24), (32, 32, 32), -1)
    cv2.putText(frame, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _mediapipe_runner():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    from robotdance_perception.mediapipe_adapter import ensure_model

    opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    lm = vision.PoseLandmarker.create_from_options(opt)

    def run(frame_rgb, ts_ms, w, h):
        res = lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb)), ts_ms)
        if not res.pose_landmarks:
            return None
        nl = res.pose_landmarks[0]
        full = np.array([[p.x * w, p.y * h] for p in nl])
        vis = np.array([p.visibility for p in nl])
        return full[_MP33_TO_COCO], vis[_MP33_TO_COCO]
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("assets/readme/pose_compare.gif"))
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--width", type=int, default=300, help="各パネルのリサイズ幅")
    args = ap.parse_args()

    import cv2
    import imageio.v2 as imageio
    from rtmlib import Body
    from ultralytics import YOLO

    mp_run = _mediapipe_runner()
    yolo = YOLO("yolo11n-pose.pt")
    rtm = Body(mode="lightweight", backend="onnxruntime", device="cpu")

    colors = {"MediaPipe": (80, 200, 120), "YOLO11-pose": (255, 170, 0), "RTMPose": (180, 120, 255)}
    stats = {k: {"det": 0, "conf": 0.0, "ms": 0.0} for k in colors}
    n_seen = 0
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_out, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            n_seen += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            panels = {}

            t = time.time()
            mpr = mp_run(rgb, int(idx * 1000.0 / fps), w, h)
            stats["MediaPipe"]["ms"] += (time.time() - t) * 1000
            panels["MediaPipe"] = mpr

            t = time.time()
            yr = yolo(frame, verbose=False)[0].keypoints
            stats["YOLO11-pose"]["ms"] += (time.time() - t) * 1000
            if yr is not None and len(yr.data):
                panels["YOLO11-pose"] = _largest_person(yr.xy.cpu().numpy(), yr.conf.cpu().numpy())
            else:
                panels["YOLO11-pose"] = None

            t = time.time()
            kxy, ksc = rtm(frame)
            stats["RTMPose"]["ms"] += (time.time() - t) * 1000
            panels["RTMPose"] = _largest_person(np.array(kxy), np.array(ksc)) if len(kxy) else None

            tiles = []
            for name in colors:
                tile = frame.copy()
                res = panels[name]
                if res is not None:
                    _draw(tile, res[0], res[1], colors[name])
                    stats[name]["det"] += 1
                    stats[name]["conf"] += float(res[1].mean())
                _label(tile, name, colors[name])
                scale = args.width / tile.shape[1]
                tiles.append(cv2.resize(tile, (args.width, int(tile.shape[0] * scale))))
            frames_out.append(cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames_out, duration=args.stride / fps, loop=0)
    print(f"✓ {len(frames_out)} frames → {args.out} ({args.out.stat().st_size // 1024} KB)\n")
    print(f"{'backend':14s} {'det_rate':>8s} {'mean_conf':>10s} {'ms/frame':>9s}")
    for k, s in stats.items():
        d = s["det"]
        print(f"{k:14s} {d / n_seen:8.2f} {(s['conf'] / d if d else 0):10.3f} {s['ms'] / n_seen:9.0f}")


if __name__ == "__main__":
    main()
