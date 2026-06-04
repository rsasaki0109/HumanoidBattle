#!/usr/bin/env python3
"""左に「人間が踊る」RD-MIR スケルトン、右に実 Unitree メッシュが同じ振付で踊る GIF を作る。

README の hero asset 生成用（パッケージ本体ではない）。RobotDance の核心
「Human → Humanoid Motion Compiler」を 1 枚で見せる: **同一の合成ダンス RD-MIR** を、
左は人間スケルトンとして matplotlib で、右は actuator-space IK で実 G1 の 23 関節角へ
retarget して pybullet メッシュでレンダリングし、フレームを横連結する。両者は同じ振付・
同じタイミングなので「人間の動き → ロボットの動き」の対応がそのまま見える。

⚠️ **メッシュ / URDF は repo に同梱しない**（license-safe）。利用者が unitree_ros 等から取得した
ローカル URDF を指す。出力 GIF は RobotDance パイプライン出力の可視化（render）であり、
メッシュ本体の再配布ではない。人間側は合成 RD-MIR なので素材ライセンスの問題も無い。

依存（dev のみ）: pybullet, imageio, Pillow, matplotlib, torch（actuator-IK）。

使い方:
    python scripts/render_human_vs_robot_gif.py /path/to/g1_23dof.urdf \
        -o assets/readme/human_vs_g1.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# リポジトリルートを import path に追加（scripts/ から実行されるため）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _render_human_frames(mir, indices, *, height: int, elev: float, azim: float):
    """RD-MIR keypoints を人間スケルトンとして matplotlib で描き、RGB フレーム列を返す。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from robotdance_core.skeleton import BONES
    from robotdance_viewer.skeleton_view import _project, _view_matrix

    kps = mir.keypoints_3d_array()  # [T, J, 3]
    view = _view_matrix(elev, azim)
    projected = np.stack([_project(kps[f], view) for f in range(kps.shape[0])])  # [T, J, 2]
    flat = projected.reshape(-1, 2)
    mins, maxs = flat.min(axis=0), flat.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max()) / 2.0 + 0.15

    dpi = 100
    fig, ax = plt.subplots(figsize=(height / dpi * 0.82, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    frames = []
    for f in indices:
        ax.clear()
        ax.set_facecolor("white")
        pts = projected[f]
        for child, parent in BONES:
            ax.plot([pts[child, 0], pts[parent, 0]], [pts[child, 1], pts[parent, 1]],
                    color="#1f77b4", linewidth=3.0, solid_capstyle="round")
        ax.scatter(pts[:, 0], pts[:, 1], color="#d62728", s=22, zorder=3)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("Human (RD-MIR)", fontsize=13, fontweight="bold", color="#333333")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[..., :3].copy())
    plt.close(fig)
    return frames


def _render_robot_frames(mir, urdf, robot, base_z, indices, *, width, height):
    """合成ダンスを実 URDF の関節角へ IK retarget し、pybullet メッシュで描いて RGB 列を返す。"""
    import pybullet as p

    from robotdance_retarget.actuator_ik import actuator_retarget
    from robotdance_unitree.urdf_import import G1_LINK_MAP, H1_LINK_MAP

    link_map = H1_LINK_MAP if robot == "h1" else G1_LINK_MAP
    motion = actuator_retarget(mir, str(urdf), steps=250,
                               link_map=link_map, robot_name=f"unitree_{robot}")
    angles = np.asarray(motion.joint_rotations["angles_rad"])
    names = [str(n) for n in motion.joint_rotations["actuated_joint_names"]]
    print(f"actuator-IK: {angles.shape[0]} frames, {angles.shape[1]} joints, "
          f"IK err {motion.retarget_metrics['ik_mean_pos_error_m']} m")

    urdf_dir = Path(urdf).resolve().parent
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(str(urdf_dir))
    gv = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 3, 0.01], rgbaColor=[0.93, 0.93, 0.95, 1])
    gc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3, 3, 0.01])
    p.createMultiBody(0, gc, gv, basePosition=[0, 0, -0.005])
    rid = p.loadURDF(Path(urdf).name, useFixedBase=True, basePosition=[0, 0, base_z])

    jmap = {}
    for i in range(p.getNumJoints(rid)):
        info = p.getJointInfo(rid, i)
        if info[2] == p.JOINT_REVOLUTE:
            jmap[info[1].decode()] = i
    pairs = [(jmap[n], k) for k, n in enumerate(names) if n in jmap]

    proj = p.computeProjectionMatrixFOV(42, width / height, 0.1, 10)
    cam_target_z = base_z * 0.78
    cam_dist = base_z * 2.45
    t_len = angles.shape[0]
    frames = []
    for f in indices:
        for ji, k in pairs:
            p.resetJointState(rid, ji, float(angles[f, k]))
        yaw = 35 + 25 * np.sin(2 * np.pi * f / t_len)
        view = p.computeViewMatrixFromYawPitchRoll([0, 0, cam_target_z], cam_dist, yaw, -10, 0, 2)
        img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER,
                               lightDirection=[0.6, 0.7, 1.2], shadow=1)
        frames.append(np.reshape(img[2], (height, width, 4))[:, :, :3].astype(np.uint8))
    p.disconnect()
    return frames


def _label_top_center(frame: np.ndarray, text: str) -> np.ndarray:
    """フレーム上部中央に太字ラベルを描く（human パネルの matplotlib title と同じ DejaVu）。"""
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    band = 44  # ロボットが上端に達しても被らないよう、白い帯を上に足してそこに描く。
    canvas = np.full((frame.shape[0] + band, frame.shape[1], 3), 255, dtype=np.uint8)
    canvas[band:] = frame
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        path = font_manager.findfont(font_manager.FontProperties(weight="bold"))
        font = ImageFont.truetype(path, 22)
    except Exception:
        font = ImageFont.load_default()
    w = draw.textlength(text, font=font)
    draw.text(((img.width - w) / 2, 12), text, fill="#333333", font=font)
    return np.asarray(img)


def _hstack(left, right, *, gutter: int = 14):
    """同じ高さの 2 フレームを白い間仕切りを挟んで横連結する（高さは left に合わせる）。"""
    from PIL import Image

    h = left.shape[0]
    rw = int(round(right.shape[1] * h / right.shape[0]))
    r = np.asarray(Image.fromarray(right).resize((rw, h), Image.LANCZOS))
    gut = np.full((h, gutter, 3), 255, dtype=np.uint8)
    return np.concatenate([left, gut, r], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf", type=Path, help="実 Unitree URDF（メッシュ付き, ローカル取得）")
    ap.add_argument("-o", "--out", type=Path, default=Path("assets/readme/human_vs_robot.gif"))
    ap.add_argument("--robot", choices=["g1", "h1"], default="g1")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--bps", type=float, default=1.3, help="beats per second")
    ap.add_argument("--arm", type=float, default=1.8)
    ap.add_argument("--sway", type=float, default=0.18)
    ap.add_argument("--base-z", type=float, default=0.793, help="pelvis 高さ（足が接地する値）")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--width", type=int, default=440)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--elev", type=float, default=10.0)
    ap.add_argument("--azim", type=float, default=-70.0)
    args = ap.parse_args()

    import imageio.v2 as imageio

    from robotdance_core.synthetic import generate_dance

    if args.robot == "h1" and args.base_z == 0.793:
        args.base_z = 1.04

    mir = generate_dance(duration=args.duration, beats_per_second=args.bps,
                         arm_amp=args.arm, sway_amp=args.sway)
    t_len = mir.num_frames
    indices = list(range(0, t_len, args.stride))

    print(f"human | {args.robot} を同一ダンス {t_len}f から生成（{len(indices)} render frames）...")
    robot_frames = _render_robot_frames(mir, args.urdf, args.robot, args.base_z,
                                        indices, width=args.width, height=args.height)
    human_frames = _render_human_frames(mir, indices, height=args.height,
                                        elev=args.elev, azim=args.azim)

    robot_label = "Unitree H1" if args.robot == "h1" else "Unitree G1"
    robot_frames = [_label_top_center(r, robot_label) for r in robot_frames]
    combined = [_hstack(h, r) for h, r in zip(human_frames, robot_frames)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, round(mir.fps / args.stride))
    imageio.mimsave(args.out, combined, duration=1.0 / fps, loop=0)
    print(f"✓ {len(combined)} frames → {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
