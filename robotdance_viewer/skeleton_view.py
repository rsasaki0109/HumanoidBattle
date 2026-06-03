"""RD-MIR の 3D スケルトンを GIF に描画するビューア（v0）。

side-by-side（原動画 | 3D | robot）の最初の 1 枚目として、3D skeleton の再生 GIF を生成する。
3D 点を自前の正射影で 2D に落として matplotlib の通常軸に描く（壊れやすい Axes3D / mplot3d に依存しない）。
外部 GPU / モデル不要。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ヘッドレス環境でのレンダリング
# システム mpl_toolkits と pip 版 matplotlib の競合で Axes3D import 警告が出るが、
# 本ビューアは自前正射影で 2D 描画するため無関係。無害な警告を抑制する。
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robotdance_core.rd_mir import RdMir  # noqa: E402
from robotdance_core.skeleton import BONES  # noqa: E402


def _view_matrix(elev: float, azim: float) -> np.ndarray:
    """elev/azim（度）から、world→camera の回転行列を作る。

    canonical は z-up。azim で z 軸回りに回し、elev で水平軸回りに傾ける。
    """
    el = np.radians(elev)
    az = np.radians(azim)
    cz, sz = np.cos(az), np.sin(az)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    cx, sx = np.cos(el), np.sin(el)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rx @ rz


def _project(pts: np.ndarray, view: np.ndarray) -> np.ndarray:
    """[J, 3] world → [J, 2] screen（正射影、x=横・z=縦）。"""
    cam = pts @ view.T
    return np.stack([cam[:, 0], cam[:, 2]], axis=1)


def render_gif(
    mir: RdMir,
    out_path: str | Path,
    *,
    stride: int = 2,
    elev: float = 12.0,
    azim: float = -70.0,
    dpi: int = 80,
) -> Path:
    """RD-MIR の keypoints_3d を回転スケルトンの GIF として書き出す。

    stride: 何フレームおきに描画するか（GIF を軽くする）。
    """
    import imageio.v2 as imageio

    kps = mir.keypoints_3d_array()  # [T, J, 3]
    n_frames = kps.shape[0]
    view = _view_matrix(elev, azim)

    # 全フレームを投影し、人物が縮まないよう等方な描画範囲を決める。
    projected = np.stack([_project(kps[f], view) for f in range(n_frames)])  # [T, J, 2]
    flat = projected.reshape(-1, 2)
    mins, maxs = flat.min(axis=0), flat.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max()) / 2.0 + 0.1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    fig, ax = plt.subplots(figsize=(4, 5), dpi=dpi)
    for f in range(0, n_frames, stride):
        ax.clear()
        pts = projected[f]
        for child, parent in BONES:
            ax.plot(
                [pts[child, 0], pts[parent, 0]],
                [pts[child, 1], pts[parent, 1]],
                color="#1f77b4",
                linewidth=2.5,
                solid_capstyle="round",
            )
        ax.scatter(pts[:, 0], pts[:, 1], color="#d62728", s=14, zorder=3)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"RD-MIR · {mir.motion_id}", fontsize=8)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[..., :3].copy())
    plt.close(fig)

    fps = max(1, round(mir.fps / stride))
    imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)
    return out_path
