"""RD-Motion の関節角軌道を実機/シム SDK 向けに書き出す（v0）。

retarget-ik が出力する actuator-space の関節角（rad）を、Unitree などの SDK が
position control で素直に食える時系列フォーマット（CSV / JSON）へ変換する「出口」。

ライセンス安全: 出力は数値（関節角）とフォーマットのみ。メッシュ / URDF は含まない。

motor index について: joint_names の並びは実 URDF の関節定義順で、これは Unitree の
LowCmd（motor_cmd[i].q）の慣例と一致する。すなわち列 index = motor index として扱える。
これは参照軌道（位置合わせ）であり、base pose・接地・バランスは含まない。実機投入前に
sim_certificate 等で別途検証すること。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from robotdance_core.rd_motion import RdMotion

EXPORT_FORMATS = ("csv", "json")


def joint_trajectory(motion: RdMotion) -> tuple[list[str], list[list[float]]]:
    """RD-Motion から actuator 関節名と角度時系列（[T, n_joints] rad）を取り出す。"""
    jr = motion.joint_rotations or {}
    names = jr.get("actuated_joint_names")
    angles = jr.get("angles_rad")
    if not names or angles is None:
        raise ValueError(
            "RD-Motion に actuator-space 関節角（joint_rotations.angles_rad）がありません。"
            " retarget-ik で生成した .rdmotion を渡してください。"
        )
    n = len(names)
    for i, row in enumerate(angles):
        if len(row) != n:
            raise ValueError(
                f"frame {i}: 関節角 {len(row)} 個が関節名 {n} 個と一致しません。"
            )
    return list(names), [[float(x) for x in row] for row in angles]


def joint_velocities(frames: list[list[float]], fps: float) -> list[list[float]]:
    """関節角時系列を有限差分して角速度（rad/s）を返す（端点は片側差分, 中央は中心差分）。

    実機の position+velocity control で velocity feedforward に使える。1 フレームのみなら 0。
    """
    arr = np.asarray(frames, dtype=np.float64)
    if arr.shape[0] < 2:
        return [[0.0] * (arr.shape[1] if arr.ndim == 2 else 0) for _ in range(arr.shape[0])]
    return (np.gradient(arr, axis=0) * fps).tolist()


def export_joint_trajectory(motion: RdMotion, out: str | Path, *, fmt: str = "csv",
                            include_velocity: bool = False) -> Path:
    """actuator 関節角（と任意で角速度）を実機/シム SDK 向けに書き出す。

    fmt="csv": 1 行目 ``time_s,<joint...>``、以降 1 フレーム 1 行（時刻 = frame/fps、角度 rad）。
        include_velocity=True で各関節の角速度列 ``d_<joint>``（rad/s, 有限差分）を後ろに付ける。
        コメント行を付けず最大互換（pandas / numpy / 各 SDK が素直に読める）。
    fmt="json": fps・units・joint_names を含むメタ付き（motor index = joint_names の index）。
        include_velocity=True で ``velocities``（rad/s）を追加。
    """
    out = Path(out)
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"未知の format: {fmt}（{' | '.join(EXPORT_FORMATS)}）")
    names, frames = joint_trajectory(motion)
    fps = float(motion.fps)
    vels = joint_velocities(frames, fps) if include_velocity else None

    if fmt == "csv":
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["time_s", *names]
            if vels is not None:
                header += [f"d_{n}" for n in names]
            w.writerow(header)
            for i, row in enumerate(frames):
                out_row = [round(i / fps, 6), *row]
                if vels is not None:
                    out_row += vels[i]
                w.writerow(out_row)
    else:  # json
        doc: dict[str, Any] = {
            "format": "robotdance.joint_trajectory.v0",
            "robot": motion.robot_name,
            "control_mode": motion.control_mode,
            "units": "rad, rad/s" if vels is not None else "rad",
            "fps": fps,
            "n_joints": len(names),
            "n_frames": len(frames),
            "joint_names": names,
            "frames": frames,
            "note": (
                "motor index = joint_names の index（実 URDF 定義順 = Unitree LowCmd の慣例）。"
                " position control の目標角。base pose・接地・バランスは含まない参照軌道。"
                " 実機投入前に sim_certificate 等で検証すること。"
            ),
        }
        if vels is not None:
            doc["velocities"] = vels
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out
