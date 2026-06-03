"""Motion-X (whole-body text-motion) → canonical RD-MIR ローダ（skeleton-first, v0）。

Motion-X は SMPL-X（whole-body: 体幹+手+顔）パラメータに**自然文記述**を付けた大規模 text-motion
データセット。標準のモーション表現は per-frame **322 次元**ベクトル:

    [ root_orient(3) | pose_body(63) | pose_hand(90) | pose_jaw(3) |
      face_expr(50) | face_shape(100) | trans(3) | betas(10) ]

v0 は canonical 19-joint（body のみ）が対象なので、**root_orient + pose_body = 66 次元（SMPL body
22 joint の axis-angle）+ trans** を取り出し、既存の skeleton-first SMPL FK で canonical 化する
（手・顔・betas は未使用）。記述文は `semantics` に格納する。

⚠️ ライセンス: Motion-X は研究用途中心。既定 license_state は "research_only"。.npy / テキストは
repo に含めない（利用者が各自取得）。SMPL-X model file は使わない（skeleton-first）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from robotdance_core.rd_mir import LicenseState, RdMir, Skeleton
from robotdance_core.skeleton import JOINT_NAMES, PARENTS, index_of

from .smpl import smpl_poses_to_canonical

# 322 次元表現の body 部分スライス。
_ROOT_ORIENT = slice(0, 3)
_POSE_BODY = slice(3, 66)
_TRANS = slice(309, 312)
_FULL_DIM = 322


def _extract_body(motion: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Motion-X モーション配列から body の axis-angle poses [T,22,3] と trans [T,3] を取り出す。

    受理: 322 次元（標準）/ 66 次元（body のみ）/ [T,22,3]（既に整形済み）。
    """
    motion = np.asarray(motion, dtype=np.float64)
    if motion.ndim == 3 and motion.shape[1:] == (22, 3):
        return motion, None
    if motion.ndim != 2:
        raise ValueError(f"Motion-X motion は [T, D] か [T,22,3] が必要: {motion.shape}")
    d = motion.shape[1]
    if d >= _FULL_DIM:
        pose = np.concatenate([motion[:, _ROOT_ORIENT], motion[:, _POSE_BODY]], axis=1)
        trans = motion[:, _TRANS]
        return pose.reshape(motion.shape[0], 22, 3), trans
    if d >= 66:
        return motion[:, :66].reshape(motion.shape[0], 22, 3), None
    raise ValueError(f"Motion-X motion の次元が不足（>=66 が必要）: {d}")


def motionx_to_mir(
    motion: np.ndarray,
    texts: list[str] | str | None = None,
    *,
    fps: float = 30.0,
    license_state: LicenseState = "research_only",
    motion_id: Optional[str] = None,
    ground_align: bool = True,
) -> RdMir:
    """Motion-X のモーション配列（322 次元等）と記述文から canonical RD-MIR を生成する。"""
    pose, trans = _extract_body(motion)
    kps = smpl_poses_to_canonical(pose, trans)  # [T, 19, 3]
    if ground_align:
        kps[:, :, 2] -= kps[:, [index_of("left_ankle"), index_of("right_ankle")], 2].min()

    if isinstance(texts, str):
        texts = [texts]
    caption = (texts[0].strip() if texts and texts[0].strip() else "unknown")
    n = kps.shape[0]
    return RdMir(
        motion_id=motion_id or "rdmir-motionx",
        source_ref={"dataset_name": "motionx", "extractor": "smplx_body_fk"},
        license_state=license_state,
        fps=float(fps),
        duration=float(n / fps),
        world_frame={"up_axis": "z", "forward_axis": "x", "handedness": "right"},
        skeleton=Skeleton(joint_names=list(JOINT_NAMES), parents=list(PARENTS)),
        root_trajectory={"position": kps[:, index_of("pelvis"), :].tolist()},
        keypoints_3d=kps.tolist(),
        contacts=_estimate_contacts(kps),
        privacy_flags={"synthetic": False},
        extractor_versions={"source": "motionx", "adapter": "robotdance.data.motionx.v0"},
        semantics={"action_label": caption, "captions": list(texts or []),
                   "source_dataset": "motionx", "whole_body_dropped": "hands,face,betas"},
    )


def load_motionx(
    motion_path: str | Path, text_path: str | Path | None = None, *,
    fps: float = 30.0, license_state: LicenseState = "research_only",
    motion_id: Optional[str] = None,
) -> RdMir:
    """Motion-X の `motion/<id>.npy`（+ `texts/<id>.txt`）→ RD-MIR。"""
    motion_path = Path(motion_path)
    motion = np.load(motion_path)
    texts: list[str] = []
    if text_path is not None and Path(text_path).exists():
        for line in Path(text_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                texts.append(line)
    return motionx_to_mir(motion, texts, fps=fps, license_state=license_state,
                          motion_id=motion_id or f"rdmir-motionx-{motion_path.stem}")


def _estimate_contacts(kps: np.ndarray) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for side in ("left", "right"):
        z = kps[:, index_of(f"{side}_ankle"), 2]
        out[f"{side}_foot"] = (z < float(z.min()) + 0.07).tolist()
    return out
