"""AMASS (.npz, SMPL-H pose params) → canonical RD-MIR ローダ（skeleton-first, v0）。

AMASS は SMPL framework で 15+ の mocap データセットを統一した DB。各 .npz は
poses（axis-angle）, trans, mocap_framerate 等を持つ。ここでは SMPL body の先頭 22 joint を
FK して canonical 19-joint に変換する（SMPL body model file は使わない / 同梱しない）。

⚠️ ライセンス: AMASS は研究用途中心で sub-dataset ごとに条件が異なる。既定 license_state は
"research_only"。.npz / SMPL model file は repo に含めない。利用者が各自取得する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from robotdance_core.rd_mir import LicenseState, RdMir, Skeleton
from robotdance_core.skeleton import JOINT_NAMES, PARENTS, index_of

from .smpl import smpl_poses_to_canonical


def _read_fps(data) -> float:
    for key in ("mocap_framerate", "mocap_frame_rate"):
        if key in data:
            return float(np.asarray(data[key]).item())
    return 30.0


def load_amass_npz(
    npz_path: str | Path,
    *,
    license_state: LicenseState = "research_only",
    target_fps: Optional[float] = 30.0,
    motion_id: Optional[str] = None,
) -> RdMir:
    """AMASS .npz から canonical RD-MIR を生成する。

    target_fps を指定すると stride ダウンサンプルする（AMASS は 100-150fps が多い）。
    """
    path = Path(npz_path)
    data = np.load(path, allow_pickle=True)
    if "poses" not in data:
        raise ValueError(f"AMASS npz に 'poses' がありません: {path}")
    poses = np.asarray(data["poses"], dtype=np.float64)  # [T, >=66]
    trans = np.asarray(data["trans"], dtype=np.float64) if "trans" in data else None
    src_fps = _read_fps(data)

    body = poses[:, :66].reshape(poses.shape[0], 22, 3)  # body 22 joint の axis-angle

    fps = src_fps
    if target_fps and src_fps > target_fps * 1.3:
        stride = max(1, int(round(src_fps / target_fps)))
        body = body[::stride]
        trans = trans[::stride] if trans is not None else None
        fps = src_fps / stride

    kps = smpl_poses_to_canonical(body, trans)  # [T, 19, 3]
    # 接地: 足の最下点を z=0 へ。
    kps[:, :, 2] -= kps[:, [index_of("left_ankle"), index_of("right_ankle")], 2].min()

    n = kps.shape[0]
    contacts = _estimate_contacts(kps)
    return RdMir(
        motion_id=motion_id or f"rdmir-amass-{path.stem}",
        source_ref={"dataset_name": "amass", "local_path": str(path), "extractor": "smpl_fk"},
        license_state=license_state,
        fps=float(fps),
        duration=float(n / fps),
        skeleton=Skeleton(joint_names=list(JOINT_NAMES), parents=list(PARENTS)),
        root_trajectory={"position": kps[:, index_of("pelvis"), :].tolist()},
        keypoints_3d=kps.tolist(),
        contacts=contacts,
        privacy_flags={"synthetic": False},
        extractor_versions={"source": "amass_smpl", "adapter": "robotdance.data.amass.v0"},
        semantics={"action_label": "unknown", "source_dataset": "amass"},
    )


def _estimate_contacts(kps: np.ndarray) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for side in ("left", "right"):
        z = kps[:, index_of(f"{side}_ankle"), 2]
        out[f"{side}_foot"] = (z < float(z.min()) + 0.07).tolist()
    return out
