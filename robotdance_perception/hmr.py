"""HMR（Human Mesh Recovery）adapter: SMPL 出力 → canonical RD-MIR（skeleton-first, v0）。

MediaPipe Pose（2D→近似 3D landmark）に対し、4DHumans / GVHMR 等の **HMR モデルは画像から
SMPL body パラメータ（per-frame の global_orient / body_pose / transl）を回帰**し、
オクルージョンや奥行き・世界座標の global trajectory に強い。本 adapter はその SMPL 出力を
**既存の skeleton-first SMPL FK**（`robotdance_data.smpl`）で canonical 19-joint に変換する。

設計方針:
  - **モデルの weight / SMPL body model file は同梱・実行しない**（license-safe）。HMR ツールが
    出力した SMPL パラメータ（axis-angle でも rotation-matrix でも可）を受け取って変換するだけ。
  - 4DHumans（HMR2.0/PHALP, rotmat 出力）と GVHMR（axis-angle・world-grounded）の双方の
    **出力構造**に対応する entry point を用意し、core は両者を共通の SMPL pose に正規化して処理する。
  - in-the-wild 動画由来なので `license_state` 既定は "unknown"（派生 motion を勝手に公開しない）。

⚠️ v0: skeleton-first（近似 rest offset・betas/shape は未使用）。特定モデル版に pin した検証では
なく**文書化された出力構造**に対する検証。HMR 推論そのもの（動画→SMPL）はツール側が担う。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from robotdance_core.rd_mir import LicenseState, RdMir, Skeleton
from robotdance_core.skeleton import JOINT_NAMES, PARENTS, index_of

from robotdance_data.smpl import smpl_poses_to_canonical


def _to_axis_angle(arr: np.ndarray) -> np.ndarray:
    """SMPL pose を axis-angle に正規化する（joint 次元は保持）。

    受理: rotation-matrix [...,3,3] / axis-angle [...,3] / flatten axis-angle [...,3K]。
    返り: axis-angle で最後の軸が 3（rotmat と 3K は joint 軸を 1 つ増やして展開）。
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[-2:] == (3, 3):
        flat = arr.reshape(-1, 3, 3)
        aa = Rot.from_matrix(flat).as_rotvec()
        return aa.reshape(arr.shape[:-2] + (3,))
    if arr.shape[-1] == 3:
        return arr
    if arr.shape[-1] % 3 == 0:  # flatten axis-angle [..., 3K] → [..., K, 3]
        return arr.reshape(arr.shape[:-1] + (arr.shape[-1] // 3, 3))
    raise ValueError(f"axis-angle/rotmat いずれでもない形状: {arr.shape}")


def hmr_smpl_to_mir(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: Optional[np.ndarray] = None,
    *,
    betas: Optional[np.ndarray] = None,
    fps: float = 30.0,
    source: str = "hmr",
    license_state: LicenseState = "unknown",
    motion_id: Optional[str] = None,
    source_ref: Optional[dict[str, Any]] = None,
    smooth: bool = True,
) -> RdMir:
    """HMR の SMPL 出力（per-frame）を canonical RD-MIR に変換する。

    global_orient: [T,3]（axis-angle）または [T,3,3]（rotmat）= pelvis の向き。
    body_pose:     [T,21,3] / [T,23,3] / [T,63] / [T,69]（axis-angle）または対応する rotmat。
                   先頭 21 body joint（SMPL joint 1..21）だけを使う。
    transl:        [T,3] root 並進（あれば world trajectory として反映）。
    betas:         SMPL shape（あれば rest offset を shape-conditioning, v0 近似）。
    """
    go = _to_axis_angle(global_orient).reshape(-1, 3)        # [T,3]
    bp = _to_axis_angle(body_pose)
    bp = bp.reshape(bp.shape[0], -1, 3)[:, :21, :]            # [T,21,3]
    if go.shape[0] != bp.shape[0]:
        raise ValueError(f"フレーム数不一致: global_orient {go.shape[0]} vs body_pose {bp.shape[0]}")
    poses = np.concatenate([go[:, None, :], bp], axis=1)     # [T,22,3]
    trans = np.asarray(transl, dtype=np.float64) if transl is not None else None

    betas_v = _mean_betas(betas)
    kps = smpl_poses_to_canonical(poses, trans, betas_v)     # [T,19,3]（z-up canonical）
    # 接地: 足の最下点を z=0 へ。
    kps[:, :, 2] -= kps[:, [index_of("left_ankle"), index_of("right_ankle")], 2].min()

    quality: dict[str, Any] = {"extractor": "hmr_smpl_fk",
                               "shape_conditioned": betas_v is not None}
    if smooth and kps.shape[0] >= 7:
        from robotdance_motion.smoothing import jitter, savgol_smooth

        quality["jitter_before"] = round(jitter(kps), 5)
        kps = savgol_smooth(kps)
        quality["jitter_after"] = round(jitter(kps), 5)
        quality["smoothing"] = "savgol(window=7,polyorder=2)"

    n = kps.shape[0]
    return RdMir(
        motion_id=motion_id or f"rdmir-hmr-{source}",
        source_ref=source_ref or {"extractor": f"hmr_{source}", "smpl_fk": True},
        license_state=license_state,
        fps=float(fps),
        duration=float(n / fps),
        world_frame={"up_axis": "z", "forward_axis": "x", "handedness": "right"},
        skeleton=Skeleton(joint_names=list(JOINT_NAMES), parents=list(PARENTS)),
        root_trajectory={"position": kps[:, index_of("pelvis"), :].tolist()},
        keypoints_3d=kps.tolist(),
        contacts=_estimate_contacts(kps),
        privacy_flags={"face_visible": True, "synthetic": False},
        quality_metrics=quality,
        extractor_versions={"hmr": source, "adapter": "robotdance.perception.hmr.v0"},
        semantics={"action_label": "unknown", "source": source},
    )


def from_gvhmr(
    result: dict[str, Any], *, world: bool = True, fps: float = 30.0, **kw: Any
) -> RdMir:
    """GVHMR の出力 dict（`smpl_params_global` / `_incam`）→ RD-MIR。

    GVHMR は世界座標で grounded な global trajectory を出すのが強み（world=True で global を使う）。
    各 params dict は `global_orient`[T,3], `body_pose`[T,63], `transl`[T,3]（axis-angle）を持つ。
    """
    key = "smpl_params_global" if world else "smpl_params_incam"
    params = result.get(key) or result.get("smpl_params") or result
    go = _as_np(params["global_orient"])
    bp = _as_np(params["body_pose"])
    transl = _as_np(params["transl"]) if "transl" in params else None
    betas = _as_np(params["betas"]) if "betas" in params else None
    return hmr_smpl_to_mir(go, bp, transl, betas=betas, fps=fps, source="gvhmr",
                           source_ref={"extractor": "gvhmr", "frame": "global" if world else "incam"},
                           **kw)


def from_4dhumans(result: dict[str, Any], *, fps: float = 30.0, **kw: Any) -> RdMir:
    """4DHumans（HMR2.0/PHALP）の consolidated track dict → RD-MIR。

    HMR2.0 は rotation-matrix で SMPL を出す: `global_orient`[T,3,3] / `body_pose`[T,23,3,3]。
    `pred_cam_t`[T,3] があれば近似 root 並進として使う（v0, weak-perspective camera 近似）。
    複数トラックがある場合は呼び出し側で 1 トラックに整えてから渡す。
    """
    smpl = result.get("smpl") or result.get("pred_smpl_params") or result
    go = _as_np(smpl["global_orient"])
    bp = _as_np(smpl["body_pose"])
    transl = _as_np(result["pred_cam_t"]) if "pred_cam_t" in result else None
    betas = _as_np(smpl["betas"]) if "betas" in smpl else None
    return hmr_smpl_to_mir(go, bp, transl, betas=betas, fps=fps, source="4dhumans",
                           source_ref={"extractor": "4dhumans_hmr2", "camera": "weak_perspective"},
                           **kw)


def load_hmr_npz(path: str | Path, *, source: str = "hmr", fps: Optional[float] = None,
                 **kw: Any) -> RdMir:
    """汎用 HMR 交換フォーマット（.npz）→ RD-MIR。

    HMR ツールの出力を一度 .npz（keys: global_orient, body_pose, 任意で transl/fps/source）へ
    書き出しておけば、本ローダで RD-MIR 化できる。axis-angle / rotmat は形状から自動判別。
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if "global_orient" not in data or "body_pose" not in data:
        raise ValueError(f"HMR npz に global_orient/body_pose がありません: {path}")
    transl = np.asarray(data["transl"]) if "transl" in data else None
    betas = np.asarray(data["betas"]) if "betas" in data else None
    src = str(data["source"]) if "source" in data else source
    f = fps if fps is not None else (float(np.asarray(data["fps"]).item()) if "fps" in data else 30.0)
    return hmr_smpl_to_mir(
        np.asarray(data["global_orient"]), np.asarray(data["body_pose"]), transl, betas=betas,
        fps=f, source=src, source_ref={"extractor": f"hmr_{src}", "local_path": str(path)}, **kw,
    )


def from_dict(result: dict[str, Any], **kw: Any) -> RdMir:
    """構造から HMR ツールを判別して RD-MIR 化する（GVHMR / 4DHumans / 汎用）。"""
    if "smpl_params_global" in result or "smpl_params_incam" in result:
        kw.pop("source", None)  # GVHMR は source を自前設定
        return from_gvhmr(result, **kw)
    if "smpl" in result or "pred_smpl_params" in result:
        kw.pop("source", None)  # 4DHumans は source を自前設定
        return from_4dhumans(result, **kw)
    if "global_orient" in result and "body_pose" in result:
        transl = _as_np(result["transl"]) if "transl" in result else None
        betas = _as_np(result["betas"]) if "betas" in result else None
        return hmr_smpl_to_mir(_as_np(result["global_orient"]), _as_np(result["body_pose"]),
                               transl, betas=betas, source=kw.pop("source", "hmr"), **kw)
    raise ValueError("HMR dict から SMPL パラメータを判別できません "
                     "（smpl_params_global / smpl / global_orient+body_pose のいずれかが必要）")


def load_hmr_file(path: str | Path, **kw: Any) -> RdMir:
    """HMR ツールの native 出力（.npz / .npy / .pkl / .pt）→ RD-MIR（構造を自動判別）。

    .pkl は pickle、.pt は torch でロードし、dict なら `from_dict` で GVHMR/4DHumans/汎用を判別する。
    .npz は `load_hmr_npz`、.npy は [T,D]/dict を試す。**weights/モデルは不要**（出力のみ読む）。
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npz":
        return load_hmr_npz(path, **kw)
    if ext == ".pkl":
        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)
        return from_dict(obj, **kw) if isinstance(obj, dict) else _from_array(obj, path, **kw)
    if ext == ".pt":
        import torch

        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        return from_dict(obj, **kw) if isinstance(obj, dict) else _from_array(_as_np(obj), path, **kw)
    if ext == ".npy":
        obj = np.load(path, allow_pickle=True)
        if obj.dtype == object and obj.shape == ():
            return from_dict(obj.item(), **kw)
        return _from_array(obj, path, **kw)
    raise ValueError(f"未対応の HMR ファイル形式: {ext}（.npz/.npy/.pkl/.pt）")


def _from_array(arr: Any, path: "Path", **kw: Any) -> RdMir:
    """[T, D] 配列（>=66 次元: root_orient+body_pose を連結）→ RD-MIR。"""
    a = _as_np(arr)
    if a.ndim != 2 or a.shape[1] < 66:
        raise ValueError(f"HMR 配列は [T, >=66] が必要: {a.shape}")
    return hmr_smpl_to_mir(a[:, :3], a[:, 3:66],
                           source=kw.pop("source", "hmr"),
                           source_ref={"extractor": "hmr_array", "local_path": str(path)}, **kw)


def _mean_betas(betas: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """betas が per-frame [T,B] でも [B] でも、shape 用に時間平均した [B] を返す。"""
    if betas is None:
        return None
    b = np.asarray(betas, dtype=np.float64)
    if b.ndim >= 2:
        b = b.reshape(-1, b.shape[-1]).mean(axis=0)
    return b.ravel()


def _as_np(x: Any) -> np.ndarray:
    """torch.Tensor / list / ndarray を float ndarray に変換する（torch 依存なし）。"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _estimate_contacts(kps: np.ndarray) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for side in ("left", "right"):
        z = kps[:, index_of(f"{side}_ankle"), 2]
        out[f"{side}_foot"] = (z < float(z.min()) + 0.07).tolist()
    return out
