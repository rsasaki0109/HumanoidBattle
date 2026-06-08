"""単眼抽出 motion の深度(前後 x)精緻化 — quasi-static balance prior。

単眼 3D lifting は**画像面（横 y・高さ z）はよく復元するが、カメラ光軸方向＝前後(x)の深度が
ill-posed**（特に正面・近接の被写体）。feasibility certificate の balance 違反（ZMP が支持多角形外）は
この前後 x のズレに支配される（[[real-video-demo-pipeline]] の v0.75/v0.76 で実証: airborne は
接地 cleanup で消えても balance は残り、残差 ZMP は前後 x に偏る）。

`balance_depth_refine` は接地（quasi-static）動作を前提に、**観測軸 y・z を一切変えず**、未観測の
x だけを「接地動作なら COM_x は支持多角形の x 重心近傍にあるべき」という物理事前分布で精緻化する。

⚠️ over-smoothing で見かけ PASS にする gimmick ではない（[[real-urdf-deepdive-thread]] の
ankle-strategy 却下・[[real-video-demo-pipeline]] v0.76 の過平滑不採用と同じ方針）。本手法は
**本質的に未観測な自由度（前後深度）だけ**を物理事前分布で解く。観測できている画像面（y,z）は
凍結し、violation を平滑で隠さない。誤魔化しと精緻化の境界はここにある。

幾何モデル: 補正は **足首ピボットの前後リーン**を近似する x-z せん断 `x' = x + k·z`。床(z≈0)の
接地足はほぼ動かず（接地アンカー保存）、高い関節ほど大きく前後する＝前傾/後傾。せん断は bone 長を
2 次でしか変えないので、`k` を `max_shear` で制限すれば歪みは小さい（誘発 bone 長変化を報告する）。

⚠️ v0 の前提:
- **接地（quasi-static）前提**。両足滞空フレーム（airborne）は補正対象外でそのまま。跳躍/走行など
  動的に COM が支持外へ出るのが正常な動作には適用しない。
- COM は anthropometric な segment 質量（Winter 近似, joint 集中質点）で算出。被写体の体格差は無視。
- `strength` は事前分布の強さ（0=無補正, 1=毎フレーム COM_x を支持重心へ完全一致）。既定 0.5。
"""

from __future__ import annotations

import numpy as np

from robotdance_core.rd_mir import RdMir
from robotdance_core.skeleton import BONES, FOOT_JOINTS, JOINT_NAMES, index_of
from robotdance_motion.grounding import _foot_floor_z
from robotdance_motion.smoothing import savgol_smooth

# anthropometric segment 質量比（Winter, Biomechanics 近似）を canonical 19 joint に集中質点として
# 割り当てる。和はおよそ 1（コード側で正規化）。前後 COM の位置決めに使う。
_SEGMENT_MASS: dict[str, float] = {
    "pelvis": 0.142, "spine": 0.139, "chest": 0.216, "neck": 0.0, "head": 0.081,
    "left_shoulder": 0.028, "left_elbow": 0.016, "left_wrist": 0.006,
    "right_shoulder": 0.028, "right_elbow": 0.016, "right_wrist": 0.006,
    "left_hip": 0.100, "left_knee": 0.0465, "left_ankle": 0.0145, "left_foot": 0.0,
    "right_hip": 0.100, "right_knee": 0.0465, "right_ankle": 0.0145, "right_foot": 0.0,
}


def _mass_weights() -> np.ndarray:
    w = np.array([_SEGMENT_MASS[n] for n in JOINT_NAMES], dtype=float)
    return w / w.sum()


def _support_center_x(kps_f: np.ndarray, contact_f: dict[str, bool]) -> float | None:
    """接地足（ankle+toe）の x 平均＝支持多角形の前後重心。接地ゼロなら None。"""
    xs: list[float] = []
    for side, (ankle, toe) in FOOT_JOINTS.items():
        if contact_f.get(f"{side}_foot", False):
            xs.append(float(kps_f[ankle, 0]))
            xs.append(float(kps_f[toe, 0]))
    return float(np.mean(xs)) if xs else None


def _bone_length_drift(before: np.ndarray, after: np.ndarray) -> float:
    """補正前後の bone 長の平均相対変化（誘発歪みの正直な指標）[-]。"""
    drifts: list[float] = []
    for j, p in BONES:
        lb = np.linalg.norm(before[:, j] - before[:, p], axis=1)
        la = np.linalg.norm(after[:, j] - after[:, p], axis=1)
        ok = lb > 1e-6
        if ok.any():
            drifts.append(float(np.mean(np.abs(la[ok] - lb[ok]) / lb[ok])))
    return float(np.mean(drifts)) if drifts else 0.0


def balance_depth_refine(
    mir: RdMir, *, strength: float = 0.5, contact_band: float = 0.06,
    max_shear: float = 0.3, smooth: bool = True,
) -> RdMir:
    """未観測の前後 x 深度のみを quasi-static balance prior で精緻化した RD-MIR を返す。

    strength: 事前分布の強さ（0..1）。COM_x を支持重心へ寄せる割合。
    contact_band: 床（最下足）から何 m 以内を接地とみなすか（接地フラグが無い場合の再判定に使用）。
    max_shear: x-z せん断係数 k の絶対値上限（bone 長歪みを抑える安全弁）。
    smooth: 補正係数 k(t) を時間方向に Savitzky-Golay 平滑して frame 間ジッタを避けるか。
    観測軸 y・z は変更しない。入力 mir は変更せず deep copy を返す。
    """
    if mir.keypoints_3d is None:
        raise ValueError("keypoints_3d が無いため深度精緻化できません")
    out = mir.model_copy(deep=True)
    kps = out.keypoints_3d_array().copy()  # [T, J, 3]
    n = kps.shape[0]
    before = kps.copy()
    w = _mass_weights()                    # [J]

    # 床高さ（per-frame の最下足 z）。接地足の z 接触高さは grounding._foot_floor_z に集約。
    foot_z = _foot_floor_z(kps)                               # {side: [T]}
    floor = np.minimum(foot_z["left"], foot_z["right"])       # [T]

    # 接地判定: 既存 contacts があればそれを、無ければ最下足 + band で再生成。[T] bool 配列に正規化。
    src = out.contacts or {}
    if src:
        contact_arr = {key: np.asarray(v, dtype=bool) for key, v in src.items()}
    else:
        contact_arr = {f"{s}_foot": (foot_z[s] - floor < contact_band) for s in ("left", "right")}

    # 入力が接地正規化されていなくても床上高さ（z − floor）でせん断する。これにより床が z=0 で
    # ない（--ground-clean 未適用や生抽出）入力でも接地足(z≈floor)は不動を保つ＝足首ピボットの
    # 前後リーン近似が成立する。床下は 0 にクランプ。
    z = np.clip(kps[:, :, 2] - floor[:, None], 0.0, None)     # [T, J] 床上高さ
    com_x = (kps[:, :, 0] * w[None, :]).sum(axis=1)           # [T]

    def _support_x(f: int) -> float | None:
        cf = {key: bool(arr[f]) for key, arr in contact_arr.items()}
        return _support_center_x(kps[f], cf)

    gap_before: list[float] = []
    gap_after: list[float] = []
    k = np.zeros(n)
    grounded = 0
    for f in range(n):
        sx = _support_x(f)
        if sx is None:        # airborne: 補正対象外
            continue
        grounded += 1
        gap_before.append(abs(com_x[f] - sx))
        target_shift = strength * (sx - com_x[f])             # COM_x をこれだけ動かしたい
        denom = float((w * z[f]).sum())                       # Σ w_j (z_j−floor)（せん断の COM 感度）
        if denom > 1e-6:
            k[f] = float(np.clip(target_shift / denom, -max_shear, max_shear))

    if smooth and n >= 5:
        # k(t) のみ平滑（motion そのものではなく補正係数を band-limit）。
        ks = savgol_smooth(k.reshape(-1, 1, 1)).reshape(-1)
        k = ks

    # x' = x + k·(z−floor)（y,z は不変）。床の接地足(z≈floor)はほぼ不動。
    kps[:, :, 0] = kps[:, :, 0] + k[:, None] * z

    # 補正後の COM_x で残差 gap を測る（support_x は接地足 x — せん断後の値で再評価）。
    com_x2 = (kps[:, :, 0] * w[None, :]).sum(axis=1)
    for f in range(n):
        sx = _support_x(f)
        if sx is not None:
            gap_after.append(abs(com_x2[f] - sx))

    out.keypoints_3d = kps.tolist()
    out.root_trajectory = {"position": kps[:, index_of("pelvis"), :].tolist()}
    drift = _bone_length_drift(before, kps)
    q = dict(out.quality_metrics or {})
    q["depth_refine"] = {
        "applied": True,
        "strength": strength,
        "max_shear": max_shear,
        "smoothed": bool(smooth),
        "grounded_frame_ratio": round(grounded / n, 3) if n else 0.0,
        "com_support_x_gap_before_m": round(float(np.mean(gap_before)), 4) if gap_before else 0.0,
        "com_support_x_gap_after_m": round(float(np.mean(gap_after)), 4) if gap_after else 0.0,
        "induced_bone_length_drift": round(drift, 4),
    }
    out.quality_metrics = q
    return out


__all__ = ["balance_depth_refine"]
