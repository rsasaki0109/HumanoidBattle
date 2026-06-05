"""単眼抽出 motion の接地クリーンアップ（foot-locking + 再接地）。

単眼 pose 抽出は**根（root）の絶対高さと足接地が不確実**で、両足が地面を離れたように
見える（airborne 誤検出）・足が床を滑る（foot skate）といったアーティファクトが出る。
これは feasibility certificate で airborne/balance 違反として現れ、抽出 motion をそのまま
ロボットに流せない大きな要因になる（[[real-video-demo-pipeline]] の v0.75 で実証）。

`ground_contact_cleanup` は **grounded performance**（接地して行う動作: スクワット・型・
立位ダンス等）を前提に、各フレームで**接地足を地面 z=0 に固定**し、接地フラグを足の高さから
再生成する。これにより airborne 誤検出を解消し、ZMP の鉛直方向ジッタを抑える。

⚠️ v0 の前提と限界:
- **跳躍は未対応**（両足が同時に地面を離れる動作は、最下足を 0 に固定してしまうと滞空が潰れる）。
  接地して行う動作に対する cleanup であり、跳躍を含む motion には適用しない。
- balance（ZMP の水平位置）は主に**単眼の深度（前後 x）誤差**が支配するため、本 cleanup では
  完全には解消しない。接地アーティファクト（airborne）と深度誤差（balance）は別問題で、本 cleanup は
  前者を対象にする。残差は contact-aware retarget / 深度復元の改善という別軸の課題。
"""

from __future__ import annotations

import numpy as np

from robotdance_core.rd_mir import RdMir
from robotdance_core.skeleton import index_of
from robotdance_motion.smoothing import savgol_smooth

_FEET = ("left", "right")


def _foot_floor_z(kps: np.ndarray) -> dict[str, np.ndarray]:
    """各足の床接触高さ = foot(toe) と ankle の低い方の z を返す。"""
    out: dict[str, np.ndarray] = {}
    for side in _FEET:
        ankle_z = kps[:, index_of(f"{side}_ankle"), 2]
        foot_z = kps[:, index_of(f"{side}_foot"), 2]
        out[side] = np.minimum(ankle_z, foot_z)
    return out


def ground_contact_cleanup(
    mir: RdMir, *, contact_band: float = 0.06, smooth: bool = True
) -> RdMir:
    """接地足を毎フレーム z=0 に固定し、接地フラグを高さから再生成した RD-MIR を返す。

    contact_band: 床（その frame の最下足）から何 m 以内を接地とみなすか。
    smooth: 固定後に Savitzky-Golay で再平滑するか（単眼ジッタ由来の COM 加速度スパイク低減）。
    入力 mir は変更せず、deep copy を返す。
    """
    if mir.keypoints_3d is None:
        raise ValueError("keypoints_3d が無いため接地クリーンアップできません")
    out = mir.model_copy(deep=True)
    kps = out.keypoints_3d_array().copy()  # [T, J, 3]

    foot_z = _foot_floor_z(kps)
    floor = np.minimum(foot_z["left"], foot_z["right"])  # frame ごとの最下足高さ
    kps[:, :, 2] -= floor[:, None]  # 接地足を z=0 へ（grounded 前提・跳躍未対応）

    foot_z = _foot_floor_z(kps)  # 固定後の高さで接地判定
    contacts = {f"{side}_foot": (foot_z[side] < contact_band).tolist() for side in _FEET}

    if smooth:
        kps = savgol_smooth(kps)

    out.keypoints_3d = kps.tolist()
    out.root_trajectory = {"position": kps[:, index_of("pelvis"), :].tolist()}
    out.contacts = contacts
    q = dict(out.quality_metrics or {})
    grounded = float(np.mean([c for cs in contacts.values() for c in cs]))
    q["ground_cleanup"] = {
        "applied": True,
        "contact_band_m": contact_band,
        "smoothed": bool(smooth),
        "grounded_foot_frame_ratio": round(grounded, 3),
    }
    out.quality_metrics = q
    return out
