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


def _foot_skate(kps: np.ndarray, foot_z: dict[str, np.ndarray], band: float) -> float:
    """接地中の足の水平移動量（frame 間 xy 変位）の平均 [m]。foot-skate の指標。"""
    disp = []
    for side in _FEET:
        fi = index_of(f"{side}_foot")
        contact = foot_z[side] < band
        for t in range(1, kps.shape[0]):
            if contact[t] and contact[t - 1]:
                disp.append(float(np.linalg.norm(kps[t, fi, :2] - kps[t - 1, fi, :2])))
    return float(np.mean(disp)) if disp else 0.0


def _remove_foot_skate(kps: np.ndarray, band: float) -> np.ndarray:
    """接地足の水平滑り（foot-skate）を除去する。z 接地済み kps を受け取り xy を補正して返す。

    各フレームで支持足（接地中・より低い方, ヒステリシス付き）を選び、その**補正後 xy が一定**に
    なるよう全関節 xy を平行移動する。支持足が切替わる時は飛びを避けて再アンカーする。grounded
    前提（常にどちらかが接地）。深度（前後 x）誤差そのものは直さない＝balance は別軸の課題。
    """
    foot_z = _foot_floor_z(kps)
    raw = kps[:, :, :2].copy()  # 補正前 xy
    idx = {s: index_of(f"{s}_foot") for s in _FEET}
    out = kps.copy()
    cum = np.zeros(2)
    anchor: np.ndarray | None = None
    prev: str | None = None
    for t in range(kps.shape[0]):
        contact = [s for s in _FEET if foot_z[s][t] < band]
        if not contact:
            out[t, :, :2] += cum  # 接地なし: 直前の補正を保持
            continue
        # ヒステリシス: 直前支持足が接地中ならそれを継続、なければ最下足。
        support = prev if (prev in contact) else min(contact, key=lambda s: foot_z[s][t])
        cur = raw[t, idx[support]]
        if support != prev or anchor is None:
            anchor = cur + cum  # 切替/初接地: 補正後現在位置を新アンカーに（飛び無し）
        else:
            cum = anchor - cur  # 補正後支持足 = anchor になるよう平行移動
        out[t, :, :2] += cum
        prev = support
    return out


def ground_contact_cleanup(
    mir: RdMir, *, contact_band: float = 0.06, smooth: bool = True,
    lock_horizontal: bool = False,
) -> RdMir:
    """接地足を毎フレーム z=0 に固定し、接地フラグを高さから再生成した RD-MIR を返す。

    contact_band: 床（その frame の最下足）から何 m 以内を接地とみなすか。
    smooth: 固定後に Savitzky-Golay で再平滑するか（単眼ジッタ由来の COM 加速度スパイク低減）。
    lock_horizontal: True なら接地足の水平滑り（foot-skate）も除去する（opt-in）。深度（前後 x）
        誤差は直さないため balance への効果は限定的だが、planted foot の滑りは消える。
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
    skate_before = _foot_skate(kps, foot_z, contact_band)
    skate_after = skate_before
    if lock_horizontal:
        kps = _remove_foot_skate(kps, contact_band)
        skate_after = _foot_skate(kps, _foot_floor_z(kps), contact_band)

    foot_z = _foot_floor_z(kps)
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
        "lock_horizontal": bool(lock_horizontal),
        "foot_skate_before_m": round(skate_before, 4),
        "foot_skate_after_m": round(skate_after, 4),
    }
    out.quality_metrics = q
    return out
