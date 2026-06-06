"""接地クリーンアップ（ground_contact_cleanup）の検証。

単眼抽出の airborne 誤検出を解消する foot-locking が、(1) 接地足を毎フレーム z=0 へ固定し、
(2) 接地フラグを再生成し、(3) 入力を破壊しないことを確認する。sim/mujoco は不要。
"""

from __future__ import annotations

import numpy as np

from robotdance_core.skeleton import index_of
from robotdance_core.synthetic import generate_squat
from robotdance_motion.grounding import ground_contact_cleanup


def _min_foot_z(mir) -> np.ndarray:
    kps = mir.keypoints_3d_array()
    fz = np.minimum(
        np.minimum(kps[:, index_of("left_ankle"), 2], kps[:, index_of("left_foot"), 2]),
        np.minimum(kps[:, index_of("right_ankle"), 2], kps[:, index_of("right_foot"), 2]),
    )
    return fz


def test_cleanup_grounds_lowest_foot_each_frame() -> None:
    mir = generate_squat(duration=2.0)
    kps = mir.keypoints_3d_array()
    # 浮遊アーティファクトを注入: 全身を時間変動する高さだけ持ち上げる。
    lift = 0.3 + 0.2 * np.sin(np.linspace(0, 6, kps.shape[0]))
    kps[:, :, 2] += lift[:, None]
    mir.keypoints_3d = kps.tolist()
    assert _min_foot_z(mir).min() > 0.1  # 浮いている

    cleaned = ground_contact_cleanup(mir, smooth=False)
    fz = _min_foot_z(cleaned)
    # 接地足が毎フレーム z≈0（持ち上げが除去された）。
    assert np.allclose(fz, 0.0, atol=1e-6)


def test_cleanup_regenerates_contacts_and_metrics() -> None:
    mir = generate_squat(duration=2.0)
    cleaned = ground_contact_cleanup(mir)
    assert set(cleaned.contacts) == {"left_foot", "right_foot"}
    n = cleaned.keypoints_3d_array().shape[0]
    assert len(cleaned.contacts["left_foot"]) == n
    # grounded 動作なので大半のフレームでどちらかの足が接地。
    grounded = np.mean([lc or rc for lc, rc in
                        zip(cleaned.contacts["left_foot"], cleaned.contacts["right_foot"])])
    assert grounded > 0.9
    gc = cleaned.quality_metrics["ground_cleanup"]
    assert gc["applied"] is True and 0.0 <= gc["grounded_foot_frame_ratio"] <= 1.0


def test_cleanup_does_not_mutate_input() -> None:
    mir = generate_squat(duration=1.5)
    before = mir.keypoints_3d_array().copy()
    _ = ground_contact_cleanup(mir)
    assert np.array_equal(mir.keypoints_3d_array(), before)  # deep copy


def test_lock_horizontal_removes_foot_skate() -> None:
    """接地足を水平に滑らせたモーションで、lock_horizontal が skate を大幅に減らす。"""
    mir = generate_squat(duration=2.0)
    kps = mir.keypoints_3d_array()
    n = kps.shape[0]
    # foot-skate を注入: 全身を x 方向へ一定速度でドリフトさせる（足は接地のまま滑る）。
    drift = np.linspace(0, 0.5, n)  # 0.5 m を全フレームかけて
    kps[:, :, 0] += drift[:, None]
    mir.keypoints_3d = kps.tolist()

    locked = ground_contact_cleanup(mir, smooth=False, lock_horizontal=True)
    gc = locked.quality_metrics["ground_cleanup"]
    assert gc["lock_horizontal"] is True
    # skate が大きく減る（接地足の frame 間水平移動が縮む）。
    assert gc["foot_skate_after_m"] < 0.5 * gc["foot_skate_before_m"] + 1e-9
    assert gc["foot_skate_before_m"] > gc["foot_skate_after_m"]


def test_lock_horizontal_off_by_default_preserves_xy() -> None:
    """既定（lock_horizontal=False）では xy を変えない（z 接地のみ）。"""
    mir = generate_squat(duration=1.5)
    base = ground_contact_cleanup(mir, smooth=False)  # default False
    kps = base.keypoints_3d_array()
    raw = mir.keypoints_3d_array()
    # xy は z 接地では不変（z のみオフセット）。
    assert np.allclose(kps[:, :, :2], raw[:, :, :2], atol=1e-9)
    gc = base.quality_metrics["ground_cleanup"]
    assert gc["lock_horizontal"] is False
    assert gc["foot_skate_before_m"] == gc["foot_skate_after_m"]
