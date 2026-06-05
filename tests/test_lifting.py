"""2D COCO-17 → canonical 3D の解析的 planar lift（robotdance_perception.lifting）。

純粋な幾何計算のみ。heavy 依存・動画なしで CI 完結する。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotdance_core.skeleton import NUM_JOINTS, index_of
from robotdance_perception.lifting import (
    DEFAULT_TORSO_M,
    compose_canonical_coco,
    lift_coco17_to_canonical,
)


def _synthetic_coco17_standing(img_h: int = 480, img_w: int = 320) -> np.ndarray:
    """正面直立の合成 COCO-17（画像 px, x:右 / y:下）。"""
    cx = img_w / 2
    xy = np.zeros((17, 2))
    # y は上から下へ増える（画像座標）。頭=上、足=下。
    xy[0] = [cx, 60]            # nose
    xy[1] = [cx - 12, 52]      # l_eye
    xy[2] = [cx + 12, 52]      # r_eye
    xy[3] = [cx - 24, 56]      # l_ear
    xy[4] = [cx + 24, 56]      # r_ear
    xy[5] = [cx - 50, 130]     # l_shoulder
    xy[6] = [cx + 50, 130]     # r_shoulder
    xy[7] = [cx - 60, 200]     # l_elbow
    xy[8] = [cx + 60, 200]     # r_elbow
    xy[9] = [cx - 65, 270]     # l_wrist
    xy[10] = [cx + 65, 270]    # r_wrist
    xy[11] = [cx - 30, 280]    # l_hip
    xy[12] = [cx + 30, 280]    # r_hip
    xy[13] = [cx - 32, 360]    # l_knee
    xy[14] = [cx + 32, 360]    # r_knee
    xy[15] = [cx - 33, 440]    # l_ankle
    xy[16] = [cx + 33, 440]    # r_ankle
    return xy


def test_compose_canonical_coco_shape_and_interp():
    xy = _synthetic_coco17_standing()
    can = compose_canonical_coco(xy)
    assert can.shape == (NUM_JOINTS, 2)
    # pelvis は左右 hip の中点。
    mid = 0.5 * (xy[11] + xy[12])
    assert np.allclose(can[index_of("pelvis")], mid)


def test_lift_is_planar_zero_depth():
    xy = _synthetic_coco17_standing()
    conf = np.ones(17)
    kps, c = lift_coco17_to_canonical(xy, conf)
    assert kps.shape == (NUM_JOINTS, 3)
    assert c.shape == (NUM_JOINTS,)
    # 深度（前後 x）は復元しない → 全関節 x=0。
    assert np.allclose(kps[:, 0], 0.0)


def test_lift_grounds_feet_and_orders_height():
    xy = _synthetic_coco17_standing()
    kps, _ = lift_coco17_to_canonical(xy, np.ones(17))
    # 足は z=0 接地、頭は膝より高い、膝は接地より高い。
    foot_z = min(kps[index_of("left_foot"), 2], kps[index_of("right_foot"), 2])
    assert foot_z == pytest.approx(0.0, abs=1e-9)
    assert kps[index_of("head"), 2] > kps[index_of("left_knee"), 2] > 0.0


def test_lift_metric_scale_matches_torso_prior():
    # 胴体長（pelvis→chest）が torso_m プライアに一致するよう metric 化される。
    xy = _synthetic_coco17_standing()
    kps, _ = lift_coco17_to_canonical(xy, np.ones(17), torso_m=DEFAULT_TORSO_M)
    torso = np.linalg.norm(kps[index_of("pelvis")] - kps[index_of("chest")])
    assert torso == pytest.approx(DEFAULT_TORSO_M, rel=1e-6)


def test_lift_uses_native_handedness_image_right_to_negative_y():
    # native MediaPipe と同手系: 画像右側（cx 右＝x 大）の点は -y 側に来る。
    xy = _synthetic_coco17_standing()
    kps, _ = lift_coco17_to_canonical(xy, np.ones(17))
    # COCO r_shoulder は画像右（cx+50）→ canonical -y。l_shoulder は +y。
    assert kps[index_of("right_shoulder"), 1] < kps[index_of("left_shoulder"), 1]


def test_lift_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\[17,2\]"):
        lift_coco17_to_canonical(np.zeros((19, 2)), np.ones(19))


def test_lift_handles_degenerate_zero_torso():
    # 胴体長 0px でも 0 割せず有限値を返す（scale=0 → 原点崩壊）。
    xy = np.zeros((17, 2))
    kps, _ = lift_coco17_to_canonical(xy, np.ones(17))
    assert np.all(np.isfinite(kps))
