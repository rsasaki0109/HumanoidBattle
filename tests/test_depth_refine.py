"""深度(前後 x)精緻化（balance_depth_refine）の検証。

単眼で ill-posed な前後 x 深度だけを quasi-static balance prior で補正する。確認項目:
(1) 観測軸 y・z を一切変えない（画像面凍結）、(2) COM_x を支持多角形の x 重心へ近づける、
(3) 入力を破壊しない、(4) 誘発 bone 長歪みが小さく報告される。sim/mujoco は不要。
"""

from __future__ import annotations

import numpy as np

from robotdance_core.skeleton import FOOT_JOINTS, index_of
from robotdance_core.synthetic import generate_squat
from robotdance_motion.depth_refine import balance_depth_refine, _mass_weights


def _lean_forward(mir, dx: float = 0.25):
    """上体（骨盤より上）を前後 x に dx ずらして COM_x を支持から外す（深度誤差を注入）。"""
    kps = mir.keypoints_3d_array().copy()
    upper = [index_of(n) for n in
             ("pelvis", "spine", "chest", "neck", "head",
              "left_shoulder", "left_elbow", "left_wrist",
              "right_shoulder", "right_elbow", "right_wrist")]
    kps[:, upper, 0] += dx
    mir.keypoints_3d = kps.tolist()
    return mir


def _com_support_gap(mir) -> float:
    kps = mir.keypoints_3d_array()
    w = _mass_weights()
    com_x = (kps[:, :, 0] * w[None, :]).sum(axis=1)
    sx = np.mean([kps[:, a, 0] for side in ("left", "right")
                  for a in FOOT_JOINTS[side]], axis=0)
    return float(np.mean(np.abs(com_x - sx)))


def test_refine_reduces_com_support_gap() -> None:
    mir = _lean_forward(generate_squat(duration=2.0), dx=0.25)
    gap0 = _com_support_gap(mir)
    refined = balance_depth_refine(mir, strength=0.6, smooth=False)
    gap1 = _com_support_gap(refined)
    assert gap1 < gap0  # COM_x が支持重心へ寄った
    dr = refined.quality_metrics["depth_refine"]
    assert dr["com_support_x_gap_after_m"] < dr["com_support_x_gap_before_m"]


def test_refine_freezes_observed_yz() -> None:
    mir = _lean_forward(generate_squat(duration=1.5), dx=0.2)
    before = mir.keypoints_3d_array().copy()
    refined = balance_depth_refine(mir, strength=0.5)
    after = refined.keypoints_3d_array()
    # 観測軸 y・z は厳密に不変、x のみが変化する。
    assert np.allclose(after[:, :, 1], before[:, :, 1], atol=1e-9)
    assert np.allclose(after[:, :, 2], before[:, :, 2], atol=1e-9)
    assert not np.allclose(after[:, :, 0], before[:, :, 0])


def test_refine_bone_drift_bounded_and_reported() -> None:
    mir = _lean_forward(generate_squat(duration=1.5), dx=0.2)
    refined = balance_depth_refine(mir, strength=0.5, max_shear=0.3)
    dr = refined.quality_metrics["depth_refine"]
    # せん断補正なので bone 長歪みは小さい（2 次オーダー）。
    assert dr["induced_bone_length_drift"] < 0.1
    assert dr["applied"] is True
    assert 0.0 <= dr["grounded_frame_ratio"] <= 1.0


def test_refine_does_not_mutate_input() -> None:
    mir = _lean_forward(generate_squat(duration=1.5), dx=0.2)
    before = mir.keypoints_3d_array().copy()
    _ = balance_depth_refine(mir)
    assert np.array_equal(mir.keypoints_3d_array(), before)  # deep copy


def test_mass_weights_normalized() -> None:
    w = _mass_weights()
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert (w >= 0).all()
