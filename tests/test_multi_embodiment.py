"""multi-embodiment（G1 + H1）と汎用 retarget の検証。"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from robotdance_core.skeleton import NUM_JOINTS
from robotdance_core.synthetic import generate_dance
from robotdance_retarget.kinematic import retarget
from robotdance_unitree import EMBODIMENTS, get_morphology

_ROOT = Path(__file__).resolve().parent.parent
_EMB_SCHEMA = json.loads(
    (_ROOT / "specs" / "rd-embodiment" / "rd-embodiment.schema.json").read_text(encoding="utf-8")
)
_MOTION_SCHEMA = json.loads(
    (_ROOT / "specs" / "rd-motion" / "rd-motion.schema.json").read_text(encoding="utf-8")
)


def test_registry_has_g1_h1_and_t1() -> None:
    assert set(EMBODIMENTS) == {"unitree_g1", "unitree_h1", "booster_t1"}
    assert get_morphology("unitree_g1").name == "unitree_g1"
    with pytest.raises(KeyError):
        get_morphology("nonexistent_robot")


def test_booster_t1_real_urdf_geometry_and_limits() -> None:
    """3 機種目 Booster T1 が実 URDF 由来の geometry / limit / 質量 / 慣性で canonical へ写像される。

    T1 は小型（~0.98m, 31.6kg）で G1/H1 と別ベンダ。機種非依存に全機能（7 軸フル実データ）が機能する
    ことを実証する。real_inertia=True で実 URDF 慣性テンソルを装着できる（G1/H1 と同格）。
    """
    import numpy as np

    m = get_morphology("booster_t1")
    # 実 T1 URDF 総質量・小型 nominal height（G1 より低い）。
    assert m.sim_defaults.total_mass == pytest.approx(31.614, abs=1e-3)
    assert m.nominal_height < get_morphology("unitree_g1").nominal_height
    # 左右対称（rest pose の y 反転で一致）。
    rest = m.rest_pose
    from robotdance_core.skeleton import JOINT_NAMES

    li, ri = JOINT_NAMES.index("left_knee"), JOINT_NAMES.index("right_knee")
    assert rest[li][0] == pytest.approx(rest[ri][0])
    assert rest[li][1] == pytest.approx(-rest[ri][1])
    # 実 effort: 膝(60) > 腕(18)、足首は狭レンジ。bone 長は全て正（root 除く）。
    jl = m.per_joint_limits
    assert jl["left_knee"]["torque"] > jl["left_shoulder"]["torque"]
    assert np.all(m.bone_lengths[1:] > 0.0)
    # real_inertia=True で実 URDF 慣性テンソルが装着される（既定は capsule = None）。
    assert getattr(m, "inertia_tensors", None) in (None, {})
    real = get_morphology("booster_t1", real_inertia=True)
    assert real.inertia_tensors and "left_knee" in real.inertia_tensors


@pytest.mark.parametrize("name", sorted(EMBODIMENTS))
def test_each_embodiment_conforms(name: str) -> None:
    jsonschema.Draft202012Validator(_EMB_SCHEMA).validate(get_morphology(name).to_rd_embodiment())


@pytest.mark.parametrize("name", sorted(EMBODIMENTS))
def test_embodiment_reports_real_joint_limits_not_placeholder(name: str) -> None:
    """既定 embodiment（URDF 無しでも）が実 actuator の joint limit を報告する。

    膝は屈曲のみ（実機は逆屈不可）で placeholder ±3.14 とは別物、合成 toe は placeholder のまま。
    """
    jl = get_morphology(name).to_rd_embodiment()["joint_limits"]
    # 膝は逆屈できない（lower > -0.5）し、placeholder ±3.14 ではない。
    assert jl["left_knee"]["position"][0] > -0.5
    assert jl["left_knee"]["position"] != [-3.14, 3.14]
    # 膝トルクは腕より大きい（実機の事実: 脚の方が強力）。
    assert jl["left_knee"]["torque"] > jl["left_elbow"]["torque"]
    # actuator の無い合成 toe は placeholder のまま（正直に区別）。
    assert jl["left_foot"]["position"] == [-3.14, 3.14]


def test_h1_is_taller_than_g1_and_human() -> None:
    g1_h = get_morphology("unitree_g1").nominal_height
    h1_h = get_morphology("unitree_h1").nominal_height
    assert h1_h > g1_h  # H1 は full-size、G1 は小型

    mir = generate_dance(duration=1.0, fps=30.0)
    h1_motion = retarget(mir, get_morphology("unitree_h1"))
    # H1 は人間より背が高い → height_scale > 1。
    assert h1_motion.retarget_metrics["height_scale"] > 1.0


@pytest.mark.parametrize("name", sorted(EMBODIMENTS))
def test_generic_retarget_shapes_and_schema(name: str) -> None:
    mir = generate_dance(duration=1.0, fps=30.0)
    motion = retarget(mir, get_morphology(name))
    assert motion.robot_name == name
    assert motion.keypoints_3d_array().shape == (30, NUM_JOINTS, 3)
    assert motion.retarget_metrics["bone_direction_cosine"] > 0.99
    jsonschema.Draft202012Validator(_MOTION_SCHEMA).validate(motion.to_dict())


def test_render_three_panels(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from robotdance_viewer.skeleton_view import render_side_by_side

    mir = generate_dance(duration=0.5, fps=20.0)
    panels = [(mir.keypoints_3d_array(), "human", "#1f77b4")]
    for name in ("unitree_g1", "unitree_h1"):
        panels.append((retarget(mir, get_morphology(name)).keypoints_3d_array(), name, "#ff7f0e"))
    out = render_side_by_side(panels, tmp_path / "multi.gif", fps=20.0, stride=2)
    assert out.exists() and out.stat().st_size > 0
