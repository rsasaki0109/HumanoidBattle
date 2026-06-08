"""demo-fight / demo-assisted の GMR retarget 配線。"""

from __future__ import annotations

import pytest

from robotdance_retarget.dispatch import check_retarget_backend_for_robots
from robotdance_sim.fight_tracking import fight_tracking_reference


def test_check_retarget_backend_kinematic_always_ok():
    check_retarget_backend_for_robots(["apptronik_apollo", "unitree_g1"], "kinematic")


def test_check_retarget_backend_gmr_unsupported_robot():
    pytest.importorskip("mujoco")
    from robotdance_retarget.gmr_backend import gmr_available

    if not gmr_available():
        pytest.skip("GMR not installed")
    with pytest.raises(ValueError, match="未対応"):
        check_retarget_backend_for_robots(["apptronik_apollo"], "gmr")


def test_fight_tracking_reference_kinematic():
    pytest.importorskip("mujoco")
    ref = fight_tracking_reference(
        "unitree_g1", "boxing", duration=2.0, retarget_backend="kinematic",
    )
    assert ref.keypoints_3d_array().shape[0] >= 10


def test_fight_tracking_reference_gmr_if_available():
    pytest.importorskip("mujoco")
    from robotdance_retarget.gmr_backend import gmr_available

    if not gmr_available():
        pytest.skip("GMR not installed")
    ref = fight_tracking_reference(
        "unitree_g1", "boxing", duration=2.0, retarget_backend="gmr",
    )
    kps = ref.keypoints_3d_array()
    assert kps.shape[0] >= 10
    assert (ref.source_provenance or {}).get("method") == "gmr"


def test_run_fight_retarget_backend_kinematic():
    pytest.importorskip("mujoco")
    from robotdance_sim.arena import run_fight
    from robotdance_unitree import get_morphology

    res = run_fight(
        get_morphology("unitree_g1"), get_morphology("unitree_h1"),
        name_a="unitree_g1", name_b="unitree_h1", duration=2.0, render=False,
        style="boxing", retarget_backend="kinematic",
    )
    assert res.winner in ("unitree_g1", "unitree_h1", "DRAW")


def test_run_fight_retarget_backend_gmr_if_available():
    pytest.importorskip("mujoco")
    from robotdance_retarget.gmr_backend import gmr_available
    from robotdance_sim.arena import run_fight
    from robotdance_unitree import get_morphology

    if not gmr_available():
        pytest.skip("GMR not installed")
    res = run_fight(
        get_morphology("unitree_g1"), get_morphology("unitree_h1"),
        name_a="unitree_g1", name_b="unitree_h1", duration=2.0, render=False,
        style="boxing", retarget_backend="gmr", assisted="p1",
    )
    assert res.assisted_survival is not None
