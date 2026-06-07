"""actuator 関節角の実機/シム SDK 向け export（CSV/JSON）のテスト。torch 非依存。"""

from __future__ import annotations

import csv
import json

import pytest

from robotdance_core.rd_mir import Skeleton
from robotdance_core.rd_motion import RdMotion
from robotdance_retarget.sdk_export import export_joint_trajectory, joint_trajectory


def _motion(names, angles, fps=30.0):
    n_frames = max(len(angles), 1)
    return RdMotion(
        robot_name="unitree_g1",
        fps=fps,
        duration=n_frames / fps,
        source_motion_id="test-motion",
        skeleton=Skeleton(joint_names=["pelvis"], parents=None),
        control_mode="position",
        joint_rotations={"actuated_joint_names": names, "angles_rad": angles},
    )


def test_joint_trajectory_extracts_names_and_angles():
    m = _motion(["a", "b"], [[0.1, 0.2], [0.3, 0.4]])
    names, frames = joint_trajectory(m)
    assert names == ["a", "b"]
    assert frames == [[0.1, 0.2], [0.3, 0.4]]


def test_csv_export_shape_and_time_column(tmp_path):
    fps = 30.0
    angles = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    m = _motion(["j0", "j1", "j2"], angles, fps=fps)
    out = export_joint_trajectory(m, tmp_path / "j.csv", fmt="csv")

    with out.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # header + 3 frames
    assert len(rows) == 1 + len(angles)
    assert rows[0] == ["time_s", "j0", "j1", "j2"]
    # 各行: time_s + n_joints 角度
    for i, row in enumerate(rows[1:]):
        assert len(row) == 1 + 3
        assert float(row[0]) == pytest.approx(i / fps, abs=1e-6)
        assert [float(x) for x in row[1:]] == pytest.approx(angles[i])


def test_json_export_metadata_and_frames(tmp_path):
    angles = [[0.1, 0.2], [0.3, 0.4]]
    m = _motion(["hip", "knee"], angles, fps=25.0)
    out = export_joint_trajectory(m, tmp_path / "j.json", fmt="json")

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["format"] == "robotdance.joint_trajectory.v0"
    assert doc["robot"] == "unitree_g1"
    assert doc["control_mode"] == "position"
    assert doc["units"] == "rad"
    assert doc["fps"] == 25.0
    assert doc["n_joints"] == 2
    assert doc["n_frames"] == 2
    assert doc["joint_names"] == ["hip", "knee"]
    assert doc["frames"] == angles


def test_json_roundtrip_preserves_angles(tmp_path):
    angles = [[0.11, -0.22, 0.33], [0.44, 0.55, -0.66]]
    m = _motion(["a", "b", "c"], angles, fps=30.0)
    out = export_joint_trajectory(m, tmp_path / "rt.json", fmt="json")
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["frames"] == angles


def test_mismatched_row_length_raises(tmp_path):
    m = _motion(["a", "b"], [[0.1, 0.2], [0.3]])  # 2 行目が短い
    with pytest.raises(ValueError, match="一致しません"):
        export_joint_trajectory(m, tmp_path / "x.csv", fmt="csv")


def test_missing_angles_raises(tmp_path):
    m = RdMotion(
        robot_name="unitree_g1",
        fps=30.0,
        duration=1.0,
        source_motion_id="t",
        skeleton=Skeleton(joint_names=["pelvis"], parents=None),
        control_mode="position",
        joint_rotations=None,
    )
    with pytest.raises(ValueError, match="angles_rad"):
        export_joint_trajectory(m, tmp_path / "x.csv", fmt="csv")


def test_unknown_format_raises(tmp_path):
    m = _motion(["a"], [[0.1]])
    with pytest.raises(ValueError, match="未知の format"):
        export_joint_trajectory(m, tmp_path / "x.xml", fmt="xml")


def test_cli_export_joints_json(tmp_path):
    from robotdance_core.cli import main

    angles = [[0.1, 0.2], [0.3, 0.4]]
    m = _motion(["hip", "knee"], angles, fps=30.0)
    src = tmp_path / "g1.rdmotion.json"
    m.save(src)

    out = tmp_path / "g1_joints.json"
    rc = main(["export-joints", str(src), "-o", str(out), "--format", "json"])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["joint_names"] == ["hip", "knee"]
    assert doc["frames"] == angles


def test_joint_velocities_finite_difference():
    from robotdance_retarget.sdk_export import joint_velocities

    # 等速の角度列 → 全フレーム同じ角速度（col0=1.0, col1=2.0 rad/s）
    v = joint_velocities([[0.0, 0.0], [0.1, 0.2], [0.2, 0.4]], fps=10.0)
    flat = [x for row in v for x in row]
    assert flat == pytest.approx([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])


def test_csv_export_with_velocity_columns(tmp_path):
    angles = [[0.0, 0.0], [0.1, 0.2], [0.2, 0.4]]
    m = _motion(["j0", "j1"], angles, fps=10.0)
    out = export_joint_trajectory(m, tmp_path / "v.csv", fmt="csv", include_velocity=True)
    with out.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["time_s", "j0", "j1", "d_j0", "d_j1"]
    assert all(len(r) == 1 + 2 + 2 for r in rows[1:])
    assert [float(x) for x in rows[1][3:]] == pytest.approx([1.0, 2.0])


def test_json_export_with_velocity(tmp_path):
    m = _motion(["a", "b"], [[0.0, 0.0], [0.1, 0.2]], fps=10.0)
    out = export_joint_trajectory(m, tmp_path / "v.json", fmt="json", include_velocity=True)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["units"] == "rad, rad/s"
    assert len(doc["velocities"]) == 2


def test_export_without_velocity_omits_it(tmp_path):
    m = _motion(["a"], [[0.1], [0.2]], fps=10.0)
    doc = json.loads(
        export_joint_trajectory(m, tmp_path / "n.json", fmt="json").read_text(encoding="utf-8"))
    assert "velocities" not in doc
    assert doc["units"] == "rad"


def test_cli_export_joints_with_velocity(tmp_path):
    from robotdance_core.cli import main

    m = _motion(["hip", "knee"], [[0.0, 0.0], [0.1, 0.2], [0.2, 0.4]], fps=10.0)
    src = tmp_path / "g.rdmotion.json"
    m.save(src)
    out = tmp_path / "g.csv"
    rc = main(["export-joints", str(src), "-o", str(out), "--with-velocity"])
    assert rc == 0
    with out.open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "d_hip" in header and "d_knee" in header
