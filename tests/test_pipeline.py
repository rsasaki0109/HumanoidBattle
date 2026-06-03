"""synth → RD-MIR → schema validate → GIF render の縦スライスを検証する。"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from robotdance_core.rd_mir import RdMir
from robotdance_core.skeleton import NUM_JOINTS
from robotdance_core.synthetic import generate_dance

_ROOT = Path(__file__).resolve().parent.parent
_MIR_SCHEMA = _ROOT / "specs" / "rd-mir" / "rd-mir.schema.json"


def test_generate_dance_shapes() -> None:
    mir = generate_dance(duration=1.0, fps=30.0)
    assert mir.num_frames == 30
    kps = mir.keypoints_3d_array()
    assert kps.shape == (30, NUM_JOINTS, 3)
    # 接地は両足が存在し、各フレーム bool の列。
    assert set(mir.contacts) == {"left_foot", "right_foot"}
    assert len(mir.contacts["left_foot"]) == 30


def test_generated_mir_conforms_to_schema() -> None:
    mir = generate_dance(duration=1.0, fps=30.0)
    schema = json.loads(_MIR_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(mir.to_dict())


def test_roundtrip_save_load(tmp_path: Path) -> None:
    mir = generate_dance(duration=1.0, fps=30.0)
    p = mir.save(tmp_path / "m.rdmir.json")
    loaded = RdMir.load(p)
    assert loaded.motion_id == mir.motion_id
    assert loaded.keypoints_3d_array().shape == mir.keypoints_3d_array().shape


def test_render_gif(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio")
    from robotdance_viewer.skeleton_view import render_gif

    mir = generate_dance(duration=0.5, fps=20.0)
    out = render_gif(mir, tmp_path / "out.gif", stride=2)
    assert out.exists() and out.stat().st_size > 0
