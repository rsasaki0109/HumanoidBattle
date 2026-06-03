"""End-to-end pipeline ショーケース（§6, robotdance_core.pipeline）の検証。

core 段（RD-MIR → retarget → motion card）は依存なしで CI 検証。sim/policy は importorskip。
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from robotdance_core.pipeline import run_pipeline

_ROOT = Path(__file__).resolve().parent.parent
_MIR_SCHEMA = json.loads((_ROOT / "specs" / "rd-mir" / "rd-mir.schema.json").read_text())
_MOTION_SCHEMA = json.loads((_ROOT / "specs" / "rd-motion" / "rd-motion.schema.json").read_text())


def test_core_pipeline_without_heavy_deps(tmp_path: Path) -> None:
    """mujoco/torch なしでも RD-MIR → retarget → motion card が通る。"""
    res = run_pipeline(tmp_path, caption="a person dances", do_sim=False, train_policy=False)
    stage_names = [s["stage"] for s in res["stages"]]
    assert stage_names[:2] == ["rd_mir", "retarget"]
    assert "motion_card" in stage_names
    assert (tmp_path / "motion.rdmir.json").exists()
    assert (tmp_path / "motion.rdmotion.json").exists()
    assert (tmp_path / "MOTION_CARD.md").exists()
    jsonschema.Draft202012Validator(_MIR_SCHEMA).validate(
        json.loads((tmp_path / "motion.rdmir.json").read_text()))
    jsonschema.Draft202012Validator(_MOTION_SCHEMA).validate(
        json.loads((tmp_path / "motion.rdmotion.json").read_text()))
    # caption が入口 RD-MIR の semantics に伝播。
    rdmir = json.loads((tmp_path / "motion.rdmir.json").read_text())
    assert rdmir["semantics"]["action_label"] == "a person dances"


def test_pipeline_uses_supplied_mir(tmp_path: Path) -> None:
    from robotdance_core.synthetic import generate_dance

    mir = generate_dance(duration=1.0)
    mir.semantics = {"action_label": "supplied"}
    res = run_pipeline(tmp_path, mir=mir, do_sim=False)
    assert res["stages"][0]["detail"].endswith("caption='supplied'")


def test_pipeline_sim_certificate(tmp_path: Path) -> None:
    """mujoco があれば sim_certificate 段が verdict を付ける。"""
    pytest.importorskip("mujoco")
    res = run_pipeline(tmp_path, do_sim=True, train_policy=False)
    assert res["verdict"] in ("PASS", "REJECT")
    sim_stage = next(s for s in res["stages"] if s["stage"] == "sim_certificate")
    assert sim_stage["ok"] is True


def test_pipeline_with_policy(tmp_path: Path) -> None:
    """torch+mujoco があれば policy 学習 + RD-Policy export まで通る。"""
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    res = run_pipeline(tmp_path, do_sim=True, train_policy=True, iterations=3)
    assert "rd_policy" in res["artifacts"]
    assert (tmp_path / "policy.rdpolicy.json").exists()
    assert (tmp_path / "POLICY_CARD.md").exists()
    assert 0.0 <= res["policy"]["survival_ratio"] <= 1.0
