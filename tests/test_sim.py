"""MuJoCo 物理検証（sim_certificate）の縦スライス。

mujoco 未インストール環境では skip する。
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from robotdance_core.synthetic import generate_backflip, generate_dance  # noqa: E402
from robotdance_retarget.kinematic import retarget  # noqa: E402
from robotdance_sim.mjcf import build_mjcf  # noqa: E402
from robotdance_sim.mujoco_backend import certify, simulate_certificate  # noqa: E402
from robotdance_unitree import get_morphology  # noqa: E402


@pytest.mark.parametrize("robot,total_mass", [("unitree_g1", 35.0), ("unitree_h1", 47.0)])
def test_mjcf_total_mass_is_conserved(robot: str, total_mass: float) -> None:
    """生成 MJCF の総質量が宣言 total_mass に一致する（宣言質量＝実質量）。

    旧実装は pelvis ハブ(3kg)+足 box(0.6kg) を total_mass の上乗せにしており、
    宣言35kg の G1 を実38.6kg(+10.3%) で sim していた。PD ゲインや逆動力学トルクは
    実質量に依存するため、宣言と実体がズレると「35kg 用に調整したつもりが 38.6kg を制御」
    という隠れ取り違えが起きる。固定質量を bone 配分予算から差し引いて質量を保存する。
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(build_mjcf(get_morphology(robot), total_mass=total_mass))
    assert model.body_mass.sum() == pytest.approx(total_mass, abs=1e-3), (
        f"{robot}: 宣言 {total_mass}kg と MJCF 実質量 {model.body_mass.sum():.3f}kg が不一致"
    )


def test_certify_uses_embodiment_torque_limit_not_g1_default() -> None:
    """certify は morphology.sim_defaults のトルク上限を使う（G1 値の固定流用ではない）。

    旧実装は simulate_certificate に torque_limit=80（G1値）をハードコードしており、
    H1（160N·m）の certify でも 80 で torque_ratio を計算していた（配線漏れ）。
    既定経路（torque_limit 未指定）と H1 値を明示した場合の torque_ratio が一致し、
    かつ G1 値を明示した場合とは異なることで、embodiment 由来であることを担保する。
    """
    morph = get_morphology("unitree_h1")
    motion = retarget(generate_dance(duration=1.0), morph)
    default = simulate_certificate(motion, morph)["metrics"]["torque_ratio"]
    h1_explicit = simulate_certificate(motion, morph, torque_limit=160.0)["metrics"]["torque_ratio"]
    g1_default = simulate_certificate(motion, morph, torque_limit=80.0)["metrics"]["torque_ratio"]
    assert default == pytest.approx(h1_explicit), "既定が H1 のトルク上限(160)を使っていない"
    assert default != pytest.approx(g1_default), "既定が G1 のトルク上限(80)に固定されたまま"


@pytest.mark.parametrize("robot", ["unitree_g1", "unitree_h1"])
def test_safe_dance_passes(robot: str) -> None:
    morph = get_morphology(robot)
    motion = retarget(generate_dance(duration=2.0), morph)
    cert = simulate_certificate(motion, morph)
    assert cert["passed"] is True
    assert cert["verdict"] == "PASS"
    # 接地して支持されている。
    assert cert["metrics"]["airborne_ratio"] == 0.0
    # 典型トルクは物理的に妥当（特異姿勢の peak ではなく p50 で判定）。
    assert cert["metrics"]["torque_ratio"] < 1.5


@pytest.mark.parametrize("robot", ["unitree_g1", "unitree_h1"])
def test_backflip_is_rejected(robot: str) -> None:
    morph = get_morphology(robot)
    motion = retarget(generate_backflip(), morph)
    cert = simulate_certificate(motion, morph)
    assert cert["passed"] is False
    assert cert["verdict"] == "REJECT"
    assert cert["reasons"]  # 理由が付く
    # 滞空（接地なし）を検出している。
    assert cert["metrics"]["airborne_ratio"] > 0.5


def test_certify_attaches_to_motion() -> None:
    morph = get_morphology("unitree_g1")
    motion = retarget(generate_dance(duration=1.0), morph)
    assert motion.sim_certificate is None
    certify(motion, morph)
    assert motion.sim_certificate is not None
    assert motion.sim_certificate["backend"] == "mujoco"
    # certificate 付き motion も RD-Motion schema に適合する。
    import json
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        (Path(__file__).resolve().parent.parent / "specs" / "rd-motion" / "rd-motion.schema.json")
        .read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(motion.to_dict())
