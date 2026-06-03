"""Sim backend 抽象 + registry（§4.3）の検証。

registry / contract / Isaac Lab scaffold は依存なしで CI でも走る。MuJoCo dispatch は importorskip。
"""

from __future__ import annotations

import pytest

from robotdance_sim.backend import (
    IsaacLabBackend,
    MujocoBackend,
    backend_names,
    backend_status,
    certify,
    get_backend,
    register_backend,
    simulate_certificate,
)


def test_registry_has_default_backends() -> None:
    names = backend_names()
    assert "mujoco" in names
    assert "isaaclab" in names


def test_backend_status_reports_availability() -> None:
    status = {b["name"]: b["available"] for b in backend_status()}
    assert set(status) == set(backend_names())
    # availability は bool。
    assert all(isinstance(v, bool) for v in status.values())


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="未知の sim backend"):
        get_backend("genesis")


def test_isaaclab_scaffold_errors_clearly() -> None:
    """Isaac Lab 未インストール環境では明示エラー（同梱しない方針）。"""
    backend = IsaacLabBackend()
    if not backend.available():
        with pytest.raises(RuntimeError, match="Isaac Lab"):
            backend.simulate_certificate(None, None)


def test_register_custom_backend() -> None:
    class DummyBackend:
        name = "dummy"

        def available(self) -> bool:
            return True

        def simulate_certificate(self, motion, morphology, **kwargs):
            return {"backend": "dummy", "passed": True, "verdict": "PASS",
                    "metrics": {}, "reasons": []}

    register_backend("dummy", DummyBackend)
    assert "dummy" in backend_names()
    cert = simulate_certificate(None, None, backend="dummy")
    assert cert["verdict"] == "PASS"


def test_contract_validation_rejects_incomplete_certificate() -> None:
    class BadBackend:
        name = "bad"

        def available(self) -> bool:
            return True

        def simulate_certificate(self, motion, morphology, **kwargs):
            return {"verdict": "PASS"}  # 必須キー不足

    register_backend("bad", BadBackend)
    with pytest.raises(ValueError, match="必須キーが不足"):
        simulate_certificate(None, None, backend="bad")


def test_mujoco_backend_dispatch() -> None:
    """MuJoCo backend 経由で certify でき、certificate に backend=mujoco が入る。"""
    pytest.importorskip("mujoco")
    from robotdance_core.synthetic import generate_dance
    from robotdance_retarget.kinematic import retarget
    from robotdance_unitree import get_morphology

    assert MujocoBackend().available()
    morph = get_morphology("unitree_g1")
    motion = retarget(generate_dance(duration=1.0), morph)
    certify(motion, morph, backend="mujoco")
    assert motion.sim_certificate is not None
    assert motion.sim_certificate["backend"] == "mujoco"
    assert motion.sim_certificate["verdict"] in ("PASS", "REJECT")
